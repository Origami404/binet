"""Annotate register and scoreboard availability from in-flight dependencies over the CFG."""
from binet.cuda.core.instr import SemanticError, Unknown
from binet.cuda.annotations.liveness import bits_of, names_of

# INSTRUCTION_TYPE values whose result has data-dependent latency: it is pending until a scoreboard wait.
DECOUPLED = "INST_TYPE_DECOUPLED"  # every DECOUPLED_* type starts with this


class Claim:
    """A scoreboard claim outstanding at a point: scoreboard `sb` (0..5), `kind` "wr" or "rd", set by instruction
    index `by`, over the register bits `regs`."""
    __slots__ = ("sb", "kind", "by", "regs")

    def __init__(self, sb, kind, by, regs):
        self.sb, self.kind, self.by, self.regs = sb, kind, by, regs

    def __repr__(self):
        return f"<Claim SB{self.sb} {self.kind} by {self.by} {names_of(self.regs)}>"


class Violation:
    """A disagreement between the model and ptxas's control word at instruction `i`."""
    __slots__ = ("i", "kind", "detail")

    def __init__(self, i, kind, detail):
        self.i, self.kind, self.detail = i, kind, detail

    def __repr__(self):
        return f"<Violation {self.kind} at {self.i}: {self.detail}>"


def annotate(cfg, isa):
    """Filter free registers, annotate free scoreboards, and return hazard diagnostics."""
    result = _Hazards(cfg, isa)
    result.run()
    return result


class _Hazards:
    def __init__(self, cfg, isa):
        self.cfg, self.isa = cfg, isa

    def run(self):
        for site in self.cfg.sites:
            site.free_scoreboards = None
        self.sound = not self.cfg.unresolved and not any(isinstance(x, Unknown) for x in self.cfg.instrs)
        self._collect_claims()
        self._solve()
        self._annotate_sites()

    def _collect_claims(self):
        """Extract scoreboard claims and waits from each instruction's control fields."""
        instrs = self.cfg.instrs
        n = len(instrs)
        # per instruction: the write claim it opens, the read claim it opens, and the scoreboards it waits on
        self.opens_wr, self.opens_rd, self.waits = [None] * n, [None] * n, [frozenset()] * n
        self.decoupled = [False] * n
        for i, x in enumerate(instrs):
            if isinstance(x, Unknown):
                continue
            c = x.ctrl
            self.waits[i] = c.wait | self._depbar_drains(x)
            self.decoupled[i] = x.instruction_type.startswith(DECOUPLED)
            if c.wr is not None:
                # a variable-latency result means the instruction executes after it issues, so its sources
                # are captured then, not at issue: they stay unread until this scoreboard is waited
                self.opens_wr[i] = (c.wr, bits_of(x.writes) | bits_of(
                    r for r in x.reads if r.file == "R"))
            if c.rd is not None:
                # a read scoreboard protects the operands read late (store data, async sources): the
                # register operands, never predicates, which every op consumes at issue
                self.opens_rd[i] = (c.rd, bits_of(r for r in x.reads if r.file in ("R", "UR")))

    def _annotate_sites(self):
        for b in self.cfg.blocks:
            for i in range(b.start, b.end):
                site = self.cfg.sites[i]
                if not self.sound or not b.reachable:
                    site.free_regs = None
                    continue
                pending, busy = 0, 0
                for claim in self._in[i]:
                    pending |= claim.regs
                    busy |= 1 << claim.sb
                if site.free_regs is not None:
                    site.free_regs &= ~pending
                site.free_scoreboards = 0x3f & ~busy

    def _depbar_drains(self, x):
        """Scoreboards a DEPBAR.LE sb, 0 fully drains (a partial drain closes nothing)."""
        if x.mnemonic == "DEPBAR" and x.slots.get("cnt") == 0 and "sbidx" in x.slots:
            return frozenset({x.slots["sbidx"]})
        return frozenset()

    def _transfer(self, claims, i):
        """Claims outstanding after instruction i, given those before it: drop everything on a waited scoreboard,
        then add the claims i opens."""
        waited = self.waits[i]
        out = [c for c in claims if c.sb not in waited]
        for opener, kind in ((self.opens_wr[i], "wr"), (self.opens_rd[i], "rd")):
            if opener is not None:
                sb, regs = opener
                out.append(Claim(sb, kind, i, regs))   # claims on one scoreboard accumulate until a wait;
                                                        # ptxas reuses a busy scoreboard (SB1 at 609 over 588's
                                                        # pending SHFL) and one wait drains them all
        return out

    def _solve(self):
        """Forward dataflow of outstanding claims to a fixpoint; `self._in[i]` is the list before instruction i."""
        blocks = self.cfg.blocks
        self._in = [[] for _ in self.cfg.instrs]
        block_in = [[] for _ in blocks]
        changed = True
        while changed:
            changed = False
            for b in blocks:
                incoming = _merge(self._after(blocks[p], block_in[p]) for p in b.preds)
                if not _same(incoming, block_in[b.id]):
                    block_in[b.id], changed = incoming, True
        for b in blocks:
            claims = block_in[b.id]
            for i in range(b.start, b.end):
                self._in[i] = claims
                claims = self._transfer(claims, i)

    def _after(self, b, block_in):
        """Claims outstanding after block b, given those entering it."""
        claims = block_in
        for i in range(b.start, b.end):
            claims = self._transfer(claims, i)
        return claims

    def in_flight(self, i):
        """{register name: Claim} for every register with a write or late read still pending (on a scoreboard)
        before instruction i."""
        out = {}
        for c in self._in[i]:
            for name in names_of(c.regs):
                out.setdefault(name, c)
        return out

    def busy(self, i):
        """Scoreboards (0..5) holding an outstanding claim before instruction i."""
        return {c.sb for c in self._in[i]}

    def free(self, i):
        """Scoreboards no outstanding claim holds before instruction i: a probe there may claim these."""
        if not self.sound:
            raise SemanticError("free scoreboards are unavailable: unresolved control flow or unknown instructions")
        return set(range(6)) - self.busy(i)

    def dependency_constraint(self, producer, consumer, dep, file):
        """Scheduling constraints for two instructions; unknown is not permission to issue immediately."""
        return self.isa.sched.query(producer, consumer, dep, file)

    def check(self):
        """Where ptxas's own control word disagrees with the model (empty when the model is complete)."""
        out = []
        for i, x in enumerate(self.cfg.instrs):
            if isinstance(x, Unknown):
                continue
            held = self.busy(i)
            for sb in self.waits[i]:
                if sb not in held:
                    out.append(Violation(i, "spurious-wait", f"waits SB{sb}, no outstanding claim"))
            flight = self._in[i]
            reads, writes = bits_of(x.reads), bits_of(x.writes)
            for c in flight:
                hazard = (reads & c.regs and c.kind == "wr") or (writes & c.regs)
                if hazard and c.sb not in self.waits[i]:
                    kinds = "RAW" if (reads & c.regs and c.kind == "wr") else ("WAR" if c.kind == "rd" else "WAW")
                    if self.decoupled[c.by]:
                        out.append(Violation(i, f"missing-wait-{kinds}",
                                             f"{names_of(reads & c.regs or writes & c.regs)} pending on SB{c.sb} "
                                             f"from {c.by}, no wait"))
        return out

    def __repr__(self):
        return f"<Hazards {len(self.cfg.instrs)} instructions, {'sound' if self.sound else 'UNSOUND'}>"


def _key(claims):
    return frozenset((c.sb, c.kind, c.by, c.regs) for c in claims)


def _same(a, b):
    return _key(a) == _key(b)


def _merge(claim_lists):
    """Union of claims over paths, de-duplicated: a register in flight on any path stays in flight."""
    seen, out = set(), []
    for claims in claim_lists:
        for c in claims:
            k = (c.sb, c.kind, c.by, c.regs)
            if k not in seen:
                seen.add(k)
                out.append(c)
    return out
