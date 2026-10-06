"""Annotate register liveness and candidate free registers over the CFG."""
from binet.cuda.core.instr import PT, SemanticError, Unknown

FILES = ("R", "UR", "P", "UP")
BASE = {"R": 0, "UR": 256, "P": 512, "UP": 520}   # bit position of register 0 of each file
WIDTH = {"R": 256, "UR": 256, "P": 8, "UP": 8}
RESERVED = {"R": {1}}                              # the stack pointer
TOP_RESERVED = 2                                   # the top two general registers of an allocation fault when written


def bits_of(regs):
    v = 0
    for r in regs:
        v |= ((1 << r.count) - 1) << (BASE[r.file] + r.base)
    return v


def names_of(bits):
    out = []
    for f in FILES:
        v = bits >> BASE[f] & ((1 << WIDTH[f]) - 1)
        while v:
            low = v & -v
            out.append(f"{f}{low.bit_length() - 1}")
            v ^= low
    return out


def unguarded(ins):
    """The default kill policy: the instruction's writes certainly land when it has no guard but PT."""
    g = ins.guard
    return g is None or (g[0] == PT and not g[1])


def annotate(cfg, *, regcount=None, kills=unguarded):
    """Annotate candidate free registers and return liveness queries for diagnostics."""
    result = _Liveness(cfg, regcount, kills)
    result.run()
    return result


class _Liveness:
    def __init__(self, cfg, regcount, kills):
        self.cfg, self.regcount, self.kills = cfg, regcount, kills

    def run(self):
        for site in self.cfg.sites:
            site.free_regs = site.free_scoreboards = None
        self.sound = not self.cfg.unresolved and not any(isinstance(i, Unknown) for i in self.cfg.instrs)
        self._collect_effects()
        self._solve_liveness()
        self._solve_definitions()
        self._solve_register_caps()
        if self.sound:
            for b in self.cfg.blocks:
                if b.reachable:
                    for i in range(b.start, b.end):
                        self.cfg.sites[i].free_regs = self._dead_mask(i)

    def _collect_effects(self):
        """Register reads, possible definitions, and definite kills per instruction."""
        instrs, n = self.cfg.instrs, len(self.cfg.instrs)
        isa = next((i.cls.isa for i in instrs if not isinstance(i, Unknown)), None)
        self.limits = {"R": self.regcount or 255, "UR": min(isa.max_ureg, isa.zero["UR"]) if isa else 63, "P": 7, "UP": 7}
        self.use, self.defs, self.kill = [0] * n, [0] * n, [0] * n
        for i, ins in enumerate(instrs):
            if isinstance(ins, Unknown):
                continue
            g = ins.guard
            if g is not None and g[0] == PT and g[1]:       # @!PT: never executes
                continue
            self.use[i], self.defs[i] = bits_of(ins.reads), bits_of(ins.writes)
            if self.kills(ins):
                self.kill[i] = bits_of(ins.definite_writes)

    def _solve_liveness(self):
        """Solve callee summaries, then propagate live registers backward through calls and returns."""
        cfg, blocks = self.cfg, self.cfg.blocks
        # block successors by edge kind: ordinary flow, the callee of a CALL block, the return sites of a RET block
        flow, rets, callee = [[] for _ in blocks], [[] for _ in blocks], {}
        for e in cfg.edges:
            s, d = cfg.block_of[e.src], cfg.block_of[e.dst]
            if e.kind == "call":
                callee[s] = d
            elif e.kind == "return":
                rets[s].append(d)
            elif d not in flow[s]:
                flow[s].append(d)
        members = {}                                   # function entry block -> its blocks
        for b in blocks:
            members.setdefault(cfg.block_of[b.function], []).append(b.id)
        self.summary = {f: 0 for f in members}         # what a function may read before writing it

        def call_use(bid):
            return self.summary.get(callee[bid], 0) if bid in callee else 0

        def solve(ids, succs):
            buse, bkill = {}, {}                       # upward-exposed uses; registers certainly written
            for bid in ids:
                b, u, k = blocks[bid], 0, 0
                for i in range(b.start, b.end):
                    u |= (self.use[i] | (call_use(bid) if i == b.end - 1 else 0)) & ~k
                    k |= self.kill[i]
                buse[bid], bkill[bid] = u, k
            live_in, live_out, changed = {bid: 0 for bid in ids}, {bid: 0 for bid in ids}, True
            while changed:
                changed = False
                for bid in reversed(ids):
                    out = 0
                    for s in succs(bid):
                        out |= live_in.get(s, 0)
                    new = buse[bid] | (out & ~bkill[bid])
                    if new != live_in[bid] or out != live_out[bid]:
                        live_in[bid], live_out[bid], changed = new, out, True
            return live_in, live_out

        changed = True                                 # callee summaries, to a fixpoint over nested calls
        while changed:
            changed = False
            for f, ids in members.items():
                if f in callee.values():
                    live_in, _ = solve(ids, lambda bid: flow[bid])
                    if live_in[f] != self.summary[f]:
                        self.summary[f], changed = live_in[f], True
        _, live_out = solve(list(range(len(blocks))), lambda bid: flow[bid] + rets[bid])
        self._in, self._out = [0] * len(cfg.instrs), [0] * len(cfg.instrs)
        for b in blocks:
            live = live_out[b.id]
            for i in range(b.end - 1, b.start - 1, -1):
                self._out[i] = live
                live = (self.use[i] | (call_use(b.id) if i == b.end - 1 else 0)) | (live & ~self.kill[i])
                self._in[i] = live

    def _solve_definitions(self):
        """Propagate possible definitions forward over every edge, including calls and returns."""
        blocks = self.cfg.blocks
        bdefs = [0] * len(blocks)
        for b in blocks:
            for i in range(b.start, b.end):
                bdefs[b.id] |= self.defs[i]
        din, dout, changed = [0] * len(blocks), [0] * len(blocks), True
        while changed:
            changed = False
            for b in blocks:
                inn = 0
                for p in b.preds:
                    inn |= dout[p]
                if inn != din[b.id] or inn | bdefs[b.id] != dout[b.id]:
                    din[b.id], dout[b.id], changed = inn, inn | bdefs[b.id], True
        self._defined = [0] * len(self.cfg.instrs)
        for b in blocks:
            v = din[b.id]
            for i in range(b.start, b.end):
                self._defined[i] = v
                v |= self.defs[i]

    def _solve_register_caps(self):
        """Propagate USETMAXREG's shrinking allocation cap; TRY_ALLOC may fail."""
        instrs, blocks = self.cfg.instrs, self.cfg.blocks
        start = self.limits["R"]

        def step(cap, i):
            ins = instrs[i]
            if not isinstance(ins, Unknown) and ins.mnemonic == "USETMAXREG" and "Sb" in ins.slots:
                cap = min(cap, ins.slots["Sb"])
            return cap

        cin, cout, changed = [start] * len(blocks), [start] * len(blocks), True
        while changed:
            changed = False
            for b in blocks:
                inn = min((cout[p] for p in b.preds), default=start)
                out = inn
                for i in range(b.start, b.end):
                    out = step(out, i)
                if inn != cin[b.id] or out != cout[b.id]:
                    cin[b.id], cout[b.id], changed = inn, out, True
        self.cap = [start] * len(instrs)
        for b in blocks:
            cap = cin[b.id]
            for i in range(b.start, b.end):
                self.cap[i] = cap
                cap = step(cap, i)

    def live_in(self, i):
        """Names of the registers live before instruction i: some later instruction may read them."""
        return names_of(self._in[i])

    def live_out(self, i):
        return names_of(self._out[i])

    def defined_in(self, i):
        """Names of the registers some path from the entry has written before instruction i."""
        return names_of(self._defined[i])

    def _dead_mask(self, i):
        limits = dict(self.limits, R=min(self.limits["R"], self.cap[i]) - TOP_RESERVED)
        allowed = 0
        for file, limit in limits.items():
            mask = (1 << max(0, limit)) - 1
            for r in RESERVED.get(file, ()):
                mask &= ~(1 << r)
            allowed |= mask << BASE[file]
        return allowed & ~self._in[i]

    def dead(self, i):
        """{file: [register numbers]} not live before instruction i, within what the kernel owns there, R1 and the
        allocation's top two general registers excepted.  This is program liveness only: a register listed here
        has no later reader, but may still have a write or late read in flight (binet.cuda.annotations.hazard).  A probe may
        take a register only when it is dead here AND not in flight."""
        if not self.sound:
            raise SemanticError("dead registers are unavailable: unresolved control flow or unknown instructions")
        free = self._dead_mask(i)
        return {f: [r for r in range(self.limits[f]) if free >> (BASE[f] + r) & 1] for f in FILES}

    def __repr__(self):
        return f"<Liveness {len(self.cfg.instrs)} instructions, limits {self.limits}, {'sound' if self.sound else 'UNSOUND'}>"
