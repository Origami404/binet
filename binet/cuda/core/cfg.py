"""Build a kernel control-flow graph with basic blocks, edges, and instruction sites."""
from dataclasses import dataclass
from typing import NamedTuple

from binet.cuda.core.instr import SemanticError, Unknown

RECONVERGENCE = {"BSSY", "WARPSYNC"}


@dataclass
class Site:
    """Before an original instruction; None resource masks are unavailable, zero masks have nothing free."""
    index: int
    free_regs: int | None = None  # bit bases: R=0, UR=256, P=512, UP=520 (liveness.BASE)
    free_scoreboards: int | None = None  # bits 0..5
    warp_mask: int = 0xffffffff  # bit w: warp w may reach this site; not a convergence guarantee
    importance: int = 0  # reading priority; annotations.importance.LEVELS describes the scores


class Edge(NamedTuple):
    src: int    # instruction index
    dst: int
    kind: str   # fall | branch | indirect | call | return


class Block:
    __slots__ = ("id", "start", "end", "function", "succs", "preds", "reachable")

    def __init__(self, id, start, end, function):
        self.id, self.start, self.end, self.function = id, start, end, function
        self.succs, self.preds, self.reachable = [], [], False

    def __len__(self):
        return self.end - self.start

    def __repr__(self):
        return f"<Block {self.id} [{self.start}, {self.end}) -> {self.succs}>"


class CFG:
    def __init__(self, instrs, indirect=None, functions=(0,)):
        """`instrs`: the section's decoded instructions (Instr or Unknown); `indirect`: {instruction index of a
        register jump: [target indices]}; `functions`: entry indices of the kernel and its device functions."""
        self.instrs, n = instrs, len(instrs)
        self.sites = [Site(i) for i in range(n)]
        indirect = indirect or {}
        self.unresolved = []
        succs = [[] for _ in range(n)]            # per instruction: [(dst, kind)]
        leaders = {0} | set(functions)
        calls, rets = {}, []                      # callee entry -> [return sites]; RET indices
        for i, ins in enumerate(instrs):
            try:
                guard = None if isinstance(ins, Unknown) else ins.guard
                kind = None if isinstance(ins, Unknown) else ins.branch
                target = None if isinstance(ins, Unknown) else ins.target(16 * i)
                conditional = False if isinstance(ins, Unknown) else ins.conditional
            except SemanticError:
                self.unresolved.append(i)
                succs[i].append((i + 1, "fall"))
                continue
            if isinstance(ins, Unknown) or (guard is not None and guard[0] == 7 and guard[1]):
                succs[i].append((i + 1, "fall"))  # undecodable, or @!PT: never executes
                continue
            if target is not None:
                if target % 16 or not 0 <= target < n * 16:
                    self.unresolved.append(i)
                    target = None
                else:
                    target //= 16
            if kind == "branch":
                if target is not None:
                    succs[i].append((target, "branch"))
                elif i in indirect:
                    succs[i] += [(t, "indirect") for t in indirect[i]]
                else:
                    self.unresolved.append(i)
                if conditional:
                    succs[i].append((i + 1, "fall"))
            elif kind == "call":
                if target is not None:
                    succs[i].append((target, "call"))
                    calls.setdefault(target, []).append(i + 1)
                else:
                    self.unresolved.append(i)
                succs[i].append((i + 1, "fall"))  # the return site
            elif kind == "return":
                rets.append(i)
                if conditional:
                    succs[i].append((i + 1, "fall"))
            elif kind == "exit":
                if conditional:
                    succs[i].append((i + 1, "fall"))
            else:
                succs[i].append((i + 1, "fall"))
                if ins.mnemonic in RECONVERGENCE and target is not None:
                    leaders.add(target)
            if kind is not None:
                leaders.add(i + 1)
            leaders.update(d for d, _ in succs[i] if _ != "fall")
        self.functions = sorted((set(functions) | set(calls) | {0}) & set(range(n)))
        for i in rets:                            # a RET returns to the sites of the calls into its function
            entry = max(f for f in self.functions if f <= i)
            succs[i] += [(site, "return") for site in calls.get(entry, ())]
        # blocks: [leader, next leader); edges from each block's last instruction
        starts = sorted(l for l in leaders if 0 <= l < n)
        self.blocks, self.block_of, self.edges = [], [None] * n, []
        self._ipdom = None                   # postdominators(), computed on demand
        for b, start in enumerate(starts):
            end = starts[b + 1] if b + 1 < len(starts) else n
            self.blocks.append(Block(b, start, end, max(f for f in self.functions if f <= start)))
            for i in range(start, end):
                self.block_of[i] = b
        for blk in self.blocks:
            last = blk.end - 1
            for dst, kind in succs[last]:
                if 0 <= dst < n:
                    self.edges.append(Edge(last, dst, kind))
                    to = self.blocks[self.block_of[dst]]
                    if to.id not in blk.succs:
                        blk.succs.append(to.id)
                    if blk.id not in to.preds:
                        to.preds.append(blk.id)
                elif kind != "fall":
                    self.unresolved.append(last)
        stack = [self.block_of[f] for f in self.functions]
        while stack:
            blk = self.blocks[stack.pop()]
            if not blk.reachable:
                blk.reachable = True
                stack += blk.succs

    @classmethod
    def of_kernel(cls, kernel, isa):
        """The CFG of a binet.cuda.core.cubin.Kernel, decoded with `isa` (Isa.load(cubin.arch))."""
        indirect = {off // 16: [t // 16 for t in targets] for off, targets in kernel.indirect_branches.items()}
        return cls(isa.decode_all(kernel.code), indirect, [f.value // 16 for f in kernel.functions])

    def apply(self, *passes):
        """Run annotation callables in order, ignore their results, and return this CFG."""
        for annotate in passes:
            annotate(self)
        return self

    def block(self, i):
        """The block holding instruction i."""
        return self.blocks[self.block_of[i]]

    def postdominators(self):
        """`ipdom[b]`: the nearest block every path leaving block b must pass through, or None when b can leave
        the graph without passing one.  Reachable blocks only; unreachable blocks get None.

        pdom(b) = {b} | the intersection of pdom(s) over b's reachable successors, and {b} where b has none; the
        immediate post-dominator is the member of pdom(b) - {b} with the largest pdom set, that being the
        nearest.  The intersection is only meaningful for a block that can reach a point with no successor, so a
        block that cannot (an endless loop, or one cut off by an unresolved jump) is given None rather than the
        arbitrary block that falls out of the lattice. This is a graph relationship, not evidence that hardware
        has reconverged its lanes.

        `return` edges make the graph context-insensitive, so a block inside a device function called from two
        sites post-dominates less than it would per call.
        """
        if self._ipdom is None:
            live = [b.id for b in self.blocks if b.reachable]
            S = set(live)
            succs = {b: [s for s in self.blocks[b].succs if s in S] for b in live}
            sink = {b for b in live if not succs[b]}
            drains = set(sink)                            # blocks that can reach a successor-less block
            changed = True
            while changed:                                # over reversed edges, so predecessors of a drain drain
                changed = False
                for b in live:
                    if b not in drains and any(s in drains for s in succs[b]):
                        drains.add(b)
                        changed = True
            pdom = {b: ({b} if b in sink or b not in drains else set(live)) for b in live}
            changed = True
            while changed:
                changed = False
                for b in reversed(live):                  # high to low: a forward edge runs low to high
                    if b in sink or b not in drains:
                        continue
                    new = {b} | set.intersection(*[pdom[s] for s in succs[b]])
                    if new != pdom[b]:
                        pdom[b], changed = new, True
            self._ipdom = [None] * len(self.blocks)
            for b in live:
                if b not in drains:
                    continue
                rest = pdom[b] - {b}
                if rest:
                    self._ipdom[b] = max(rest, key=lambda x: len(pdom[x]))
        return self._ipdom

    def __repr__(self):
        live = sum(b.reachable for b in self.blocks)
        return f"<CFG {len(self.instrs)} instructions, {len(self.blocks)} blocks ({live} reachable), {len(self.edges)} edges>"
