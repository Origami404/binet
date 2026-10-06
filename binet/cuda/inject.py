"""Offline kernel injection: buffer setup, probe emission, and image construction."""
from dataclasses import dataclass
from functools import partial
import hashlib
import struct

from binet.cuda.annotations import hazard, liveness, warps
from binet.cuda.core.cfg import CFG
from binet.cuda.core import elf
from binet.cuda.core.cubin import Cubin, HVAL, SVAL, parse_attrs
from binet.cuda.core.instr import PT, SIGNED, Const, Ctrl, Instr, Mem, P, R, UP, UR, Unknown
from binet.cuda.isa import Isa
from binet.records import RECORD_BYTES, depth_checked, metadata as record_metadata, sites_checked

RING_DEPTH = 16


@dataclass(frozen=True)
class InjectedKernel:
    image: bytes
    kernel: str
    arch: str
    source_cubin_sha256: str
    param_count: int
    param_offset: int
    sites: tuple
    ring_depth: int

    @property
    def slot_bytes(self):
        return self.ring_depth * RECORD_BYTES

    def metadata(self, grid, block):
        """Combine this build's raw record contract with one actual launch's geometry."""
        return dict(record_metadata(self.sites, self.ring_depth, grid, block),
                    kernel=self.kernel, arch=self.arch, source_cubin_sha256=self.source_cubin_sha256)


def _record_quad(site):
    mask = site.free_regs or 0
    return next((r for r in range(0, liveness.WIDTH["R"], 4) if (mask >> r) & 15 == 15), None)


def check_site(cfg, site):
    """Return (probeable, reason) for RECORD placement before an original instruction.

    Requires liveness and hazard annotations. Kernel-wide initialization and
    image layout are checked separately when building the injected cubin.
    """
    if site == 0:
        return False, "site 0 is reserved for initialization"
    if not 0 <= site < len(cfg.instrs):
        return False, "out of range"
    x = cfg.instrs[site]
    if cfg.block_of[site] is None or not cfg.blocks[cfg.block_of[site]].reachable:
        return False, "unreachable"
    if x.cls is not None and x.branch in ("exit", "return"):
        return False, f"{x.mnemonic} ends the kernel; a probe must leave an instruction after it"
    attrs = cfg.sites[site]
    if attrs.free_regs is None or attrs.free_scoreboards is None:
        return False, "resource analysis unavailable"
    if not (attrs.free_regs >> liveness.BASE["P"]) & ((1 << liveness.WIDTH["P"]) - 1):
        return False, "no dead predicate for leader election"
    if _record_quad(attrs) is None:
        return False, "no dead 4-aligned GPR quartet for the record"
    if not attrs.free_scoreboards:
        return False, "no free scoreboard to protect record operands"
    return True, None


def inject(cubin, kernel, sites, *, ring_depth=RING_DEPTH):
    """Build all selected original instruction sites, or fail without returning an image."""
    sites, ring_depth = sites_checked(sites), depth_checked(ring_depth)
    cb = cubin if isinstance(cubin, Cubin) else Cubin.load(cubin)
    k, isa = cb.kernel(kernel), Isa.load(cb.arch)
    injection = _Injector(k, isa, ring_depth)
    param_offset = injection.insert(sites)
    return InjectedKernel(image=bytes(injection.image()), kernel=k.name, arch=cb.arch,
                          source_cubin_sha256=hashlib.sha256(cb.image).hexdigest(),
                          param_count=len(k.params), param_offset=param_offset,
                          sites=sites, ring_depth=ring_depth)


# Constant-bank launch ABI and uniform-load instruction. The probe body is shared;
# these are real architecture differences the probe body has to spell out per arch.
_ARCH_FACTS = {
    "sm_90": ((0x0, 0x4, 0x8), (0xc, 0x10, 0x14), "ULDC"),
    "sm_100": ((0x360, 0x364, 0x368), (0x370, 0x374, 0x378), "LDCU"),
    "sm_103": ((0x360, 0x364, 0x368), (0x370, 0x374, 0x378), "LDCU"),
}
AND_LUT = 0xc0                        # ULOP3 truth table for a & b
LOAD_STALL, MATH_STALL, R2UR_STALL, TIMER_STALL = 2, 6, 11, 8
WARP = 32


def _ctrl(stall=MATH_STALL, wait=(), wr=None, rd=None):
    return Ctrl(frozenset(wait), rd, wr, stall, False)


INDIRECT = {"BRX", "BRXU", "JMX", "JMXU"}
POSITION_ATTRS = {"EXIT_INSTR_OFFSETS", "S2RCTAID_INSTR_OFFSETS", "LD_CACHEMOD_INSTR_OFFSETS", "ATOM_SYS_INSTR_OFFSETS",
                  "COOP_GROUP_INSTR_OFFSETS", "ATOMF16_EMUL_INSTR_OFFSETS", "INT_WARP_WIDE_INSTR_OFFSETS",
                  "SYSCALL_OFFSETS", "SW_WAR_MEMBAR_SYS_INSTR_OFFSETS", "STACK_CANARY_TRAP_OFFSETS",
                  "LOCAL_CTA_ASYNC_STORE_OFFSETS"}
RECORD_ATTRS = {"MBARRIER_INSTR_OFFSETS": 4}   # u32s per record; the instruction offset is the first
CODE_ALIGN = 128
KPARAM_FLAGS = 0x1f << 12                        # space 0, cbank 31, log_align 0: what ptxas gives every parameter


def _filler(isa):
    """ptxas's stall extender, `@!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ` with stall 1: the padding word."""
    z = isa.zero["UR"]
    return isa.make("uiadd3__URURUR_URURUR", ctrl=Ctrl(frozenset(), None, None, 1, False),
                    URd=z, UPu=PT, UPv=PT, URa=z, URb=z, URc=z, **{"UPg@not": 1})


class _Injector:
    """One kernel and annotated CFG for probe emission, layout, and relocation."""
    def __init__(self, kernel, isa, ring_depth=RING_DEPTH):
        self.kernel, self.isa, self.cubin = kernel, isa, kernel.cubin
        if self.cubin.arch != isa.arch:
            raise ValueError(f"kernel architecture {self.cubin.arch} differs from ISA {isa.arch}")
        self.ring_depth = depth_checked(ring_depth)
        self.cfg = CFG.of_kernel(kernel, isa).apply(
            partial(liveness.annotate, regcount=kernel.regcount),
            partial(hazard.annotate, isa=isa),
            warps.annotate,
        )
        if isa.arch not in _ARCH_FACTS:
            raise ValueError(f"no probe template for {isa.arch}")
        self.probes = {}        # instruction index -> instructions inserted before it
        self.params = []        # (ordinal, offset from param_base, size) appended
        self._layout = None
        debug = {s.index for s in self.cubin.sections if s.type not in (elf.SHT_REL, elf.SHT_RELA)
                 and s.name.startswith((".debug_", ".zdebug_", ".nv_debug_"))}
        self._debug_sections = debug | {s.index for s in self.cubin.sections
                                       if s.type in (elf.SHT_REL, elf.SHT_RELA) and s.info in debug}

    def _reachable(self):
        return [i for b in self.cfg.blocks if b.reachable for i in range(b.start, b.end)]

    def _free_regs(self, i, file):
        mask = self.cfg.sites[i].free_regs
        return [] if mask is None else [r for r in range(liveness.WIDTH[file])
                                       if mask >> (liveness.BASE[file] + r) & 1]

    def _scoreboard_at(self, i):
        mask = self.cfg.sites[i].free_scoreboards
        return (mask & -mask).bit_length() - 1 if mask else None

    def _uniform_regs(self):
        """Persistent ring state must remain free and untouched throughout reachable code."""
        pool = set(range(min(self.isa.max_ureg, self.isa.zero["UR"])))
        for i in self._reachable():
            pool.intersection_update(self._free_regs(i, "UR"))
            # Even a dead destination can overwrite persistent ring state.
            for r in self.cfg.instrs[i].writes:
                if r.file == "UR":
                    pool.difference_update(range(r.base, r.base + r.count))
        def pair():
            r = next((r for r in sorted(pool, reverse=True) if r % 2 == 0 and r + 1 in pool), None)
            if r is not None:
                pool.difference_update((r, r + 1))
            return r
        base, addr = pair(), pair()
        counter = max(pool) if pool else None
        pool.discard(counter)
        cta_id = max(pool) if pool else None
        return base, counter, addr, cta_id

    def insert(self, sites):
        """Append the buffer pointer and insert initialization plus every selected probe."""
        param_offset = self._add_param()
        if not self.cfg.sites or any(self.cfg.sites[i].free_regs is None or self.cfg.sites[i].free_scoreboards is None
                                     for i in self._reachable()):
            raise ValueError(f"{self.kernel.name}: resource analysis unavailable (unresolved jumps or undecodable words)")
        refused = []
        for site in sites:
            ok, why = check_site(self.cfg, site)
            if not ok:
                refused.append(f"site {site}: {why}")
        if refused:
            raise ValueError("sites were refused: " + "; ".join(refused))
        base, counter, addr, cta_id = self._uniform_regs()
        if base is None or counter is None or addr is None or cta_id is None:
            raise ValueError(f"{self.kernel.name}: not enough free uniform registers for the ring (base pair, counter, address pair, CTA ID)")
        sb = self._scoreboard_at(0)
        scratch = self._free_regs(0, "R")[:4]
        if sb is None or len(scratch) < 4:
            raise ValueError(f"{self.kernel.name}: entry lacks a free scoreboard / 4 scratch GPRs")
        self._insert_probe(0, self._init(param_offset, base, counter, addr, scratch, sb, cta_id))
        for tag, site in enumerate(sites, 1):
            quad = _record_quad(self.cfg.sites[site])
            predicate = self._free_regs(site, "P")[0]
            self._insert_probe(site, self._record(base, counter, addr, quad, tag, self._scoreboard_at(site), cta_id, predicate))
        return param_offset

    def _init(self, param_offset, base, counter, us, scratch, sb, cta_id):
        b, (a, bb, c, d) = self.isa, map(R, scratch)
        base, counter, us, cta_id = map(UR, (base, counter, us, cta_id))
        rz, urz = R(self.isa.zero["R"]), UR(self.isa.zero["UR"])
        (nx, ny, nz), (gx, gy, _), uniform_load = _ARCH_FACTS[self.isa.arch]
        load, math, loaded = _ctrl(LOAD_STALL, wr=sb), _ctrl(), _ctrl(wait=(sb,))
        # Blackwell's LDCU needs a write scoreboard; Hopper's ULDC has fixed latency.
        pointer = getattr(b, uniform_load)(base, Const(0, param_offset), mods=("64",),
                                          ctrl=load if uniform_load == "LDCU" else math)
        first = b.S2R(a, "SR_CTAID.Z", ctrl=load)
        if pointer.ctrl.wr is not None:
            # Drain LDCU before reusing its scoreboard; LOAD_STALL covers claim-to-wait latency.
            first = first.with_ctrl(first.ctrl._replace(wait=frozenset({sb})))
        return [
            pointer,                                         # {base,base+1} = buffer pointer
            first, b.S2R(bb, "SR_CTAID.Y", ctrl=load), b.LDC(c, Const(0, gy), ctrl=load),
            b.IMAD(a, a, c, bb, mods=("U32",), ctrl=loaded),
            b.S2R(bb, "SR_CTAID.X", ctrl=load), b.LDC(c, Const(0, gx), ctrl=load),
            b.IMAD(a, a, c, bb, mods=("U32",), ctrl=loaded),    # a = block_linear
            b.R2UR(cta_id, a, ctrl=_ctrl(R2UR_STALL)),         # snapshot CTA before a becomes warp_global
            b.S2R(c, "SR_TID.Z", ctrl=load), b.S2R(bb, "SR_TID.Y", ctrl=load), b.LDC(d, Const(0, ny), ctrl=load),
            b.IMAD(c, c, d, bb, mods=("U32",), ctrl=loaded),
            b.S2R(bb, "SR_TID.X", ctrl=load), b.LDC(d, Const(0, nx), ctrl=load),
            b.IMAD(c, c, d, bb, mods=("U32",), ctrl=loaded),    # c = tid_linear
            b.LDC(d, Const(0, nx), ctrl=load), b.LDC(bb, Const(0, ny), ctrl=load),
            b.IMAD(d, d, bb, rz, mods=("U32",), ctrl=loaded),
            b.LDC(bb, Const(0, nz), ctrl=load), b.IMAD(d, d, bb, rz, mods=("U32",), ctrl=loaded),  # d = threads per block
            b.IADD3(d, d, WARP - 1, rz, ctrl=math),
            b.SHF(d, rz, 5, d, mods=("R", "U32", "HI"), ctrl=math),  # d = warps per block = (tpb + 31) >> 5
            b.SHF(c, rz, 5, c, mods=("R", "U32", "HI"), ctrl=math),  # c = warp in block = tid_linear >> 5
            b.IMAD(a, a, d, c, mods=("U32",), ctrl=loaded),    # a = warp_global
            b.R2UR(us, a, ctrl=_ctrl(R2UR_STALL)),             # us = warp_global (uniform)
            _filler(self.isa).with_ctrl(Ctrl(frozenset(), None, None, 4, False)),  # cover R2UR's 13-cyc latency (11 + 4 > 13)
            b.UIMAD(base, us, self.ring_depth * RECORD_BYTES, base, mods=("WIDE", "U32"), ctrl=math),
            b.UMOV(counter, urz, ctrl=math),                   # ring counter = 0
        ]

    def _record(self, base, counter, addr, quad, tag, sb, cta_id, predicate):
        # One store, one read barrier: its wait protects kernel GPR reuse and
        # the next probe's address rewrite. The per-warp counter advances once
        # through the uniform datapath, whose guard is UP rather than lane P.
        # ELECT -> first guarded MOV = 6+6+6 = 18 cycles (required 13).
        # Address -> STG = 26; MOV CTA ID -> STG = 14; CS2R -> STG = 8.
        b, math = self.isa, _ctrl()
        base, counter, addr, cta_id = map(UR, (base, counter, addr, cta_id))
        rz, urz = R(self.isa.zero["R"]), UR(self.isa.zero["UR"])
        guard = P(predicate)
        return [
            b.ELECT(guard, urz, ctrl=math),
            b.ULOP3(addr, counter, self.ring_depth - 1, urz, AND_LUT, UP(PT), mods=("LUT",), ctrl=math),
            b.UIMAD(addr, addr, RECORD_BYTES, base, mods=("WIDE", "U32"), ctrl=math),
            b.MOV(R(quad), tag, guard=guard, ctrl=math),
            b.MOV(R(quad + 1), cta_id, guard=guard, ctrl=math),
            b.CS2R(R(quad + 2), "SR_GLOBALTIMERLO", guard=guard, ctrl=_ctrl(TIMER_STALL)),
            b.STG(Mem(rz, uniform=addr, mods=("U32",)), R(quad), mods=("E", "128"), guard=guard, ctrl=_ctrl(rd=sb)),
            b.UIADD3(counter, counter, 1, urz, ctrl=_ctrl(wait=(sb,))),
        ]

    def _insert_probe(self, index, probe):
        """Run `probe` ([Instr]) every time control reaches instruction `index`, before it."""
        if not 0 <= index < len(self.cfg.instrs):
            raise ValueError(f"{self.kernel.name}: no instruction {index}")
        for p in probe:
            if not isinstance(p, Instr) or p.cls.isa is not self.isa:
                raise ValueError("probe instructions must use the selected ISA")
            p.encode()
        self.probes.setdefault(index, []).extend(probe)
        self._layout = None

    def _add_param(self, size=8):
        """Append a kernel parameter of `size` bytes; returns its offset in constant bank 0."""
        k = self.kernel
        used = max([o + s for _, o, s in self.params], default=k.param_size)
        offset = (used + size - 1) // size * size
        ordinal = max([p["ordinal"] for p in k.params], default=-1) + 1 + len(self.params)
        self.params.append((ordinal, offset, size))
        return k.param_base + offset

    @property
    def _moving(self):
        return any(self.probes.values())

    def _validate_relocation(self):
        """Check relocation support; return {MOV index: (immediate slot, continuation index)}."""
        for i, ins in enumerate(self.cfg.instrs):
            if isinstance(ins, Unknown):
                raise ValueError(f"{self.kernel.name}: instruction {i} is undecodable ({ins}); cannot move code")
            if ins.branch == "call" and (ins.target(16 * i) is None or not ins.cls.target_relative):
                raise ValueError(f"{self.kernel.name}: instruction {i}: only direct relative CALL targets are supported")
            if ins.branch == "return" and ("REL" not in dict(ins.modifiers).values() or
                    len([s for s, t in ins.cls.slots.items() if t["type"] in SIGNED]) != 1):
                raise ValueError(f"{self.kernel.name}: instruction {i}: RET has no supported relative immediate anchor")
        for r, where in self._code_relocs():
            raise ValueError(f"{self.kernel.name}: a relocation in {where} at {r.offset:#x} names a code offset "
                              f"of this kernel; those are not moved yet")
        return self._return_literals()

    def _return_literals(self):
        """Locate ptxas's NOINC return literals, bounded by block and register writes."""
        leaders = {0} | {f.value // 16 for f in self.kernel.functions}
        for i, ins in enumerate(self.cfg.instrs):
            if ins.branch is not None:
                leaders.add(i + 1)
            t = ins.target(16 * i)
            if t is not None:
                leaders.add(t // 16)
        out = {}
        for i, ins in enumerate(self.cfg.instrs):
            if ins.branch != "call" or "NOINC" not in dict(ins.modifiers).values():
                continue
            start = max(max(j for j in leaders if j <= i), i - 31)
            matches, overwritten = [], set()
            for j in range(i - 1, start - 1, -1):
                mov = self.cfg.instrs[j]
                dest = {name for reg in mov.writes for name in reg.names()}
                if mov.mnemonic in ("MOV", "UMOV") and not mov.conditional and len(dest) == 1 \
                        and not dest & overwritten and mov.slots.get("PixMaskU04", 15) == 15:
                    matches += [(j, slot) for slot, typ in mov.cls.slots.items()
                                if typ["type"] in SIGNED | {"UImm"} and mov.slots.get(slot) == 16 * (i + 1)]
                overwritten |= dest
            if len(matches) != 1:
                raise ValueError(f"{self.kernel.name}: instruction {i}: CALL.REL.NOINC needs one unambiguous "
                                  f"return-address MOV/UMOV for {16 * (i + 1):#x}; found {len(matches)}")
            j, slot = matches[0]
            out[j] = (slot, i + 1)
        return out

    def _layout_code(self):
        """Return (target, pos, code) over original instruction indices 0..n.

        target[i] starts the probe before i; pos[i] locates i itself. Branches
        take target, while instruction-offset metadata takes pos.
        """
        if self._layout is not None:
            return self._layout
        n, moving = len(self.cfg.instrs), self._moving
        literals = self._validate_relocation() if moving else {}
        target, pos, out = [0] * (n + 1), [0] * (n + 1), []
        for i, ins in enumerate(self.cfg.instrs):
            target[i] = 16 * len(out)
            out += [(p, None) for p in self.probes.get(i, ())]
            pos[i] = 16 * len(out)
            out.append((ins, i))
        target[n] = pos[n] = 16 * len(out)
        fill = _filler(self.isa)
        while 16 * len(out) % CODE_ALIGN:
            out.append((fill, None))
        code = [self._retarget(ins, i, 16 * j, target, literals) if i is not None else ins
                for j, (ins, i) in enumerate(out)]
        self._layout = (target, pos, code)
        return self._layout

    def _code_relocs(self):
        """Relocations into this kernel's code, excluding discarded debug sections."""
        cb, text = self.cubin, self.kernel.section
        for sec in cb.sections:
            if sec.type not in (elf.SHT_REL, elf.SHT_RELA) or not sec.size or sec.index in self._debug_sections:
                continue
            symbols = cb.elf.symbols(sec.link)
            for r in cb.relocations(sec):
                if sec.info == text.index or symbols[r.sym].section_index == text.index:
                    yield r, sec.name

    def _retarget(self, ins, i, new_pc, target, literals):
        """Instruction i re-encoded for its new address, every code offset it names moved."""
        if isinstance(ins, Unknown):
            return ins
        old_pc, n = 16 * i, len(self.cfg.instrs)
        if i in literals:
            slot, continuation = literals[i]
            return ins.replace(**{slot: target[continuation]})
        elif ins.mnemonic in INDIRECT or ins.branch == "return":
            rel = [s for s, t in ins.cls.slots.items() if t["type"] in SIGNED]
            if not rel or new_pc == old_pc:
                return ins
            if len(rel) != 1:
                raise ValueError(f"{ins.cls.name} has {len(rel)} signed immediates; which is the addend?")
            return ins.replace(**{rel[0]: ins.slots[rel[0]] - (new_pc - old_pc)})
        else:
            t = ins.target(old_pc)
            if t is None:
                return ins
            if t % 16 or not 0 <= t // 16 <= n:
                raise ValueError(f"{self.kernel.name}: instruction {i} targets {t:#x}, outside the code")
            return ins.with_target(pc=new_pc, target=target[t // 16])

    def image(self):
        target, pos, code = self._layout_code()
        k, cb, n = self.kernel, self.cubin, len(self.cfg.instrs)
        total = 16 * len(code)

        def mapper(table, what):
            def f(off):
                if off % 16 or not 0 <= off // 16 <= n:
                    raise ValueError(f"{k.name}: {what} {off:#x} is outside the code")
                return table[off // 16]
            return f

        tmap, pmap = mapper(target, "branch target"), mapper(pos, "instruction offset")
        rep = {k.section.index: b"".join(x.to_bytes() for x in code)}
        rep[cb.section(f".nv.info.{k.name}").index] = self._attrs(tmap, pmap)
        c2 = cb.by_name.get(f".nv.constant2.{k.name}")
        if c2 is not None and k.indirect_branches:
            rep[c2.index] = self._tables(cb.data(c2), tmap)
        symbols = {}
        for s in k.functions:
            start, end = tmap(s.value), (s.value + s.size) // 16
            symbols[s.index] = (start, (total if end >= n else target[end]) - start)
        rep[cb.section(".symtab").index] = cb.patched_symtab(symbols)
        if self.params:
            c0 = cb.section(f".nv.constant0.{k.name}")
            data, size = cb.data(c0), max(o + s for _, o, s in self.params)
            if len(data) != k.param_base + k.param_size:
                raise ValueError(f"{c0.name} is {len(data)} bytes, not param_base + param_size; cannot append")
            rep[c0.index] = data + b"\0" * (k.param_base + size - len(data))
        if self._moving:
            rep.update((index, b"") for index in self._debug_sections)
        if self._moving or self.params:
            for sec in cb.sections:
                if sec.mercury and sec.type != elf.SHT_NOBITS and sec.size:
                    rep[sec.index] = b""
        return cb.rebuild(rep)

    def _attrs(self, tmap, pmap):
        """.nv.info.<kernel> with the offset lists moved and the appended parameters declared."""
        k, cb = self.kernel, self.cubin
        data = cb.data(f".nv.info.{k.name}")
        attrs, out, declared = parse_attrs(data), bytearray(), False
        size = max([o + s for _, o, s in self.params], default=k.param_size)
        flags = next((a.u32s()[2] for a in attrs if a.name == "KPARAM_INFO"), KPARAM_FLAGS) & ((1 << 18) - 1)
        for a in attrs:
            raw = data[a.offset:a.offset + 4 + (len(a.payload) if a.fmt == SVAL else 0)]
            if a.name == "KPARAM_INFO" and self.params and not declared:   # ptxas lists them by descending ordinal
                for ordinal, offset, psize in reversed(self.params):
                    out += struct.pack("<BBHIHHI", SVAL, a.type, 12, 0, ordinal, offset, psize << 18 | flags)
                declared = True
            if a.name in POSITION_ATTRS:
                raw = _sval(a, [pmap(v) for v in a.u32s()])
            elif a.name in RECORD_ATTRS:
                u, stride = a.u32s(), RECORD_ATTRS[a.name]
                raw = _sval(a, [pmap(v) if j % stride == 0 else v for j, v in enumerate(u)])
            elif a.name == "INDIRECT_BRANCH_TARGETS":
                u, j = a.u32s(), 0
                while j + 3 <= len(u):
                    count = u[j + 2]
                    u[j] = pmap(u[j])
                    u[j + 3:j + 3 + count] = [tmap(t) for t in u[j + 3:j + 3 + count]]
                    j += 3 + count
                raw = _sval(a, u)
            elif a.name == "CBANK_PARAM_SIZE" and self.params:
                raw = struct.pack("<BBH", HVAL, a.type, size)
            elif a.name == "PARAM_CBANK" and self.params:
                u = a.u32s()
                u[1] = u[1] & 0xffff | size << 16
                raw = _sval(a, u)
            out += raw
        if self.params and not declared:
            raise ValueError(f"{k.name} has no KPARAM_INFO record to append a parameter after")
        return bytes(out)

    def _tables(self, data, tmap):
        """.nv.constant2.<kernel> with each jump table moved.  A table is the run of absolute u32 offsets that spells
        its branch's EIATTR_INDIRECT_BRANCH_TARGETS list in order, with no relocation; only that run is rewritten,
        never a constant that merely equals a code offset."""
        out = bytearray(data)
        for off, ts in self.kernel.indirect_branches.items():
            run = struct.pack(f"<{len(ts)}I", *ts)
            at = [p for p in range(0, len(data) - len(run) + 1, 4) if data[p:p + len(run)] == run]
            if not at:
                raise ValueError(f"{self.kernel.name}: the jump table of the branch at {off:#x} is not in .nv.constant2")
            for p in at:
                out[p:p + len(run)] = struct.pack(f"<{len(ts)}I", *map(tmap, ts))
        return bytes(out)


def _sval(a, u32s):
    return struct.pack("<BBH", SVAL, a.type, 4 * len(u32s)) + struct.pack(f"<{len(u32s)}I", *u32s)
