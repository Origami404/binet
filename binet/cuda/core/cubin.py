"""Read and rebuild CUDA cubins, including kernel code, metadata, and relocations."""
import struct
from pathlib import Path
from typing import NamedTuple

from binet.cuda.core import elf

EM_CUDA = 190
SHT_NAMES = {0: "NULL", 1: "PROGBITS", 2: "SYMTAB", 3: "STRTAB", 4: "RELA", 7: "NOTE", 8: "NOBITS", 9: "REL",
             0x70000000: "CUDA_INFO", 0x70000001: "CUDA_CALLGRAPH", 0x70000086: "CUDA_COMPAT_INFO",
             # the Mercury capsule of sm_100+ cubins (names as cuobjdump prints them; *_MERCURY_{RELA,INFO,SYMTAB,
             # CALLGRAPH} are ours: cuobjdump shows those as RELA / CUDA_INFO / SYMTAB / "UNKNOWN: 138")
             0x70000015: "CUDA_RESERVED_SHARED", 0x70000016: "CUDA_CAPMERC", 0x7000007d: "CUDA_MERCURY_CONSTANT_PIC",
             0x70000082: "CUDA_MERCURY_RELA", 0x70000083: "CUDA_MERCURY_INFO", 0x70000084: "CUDA_MERCURY_CONSTANT_OPT",
             0x70000085: "CUDA_MERCURY_SYMTAB", 0x7000008a: "CUDA_MERCURY_CALLGRAPH"}
STO_ENTRY = 0x10  # st_other: a host-launchable kernel
NVAL, BVAL, HVAL, SVAL = 1, 2, 3, 4  # EIATTR record formats: no value, u8 at +2, u16 at +2, u16 size at +2 + payload

# EIATTR type ids (0x66+ as cuobjdump 13.4 names them)
EIATTR = {
    0x01: "PAD", 0x02: "IMAGE_SLOT", 0x03: "JUMPTABLE_RELOCS", 0x04: "CTAIDZ_USED", 0x05: "MAX_THREADS",
    0x06: "IMAGE_OFFSET", 0x07: "IMAGE_SIZE", 0x08: "TEXTURE_NORMALIZED", 0x09: "SAMPLER_INIT", 0x0a: "PARAM_CBANK",
    0x0b: "SMEM_PARAM_OFFSETS", 0x0c: "CBANK_PARAM_OFFSETS", 0x0d: "SYNC_STACK", 0x0e: "TEXID_SAMPID_MAP",
    0x0f: "EXTERNS", 0x10: "REQNTID", 0x11: "FRAME_SIZE", 0x12: "MIN_STACK_SIZE", 0x13: "SAMPLER_FORCE_UNNORMALIZED",
    0x14: "BINDLESS_IMAGE_OFFSETS", 0x15: "BINDLESS_TEXTURE_BANK", 0x16: "BINDLESS_SURFACE_BANK", 0x17: "KPARAM_INFO",
    0x18: "SMEM_PARAM_SIZE", 0x19: "CBANK_PARAM_SIZE", 0x1a: "QUERY_NUMATTRIB", 0x1b: "MAXREG_COUNT",
    0x1c: "EXIT_INSTR_OFFSETS", 0x1d: "S2RCTAID_INSTR_OFFSETS", 0x1e: "CRS_STACK_SIZE", 0x1f: "NEED_CNP_WRAPPER",
    0x20: "NEED_CNP_PATCH", 0x21: "EXPLICIT_CACHING", 0x22: "ISTYPEP_USED", 0x23: "MAX_STACK_SIZE", 0x24: "SUQ_USED",
    0x25: "LD_CACHEMOD_INSTR_OFFSETS", 0x26: "LOAD_CACHE_REQUEST", 0x27: "ATOM_SYS_INSTR_OFFSETS",
    0x28: "COOP_GROUP_INSTR_OFFSETS", 0x29: "COOP_GROUP_MASK_REGIDS", 0x2a: "SW1850030_WAR", 0x2b: "WMMA_USED",
    0x2c: "HAS_PRE_V10_OBJECT", 0x2d: "ATOMF16_EMUL_INSTR_OFFSETS", 0x2e: "ATOM16_EMUL_INSTR_REG_MAP", 0x2f: "REGCOUNT",
    0x30: "SW2393858_WAR", 0x31: "INT_WARP_WIDE_INSTR_OFFSETS", 0x32: "SHARED_SCRATCH", 0x33: "STATISTICS",
    0x34: "INDIRECT_BRANCH_TARGETS", 0x35: "SW2861232_WAR", 0x36: "SW_WAR", 0x37: "CUDA_API_VERSION",
    0x38: "NUM_MBARRIERS", 0x39: "MBARRIER_INSTR_OFFSETS", 0x3a: "COROUTINE_RESUME_ID_OFFSETS",
    0x3b: "SAM_REGION_STACK_SIZE", 0x3c: "PER_REG_TARGET_PERF_STATS", 0x3d: "CTA_PER_CLUSTER", 0x3e: "EXPLICIT_CLUSTER",
    0x3f: "MAX_CLUSTER_RANK", 0x40: "INSTR_REG_MAP", 0x41: "RESERVED_SMEM_USED", 0x42: "RESERVED_SMEM_0_SIZE",
    0x43: "UCODE_SECTION_DATA", 0x44: "UNUSED_LOAD_BYTE_OFFSET", 0x45: "KPARAM_INFO_V2", 0x46: "SYSCALL_OFFSETS",
    0x47: "SW_WAR_MEMBAR_SYS_INSTR_OFFSETS", 0x48: "GRAPHICS_GLOBAL_CBANK", 0x49: "SHADER_TYPE", 0x4a: "VRC_CTA_INIT_COUNT",
    0x4b: "TOOLS_PATCH_FUNC", 0x4c: "NUM_BARRIERS", 0x4d: "TEXMODE_INDEPENDENT", 0x4e: "PERF_STATISTICS",
    0x4f: "AT_ENTRY_FRAGMENTS", 0x50: "SPARSE_MMA_MASK", 0x51: "TCGEN05_1CTA_USED", 0x52: "TCGEN05_2CTA_USED",
    0x53: "GEN_ERRBAR_AT_EXIT", 0x54: "REG_RECONFIG", 0x55: "ANNOTATIONS", 0x56: "SANITIZE",
    0x57: "STACK_CANARY_TRAP_OFFSETS", 0x58: "STUB_FUNCTION_KIND", 0x59: "LOCAL_CTA_ASYNC_STORE_OFFSETS",
    0x5a: "MERCURY_FINALIZER_OPTIONS", 0x5b: "BLOCKS_ARE_CLUSTERS", 0x5c: "SANITIZE", 0x5d: "SYSCALLS_FALLBACK",
    0x5e: "CUDA_REQ", 0x5f: "MERCURY_ISA_VERSION", 0x60: "EXPORTED_FUNCTION", 0x61: "RTCORE_ENTRY",
    0x66: "LANGUAGE", 0x6b: "NVSAL_SW_WAR", 0x6d: "PREEXIT_USED",
}
EIATTR_ID = {name: t for t, name in EIATTR.items()}

# Relocation types: r_type indexes nvdisasm's RELOCATORS table (identical for sm_90, sm_100 and sm_103)
RELOC = (
    "R_CUDA_NONE", "R_CUDA_32", "R_CUDA_64", "R_CUDA_G32", "R_CUDA_G64", "R_CUDA_ABS32_26", "R_CUDA_TEX_HEADER_INDEX",
    "R_CUDA_SAMP_HEADER_INDEX", "R_CUDA_SURF_HW_DESC", "R_CUDA_SURF_HW_SW_DESC", "R_CUDA_ABS32_LO_26", "R_CUDA_ABS32_HI_26",
    "R_CUDA_ABS32_23", "R_CUDA_ABS32_LO_23", "R_CUDA_ABS32_HI_23", "R_CUDA_ABS24_26", "R_CUDA_ABS24_23", "R_CUDA_ABS16_26",
    "R_CUDA_ABS16_23", "R_CUDA_TEX_SLOT", "R_CUDA_SAMP_SLOT", "R_CUDA_SURF_SLOT", "R_CUDA_TEX_BINDLESSOFF13_32",
    "R_CUDA_TEX_BINDLESSOFF13_47", "R_CUDA_CONST_FIELD19_28", "R_CUDA_CONST_FIELD19_23", "R_CUDA_TEX_SLOT9_49", "R_CUDA_6_31",
    "R_CUDA_2_47", "R_CUDA_TEX_BINDLESSOFF13_41", "R_CUDA_TEX_BINDLESSOFF13_45", "R_CUDA_FUNC_DESC32_23",
    "R_CUDA_FUNC_DESC32_LO_23", "R_CUDA_FUNC_DESC32_HI_23", "R_CUDA_FUNC_DESC_32", "R_CUDA_FUNC_DESC_64",
    "R_CUDA_CONST_FIELD21_26", "R_CUDA_QUERY_DESC21_37", "R_CUDA_CONST_FIELD19_26", "R_CUDA_CONST_FIELD21_23",
    "R_CUDA_PCREL_IMM24_26", "R_CUDA_PCREL_IMM24_23", "R_CUDA_ABS32_20", "R_CUDA_ABS32_LO_20", "R_CUDA_ABS32_HI_20",
    "R_CUDA_ABS24_20", "R_CUDA_ABS16_20", "R_CUDA_FUNC_DESC32_20", "R_CUDA_FUNC_DESC32_LO_20", "R_CUDA_FUNC_DESC32_HI_20",
    "R_CUDA_CONST_FIELD19_20", "R_CUDA_BINDLESSOFF13_36", "R_CUDA_SURF_HEADER_INDEX", "R_CUDA_INSTRUCTION64",
    "R_CUDA_CONST_FIELD21_20", "R_CUDA_ABS32_32", "R_CUDA_ABS32_LO_32", "R_CUDA_ABS32_HI_32", "R_CUDA_ABS47_34",
    "R_CUDA_ABS16_32", "R_CUDA_ABS24_32", "R_CUDA_FUNC_DESC32_32", "R_CUDA_FUNC_DESC32_LO_32", "R_CUDA_FUNC_DESC32_HI_32",
    "R_CUDA_CONST_FIELD19_40", "R_CUDA_BINDLESSOFF14_40", "R_CUDA_CONST_FIELD21_38", "R_CUDA_INSTRUCTION128",
    "R_CUDA_YIELD_OPCODE9_0", "R_CUDA_YIELD_CLEAR_PRED4_87", "R_CUDA_32_LO", "R_CUDA_32_HI", "R_CUDA_UNUSED_CLEAR32",
    "R_CUDA_UNUSED_CLEAR64", "R_CUDA_ABS24_40", "R_CUDA_ABS55_16_34", "R_CUDA_8_0", "R_CUDA_8_8", "R_CUDA_8_16", "R_CUDA_8_24",
    "R_CUDA_8_32", "R_CUDA_8_40", "R_CUDA_8_48", "R_CUDA_8_56", "R_CUDA_G8_0", "R_CUDA_G8_8", "R_CUDA_G8_16", "R_CUDA_G8_24",
    "R_CUDA_G8_32", "R_CUDA_G8_40", "R_CUDA_G8_48", "R_CUDA_G8_56", "R_CUDA_FUNC_DESC_8_0", "R_CUDA_FUNC_DESC_8_8",
    "R_CUDA_FUNC_DESC_8_16", "R_CUDA_FUNC_DESC_8_24", "R_CUDA_FUNC_DESC_8_32", "R_CUDA_FUNC_DESC_8_40", "R_CUDA_FUNC_DESC_8_48",
    "R_CUDA_FUNC_DESC_8_56", "R_CUDA_ABS20_44", "R_CUDA_SAMP_HEADER_INDEX_0", "R_CUDA_UNIFIED", "R_CUDA_UNIFIED_32",
    "R_CUDA_UNIFIED_8_0", "R_CUDA_UNIFIED_8_8", "R_CUDA_UNIFIED_8_16", "R_CUDA_UNIFIED_8_24", "R_CUDA_UNIFIED_8_32",
    "R_CUDA_UNIFIED_8_40", "R_CUDA_UNIFIED_8_48", "R_CUDA_UNIFIED_8_56", "R_CUDA_UNIFIED32_LO_32", "R_CUDA_UNIFIED32_HI_32",
    "R_CUDA_ABS56_16_34", "R_CUDA_CONST_FIELD22_37",
)

PARAM_BASE = {90: 0x210, 100: 0x380, 103: 0x380}  # c[0x0][param_base]: the first kernel parameter


class CubinError(elf.ElfError):
    pass


class Section(elf.Section):
    __slots__ = ()

    @property
    def type_name(self):
        return SHT_NAMES.get(self.type, f"0x{self.type:x}")

    @property
    def mercury(self):
        """Part of ptxas's Mercury capsule (sm_100+): a stale clone once the native code changes."""
        return self.name.startswith((".nv.merc.", ".nv.capmerc."))


class Symbol(elf.Symbol):
    __slots__ = ()

    @property
    def is_entry(self):
        return self.type == elf.STT_FUNC and bool(self.other & STO_ENTRY)


class Reloc(elf.Relocation):
    __slots__ = ()

    @property
    def type_name(self):
        return RELOC[self.type] if self.type < len(RELOC) else f"R_CUDA_{self.type}"


class Attr(NamedTuple):
    """One EIATTR record.  `value` is the BVAL / HVAL value, `payload` the SVAL bytes."""
    fmt: int
    type: int
    value: int | None
    payload: bytes
    offset: int   # of the record in its section

    @property
    def name(self):
        return EIATTR.get(self.type, f"0x{self.type:02x}")

    def u32s(self):
        return list(struct.unpack_from(f"<{len(self.payload) // 4}I", self.payload))

    def __repr__(self):
        v = f"0x{self.value:x}" if self.value is not None else self.payload.hex() if self.payload else ""
        return f"<Attr {self.name} {('NVAL', 'BVAL', 'HVAL', 'SVAL')[self.fmt - 1]} {v}>"


def parse_attrs(data, base=0):
    """The EIATTR records of an .nv.info* section's bytes."""
    out, i = [], 0
    while i + 4 <= len(data):
        fmt, typ, v = data[i], data[i + 1], struct.unpack_from("<H", data, i + 2)[0]
        if fmt == SVAL:
            if i + 4 + v > len(data):
                raise CubinError(f"EIATTR record at +0x{base + i:x}: payload of {v} bytes runs past the section")
            out.append(Attr(fmt, typ, None, bytes(data[i + 4:i + 4 + v]), base + i))
            i += 4 + v
        elif fmt in (NVAL, BVAL, HVAL):
            out.append(Attr(fmt, typ, {NVAL: None, BVAL: data[i + 2], HVAL: v}[fmt], b"", base + i))
            i += 4
        else:
            raise CubinError(f"EIATTR record at +0x{base + i:x}: unknown format {fmt}")
    return out


class Kernel:
    """One entry function: its .text section, the FUNC symbols in it, its EIATTRs and relocations."""

    def __init__(self, cubin, symbol):
        self.cubin, self.symbol, self.name = cubin, symbol, symbol.name
        if symbol.section_index is None:
            raise CubinError(f"{self.name}: kernel symbol does not refer to a section")
        self.section = cubin.sections[symbol.section_index]
        self.code = cubin.data(self.section)
        if len(self.code) % 16:
            raise CubinError(f"{self.name}: {self.section.name} is {len(self.code)} bytes, not a multiple of 16")
        self.functions = sorted((s for s in cubin.symbols if s.type == elf.STT_FUNC and s.section_index == symbol.section_index),
                                key=lambda s: s.value)
        self.attrs = cubin.attrs(f".nv.info.{self.name}")
        self.info = [a for a in cubin.attrs(".nv.info")  # the per-function records of .nv.info naming this symbol
                     if a.fmt == SVAL and len(a.payload) >= 8 and a.u32s()[0] == symbol.index]

    # -- EIATTRs -------------------------------------------------------------
    def attrs_of(self, name):
        t = EIATTR_ID.get(name, name)
        return [a for a in self.attrs if a.type == t] + [a for a in self.info if a.type == t]

    def attr(self, name):
        found = self.attrs_of(name)
        return found[0] if found else None

    def _info_u32(self, name):
        a = self.attr(name)
        return a.u32s()[1] if a is not None and a in self.info else None

    def _value(self, name):
        a = self.attr(name)
        return a.value if a is not None else None

    @property
    def regcount(self):
        return self._info_u32("REGCOUNT")

    @property
    def cbank_param_size(self):
        return self._value("CBANK_PARAM_SIZE")

    @property
    def param_base(self):
        """Offset of the first parameter in constant bank 0 (EIATTR_PARAM_CBANK), or the arch's default."""
        a = self.attr("PARAM_CBANK")
        return a.u32s()[1] & 0xffff if a is not None else PARAM_BASE.get(self.cubin.sm)

    @property
    def param_size(self):
        a = self.attr("PARAM_CBANK")
        return a.u32s()[1] >> 16 if a is not None else self.cbank_param_size

    @property
    def params(self):
        """[{ordinal, offset, size, space, log_align, cbank}] from EIATTR_KPARAM_INFO, by ordinal.  `offset` is
        relative to param_base; a __grid_constant__ CUtensorMap has size 128."""
        out = []
        for a in self.attrs_of("KPARAM_INFO"):
            _, pos, flags = a.u32s()[:3]
            out.append({"ordinal": pos & 0xffff, "offset": pos >> 16, "size": flags >> 18,
                        "space": flags >> 8 & 0xf, "log_align": flags & 0xff, "cbank": flags >> 12 & 0x1f})
        return sorted(out, key=lambda p: p["ordinal"])

    @property
    def indirect_branches(self):
        """{byte offset of a BRX / JMX: [target byte offsets]} from EIATTR_INDIRECT_BRANCH_TARGETS."""
        out = {}
        for a in self.attrs_of("INDIRECT_BRANCH_TARGETS"):
            u, i = a.u32s(), 0
            while i + 3 <= len(u):
                offset, _, n = u[i:i + 3]
                out.setdefault(offset, []).extend(u[i + 3:i + 3 + n])
                i += 3 + n
        return out

    def __repr__(self):
        return f"<Kernel {self.name} {len(self.code) // 16} instructions>"


class Cubin:
    """A cubin image.  `sections`, `symbols` and `by_name` are read eagerly; everything else on demand."""

    def __init__(self, data, path=None):
        self.path = path
        try:
            self.elf = elf.Elf(data)
        except elf.ElfError as e:
            raise CubinError(f"{path or 'data'}: {e}") from e
        self.image = self.elf.image
        self.osabi = self.elf.header.ident[7]
        self.e_machine, self.e_flags = self.elf.header.machine, self.elf.header.flags
        if self.e_machine != EM_CUDA:
            raise CubinError(f"{path or 'data'}: e_machine {self.e_machine} is not EM_CUDA ({EM_CUDA})")
        self.sections = tuple(Section(*s) for s in self.elf.sections)
        self.by_name = {s.name: s for s in self.sections}
        try:
            self.symbols = tuple(Symbol(*s) for s in self.elf.symbols(".symtab")) if ".symtab" in self.by_name else ()
        except elf.ElfError as e:
            raise CubinError(f"{path or 'data'}: {e}") from e
        self._kernels = None

    @classmethod
    def load(cls, path):
        return cls(Path(path).read_bytes(), str(path))

    @property
    def sm(self):
        """SM version from e_flags: bits [15:8] in the OSABI 0x41 layout, [7:0] in the old 0x33 one."""
        return self.e_flags >> 8 & 0xff if self.osabi == 0x41 else self.e_flags & 0xff

    @property
    def arch(self):
        return f"sm_{self.sm}"

    # -- ELF structure -------------------------------------------------------
    def section(self, name):
        try:
            return self.sections[self.elf.section(name).index]
        except KeyError:
            raise KeyError(f"{self.path or 'cubin'}: no section {name!r}") from None

    def data(self, section):
        """A section's bytes (empty for NOBITS)."""
        return self.elf.data(section)

    def relocations(self, section):
        """The Reloc entries of a RELA / REL section (given as a Section or by name)."""
        section = self.section(section)
        if section.type not in (elf.SHT_RELA, elf.SHT_REL) or not section.entsize:
            return []
        try:
            return [Reloc(*r) for r in self.elf.relocations(section)]
        except elf.ElfError as e:
            raise CubinError(f"{self.path or 'cubin'}: {e}") from e

    def attrs(self, section):
        """The EIATTR records of .nv.info or .nv.info.<kernel> (empty when the section is absent)."""
        if isinstance(section, str):
            section = self.by_name.get(section)
            if section is None:
                return []
        return parse_attrs(self.data(section))

    # -- kernels -------------------------------------------------------------
    def functions(self):
        """Every FUNC symbol defined in a .text* section, in section order."""
        return sorted((s for s in self.symbols if s.type == elf.STT_FUNC and s.section_index is not None
                       and 0 < s.section_index < len(self.sections) and self.sections[s.section_index].name.startswith(".text")),
                      key=lambda s: (s.section_index, s.value))

    @property
    def kernels(self):
        """{name: Kernel} for every entry (STO_ENTRY FUNC symbol owning .text.<name>), in section order."""
        if self._kernels is None:
            own = [s for s in self.functions() if self.sections[s.section_index].name == f".text.{s.name}"]
            entries = [s for s in own if s.is_entry] or [s for s in own if s.bind == elf.STB_GLOBAL]
            self._kernels = {s.name: Kernel(self, s) for s in entries}
        return self._kernels

    def kernel(self, name=None):
        """The kernel called `name` (exact, or a unique prefix / demangled-name prefix); the only one if None."""
        ks = self.kernels
        if name is None:
            if len(ks) != 1:
                raise KeyError(f"{self.path or 'cubin'}: {len(ks)} kernels, name one of {list(ks)[:8]}")
            return next(iter(ks.values()))
        if name in ks:
            return ks[name]
        found = [k for k in ks if k.startswith(name) or k.startswith(f"_Z{len(name)}{name}")]
        if len(found) == 1:
            return ks[found[0]]
        raise KeyError(f"{self.path or 'cubin'}: kernel {name!r} is " + ("ambiguous: " if found else "unknown; have: ")
                       + ", ".join(found or list(ks)))

    # -- writing -------------------------------------------------------------
    def patched_symtab(self, changes):
        """The .symtab bytes with the (value, size) of the symbols in `changes` ({index: (value, size)}) replaced."""
        return self.elf.patched_symbols(".symtab", changes)

    def rebuild(self, replace):
        """Replace existing section bodies through the ELF writer, retaining their indices.

        Equal-size replacements preserve the rest of the image exactly. Resizing
        drops segment mappings: the CUDA driver, cuobjdump and nvdisasm can load
        cubins from section headers alone. This CUDA policy is explicit here.
        """
        return self.elf.rebuild(replace, allow_drop_program_headers=True)

    def __repr__(self):
        return f"<Cubin {self.path or ''} {self.arch} {len(self.sections)} sections, kernels {list(self.kernels)}>"
