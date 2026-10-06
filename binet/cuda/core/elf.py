"""Read and rebuild little-endian ELF64 images without interpreting processor-specific metadata."""
from struct import Struct, error as StructError
from typing import NamedTuple

EHDR = Struct("<16sHHIQQQIHHHHHH")  # ELF file header.
PHDR = Struct("<IIQQQQQQ")          # Program (segment) header.
SHDR = Struct("<IIQQQQIIQQ")        # Section header.
SYM = Struct("<IBBHQQ")            # Symbol table entry.
REL = Struct("<QQ")                # Relocation without an explicit addend.
RELA = Struct("<QQq")              # Relocation with an explicit signed addend.
WORD = Struct("<I")

SHT_NULL, SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB, SHT_RELA = 0, 1, 2, 3, 4
SHT_NOTE, SHT_NOBITS, SHT_REL, SHT_DYNSYM, SHT_SYMTAB_SHNDX = 7, 8, 9, 11, 18
PT_NULL = 0
# SHN_UNDEF means no defining section; SHN_LORESERVE starts reserved indices.
# SHN_XINDEX redirects to an extended section-index field/table.
# PN_XNUM redirects the program-header count to section zero's sh_info.
SHN_UNDEF, SHN_LORESERVE, SHN_XINDEX, PN_XNUM = 0, 0xff00, 0xffff, 0xffff
STT_NOTYPE, STT_OBJECT, STT_FUNC, STT_SECTION = 0, 1, 2, 3
STB_LOCAL, STB_GLOBAL, STB_WEAK = 0, 1, 2


class ElfError(ValueError):
    pass


class Header(NamedTuple):
    """ELF file header; names omit the standard e_ prefix."""
    ident: bytes  # Magic, ELF class, byte order, version and ABI identification.
    type: int     # Object-file type (ET_*).
    machine: int  # Target instruction set (EM_*).
    version: int
    entry: int    # Entry-point virtual address.
    phoff: int    # Program-header table file offset, in bytes.
    shoff: int    # Section-header table file offset, in bytes.
    flags: int
    ehsize: int     # ELF header size, in bytes.
    phentsize: int  # Program-header entry size, in bytes.
    phnum: int      # Program-header count; PN_XNUM uses section zero's info.
    shentsize: int  # Section-header entry size, in bytes.
    shnum: int      # Section-header count; zero uses section zero's size if present.
    shstrndx: int   # Section-name string-table index; SHN_XINDEX uses section zero's link.


class ProgramHeader(NamedTuple):
    """A segment's file/memory mapping; names omit the standard p_ prefix."""
    type: int  # Segment type (PT_*).
    flags: int
    offset: int  # Segment file offset, in bytes.
    vaddr: int   # Virtual address in memory.
    paddr: int   # Physical address, where applicable.
    filesz: int  # Segment size in the file, in bytes.
    memsz: int   # Segment size in memory, in bytes.
    align: int   # Required file/memory alignment, in bytes (0 or 1 means none).


class Section(NamedTuple):
    """A section header with its table index and resolved name (sh_* in ELF)."""
    index: int  # Position in this image's section-header table.
    name: str   # Resolved from the string table selected by Header.shstrndx.
    type: int   # Section type (SHT_*).
    flags: int
    addr: int    # Virtual address when loaded, or zero for nonloaded sections.
    offset: int  # Section file offset, in bytes.
    size: int    # Byte size; NOBITS has memory size but no file payload.
    link: int    # Related section index: symbol table -> strings; relocations -> symbols.
    info: int    # Type-specific: relocation target section, or first nonlocal symbol index.
    align: int   # sh_addralign: required byte alignment (0 or 1 means none).
    entsize: int  # Fixed-size table entry size, in bytes; zero for other sections.


class Symbol(NamedTuple):
    """A symbol-table entry with resolved name and section index (st_* in ELF)."""
    index: int  # Position in the selected symbol table, not a section index.
    name: str
    value: int  # Value, often a section-relative offset or virtual address.
    size: int   # Symbol's byte size, or zero if unspecified.
    info: int   # Binding in the high 4 bits; symbol type in the low 4 bits.
    other: int  # Visibility in the low 2 bits; remaining bits are ABI-specific.
    shndx: int  # Resolved section index; reserved indices retain their ELF values.
    raw_shndx: int | None = None  # Original st_shndx, possibly the SHN_XINDEX sentinel.

    @property
    def section_index(self):
        """The referenced section, or None for undefined/absolute/reserved symbols."""
        raw = self.shndx if self.raw_shndx is None else self.raw_shndx
        return self.shndx if self.shndx and (raw == SHN_XINDEX or raw < SHN_LORESERVE) else None

    @property
    def type(self):
        return self.info & 0xf

    @property
    def bind(self):
        return self.info >> 4


class Relocation(NamedTuple):
    """A relocation entry with r_info split into symbol index and type."""
    offset: int  # Location to fix: section offset or virtual address, per ELF type.
    type: int    # Processor-specific relocation operation (low 32 bits of r_info).
    sym: int     # Index in the symbol table linked by the relocation section.
    addend: int | None  # REL has no explicit addend; its encoding depends on type.


class Elf:
    def __init__(self, image):
        self.image = bytes(image)
        self._range(0, EHDR.size, "ELF header")
        self.header = h = Header(*EHDR.unpack_from(self.image))
        if h.ident[:7] != b"\x7fELF\x02\x01\x01" or h.version != 1:
            raise ElfError("not a version-1 little-endian ELF64 image")
        if h.ehsize < EHDR.size:
            raise ElfError("ELF header size is too small")
        self._range(0, h.ehsize, "ELF header")
        if h.shoff:
            self._table(h.shoff, 1, h.shentsize, SHDR, "section header zero")
            zero = SHDR.unpack_from(self.image, h.shoff)
            if zero[1] != SHT_NULL:
                raise ElfError("section zero is not SHT_NULL")
            # Extended header fields live in section zero: sh_size holds shnum,
            # sh_link holds shstrndx, and sh_info holds phnum.
            shnum = h.shnum or zero[5]
            if not shnum:
                raise ElfError("section table has no null section")
        else:
            zero, shnum = (0,) * 10, 0
            if h.shnum or h.shstrndx or h.phnum == PN_XNUM:
                raise ElfError("ELF header requires a missing section table")
        shstrndx = zero[6] if h.shstrndx == SHN_XINDEX else h.shstrndx
        phnum = zero[7] if h.phnum == PN_XNUM else h.phnum
        self._table(h.shoff, shnum, h.shentsize, SHDR, "section headers")
        self._raw_sections = tuple(SHDR.unpack_from(self.image, h.shoff + i * h.shentsize)
                                   for i in range(shnum))
        for i, raw in enumerate(self._raw_sections[1:], 1):
            if raw[1] == SHT_NULL:
                continue  # An inactive header's other fields are undefined.
            if raw[1] != SHT_NOBITS:
                self._range(raw[4], raw[5], f"section {i}")
            self._alignment(raw[8], f"section {i}")
        names = b""
        if shstrndx:
            if shstrndx >= shnum or self._raw_sections[shstrndx][1] != SHT_STRTAB:
                raise ElfError("invalid section-name string table index")
            raw = self._raw_sections[shstrndx]
            names = self.image[raw[4]:raw[4] + raw[5]]
        self.sections = tuple(Section(i, self._string(names, r[0], "section name")
                                      if shstrndx and r[1] != SHT_NULL else "", *r[1:])
                              for i, r in enumerate(self._raw_sections))
        if not shstrndx and any(r[0] for r in self._raw_sections if r[1] != SHT_NULL):
            raise ElfError("section names require a section-name string table")
        self._by_name = {}
        for s in self.sections:
            self._by_name.setdefault(s.name, []).append(s)
        self._table(h.phoff, phnum, h.phentsize, PHDR, "program headers")
        self.program_headers = tuple(ProgramHeader(*PHDR.unpack_from(self.image, h.phoff + i * h.phentsize))
                                     for i in range(phnum))
        for p in self.program_headers:
            if p.type == PT_NULL:
                continue
            self._range(p.offset, p.filesz, "program segment")
            self._alignment(p.align, "program segment")

    def _range(self, offset, size, what):
        if offset < 0 or size < 0 or offset > len(self.image) or size > len(self.image) - offset:
            raise ElfError(f"{what} runs past the image")

    def _table(self, offset, count, stride, layout, what):
        if count and (not offset or stride < layout.size):
            raise ElfError(f"{what} has an invalid offset or entry size")
        if count:
            self._range(offset, count * stride, what)

    @staticmethod
    def _alignment(align, what):
        if align and align & (align - 1):
            raise ElfError(f"{what} alignment is not a power of two")

    @staticmethod
    def _string(data, offset, what):
        if not data and offset == 0:
            return ""
        if not 0 <= offset < len(data):
            raise ElfError(f"{what} offset is outside its string table")
        end = data.find(b"\0", offset)
        if end == -1:
            raise ElfError(f"{what} is not terminated in its string table")
        return data[offset:end].decode("utf-8", "replace")

    def section(self, key):
        """Look up by index, unique name, or a Section belonging to this image."""
        if isinstance(key, str):
            found = self._by_name.get(key, ())
            if not found:
                raise KeyError(f"no section {key!r}")
            if len(found) != 1:
                raise ElfError(f"section name {key!r} is ambiguous; use an index")
            return found[0]
        index = key.index if isinstance(key, Section) else key
        if not isinstance(index, int) or not 0 <= index < len(self.sections):
            raise ElfError(f"invalid section index {index!r}")
        section = self.sections[index]
        if isinstance(key, Section) and key != section:
            raise ElfError("section does not belong to this image")
        return section

    def data(self, section):
        s = self.section(section)
        return b"" if s.type in (SHT_NULL, SHT_NOBITS) else self.image[s.offset:s.offset + s.size]

    def _entries(self, section, layout):
        s = self.section(section)
        if s.entsize < layout.size or s.size % s.entsize:
            raise ElfError(f"{s.name!r} has an invalid entry size or table length")
        return tuple(layout.unpack_from(self.image, s.offset + i * s.entsize)
                     for i in range(s.size // s.entsize))

    def symbols(self, section):
        """Read the selected SYMTAB/DYNSYM, resolving strings and extended indices."""
        s = self.section(section)
        if s.type not in (SHT_SYMTAB, SHT_DYNSYM):
            raise ElfError(f"{s.name!r} is not a symbol table")
        strings = self.section(s.link)
        if strings.type != SHT_STRTAB:
            raise ElfError(f"{s.name!r} does not link to a string table")
        raw, names = self._entries(s, SYM), self.data(strings)
        extended = [x for x in self.sections if x.type == SHT_SYMTAB_SHNDX and x.link == s.index]
        indices = ()
        if extended:
            if len(extended) != 1:
                raise ElfError(f"{s.name!r} has multiple extended-index tables")
            indices = self._entries(extended[0], WORD)
            if len(indices) != len(raw):
                raise ElfError(f"{s.name!r} extended-index table has the wrong length")
        out = []
        for i, (name, info, other, shndx, value, size) in enumerate(raw):
            raw_shndx = shndx
            if shndx == SHN_XINDEX:
                if not indices:
                    raise ElfError(f"symbol {i} needs a missing extended-index table")
                shndx = indices[i][0]
                if shndx >= len(self.sections):
                    raise ElfError(f"symbol {i} has an invalid extended section index")
            elif shndx < SHN_LORESERVE and shndx >= len(self.sections):
                raise ElfError(f"symbol {i} has an invalid section index")
            out.append(Symbol(i, self._string(names, name, f"symbol {i} name"), value, size, info, other,
                              shndx, raw_shndx))
        return tuple(out)

    def relocations(self, section):
        """Read REL/RELA records; REL implicit addends remain processor-specific."""
        s = self.section(section)
        if s.type not in (SHT_REL, SHT_RELA):
            raise ElfError(f"{s.name!r} is not a relocation table")
        self.section(s.info)  # sh_info identifies the target (zero is permitted).
        symbols = self.symbols(s.link)
        out = []
        for raw in self._entries(s, RELA if s.type == SHT_RELA else REL):
            offset, info = raw[:2]
            if info >> 32 >= len(symbols):
                raise ElfError(f"{s.name!r} has an invalid relocation symbol index")
            out.append(Relocation(offset, info & 0xffffffff, info >> 32, raw[2] if s.type == SHT_RELA else None))
        return tuple(out)

    def patched_symbols(self, section, changes):
        """Replace {symbol_index: (value, size)} in a selected table's bytes.

        Names, encoded section indices (including SHN_XINDEX) and any extended
        entry bytes stay intact. The associated string/index tables are unchanged.
        """
        s = self.section(section)
        symbols = self.symbols(s)
        out = bytearray(self.data(s))
        for index, (value, size) in changes.items():
            if not isinstance(index, int) or not 0 <= index < len(symbols):
                raise ElfError(f"invalid symbol index {index!r}")
            raw = SYM.unpack_from(out, index * s.entsize)
            try:
                SYM.pack_into(out, index * s.entsize, *raw[:4], value, size)
            except StructError as e:
                raise ElfError(f"invalid value or size for symbol {index}") from e
        return bytes(out)

    def rebuild(self, replacements, *, allow_drop_program_headers=False):
        """Replace existing section bodies, retaining section indices and flags.

        Equal-sized writes preserve every other byte. Resizing aligns and lays
        out sections again; NOBITS retains its logical size and consumes no file
        bytes. Nonempty NOBITS replacements and NULL-section edits are rejected.
        Segment mappings cannot survive relayout without processor knowledge:
        explicit permission is required to drop existing program headers.
        """
        replace = {}
        for key, value in replacements.items():
            s, value = self.section(key), bytes(value)
            if s.type == SHT_NULL or (s.type == SHT_NOBITS and value):
                raise ElfError("cannot replace a NULL section or supply bytes for NOBITS")
            if s.index in replace:
                raise ElfError(f"multiple replacements for section {s.index}")
            replace[s.index] = value
        if all(len(value) == len(self.data(index)) for index, value in replace.items()):
            out = bytearray(self.image)
            for index, value in replace.items():
                s = self.sections[index]
                if s.type != SHT_NOBITS:
                    out[s.offset:s.offset + s.size] = value
            return bytes(out)
        if self.program_headers and not allow_drop_program_headers:
            raise ElfError("resizing requires allow_drop_program_headers=True for this image")
        h = self.header
        headers = [list(raw) for raw in self._raw_sections]
        out, cursor = bytearray(self.image[:h.ehsize]), h.ehsize
        for s in self.sections[1:]:
            if s.type == SHT_NULL:
                continue
            if s.type == SHT_NOBITS:
                headers[s.index][4] = cursor  # Only memory, not file storage, needs alignment.
                continue
            align = max(s.align, 1)
            cursor = (cursor + align - 1) // align * align
            headers[s.index][4] = cursor
            body = replace.get(s.index, self.data(s))
            out += b"\0" * (cursor - len(out)) + body
            headers[s.index][5] = len(body)
            cursor += len(body)
        shoff = (cursor + 7) // 8 * 8
        out += b"\0" * (shoff - len(out))
        if h.phnum == PN_XNUM:
            headers[0][7] = 0
        for i, raw in enumerate(headers):
            out += SHDR.pack(*raw)
            start = h.shoff + i * h.shentsize
            out += self.image[start + SHDR.size:start + h.shentsize]
        h = h._replace(phoff=0, shoff=shoff, phentsize=0, phnum=0)
        out[:EHDR.size] = EHDR.pack(*h)
        return bytes(out)
