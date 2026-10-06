"""Architecture instruction-set loading and lookup over the core SASS codec."""
import gzip
import json
import os
from pathlib import Path

from binet.cuda.core.instr import PT, DecodeError, EncodeError, Instr, InstrClass, Unknown, _Mismatch, _extract, _word
from binet.cuda.core.sched import Sched

DB = Path(os.environ.get("BINET_DB_DIR", Path(__file__).resolve().parents[1] / "db"))


def _db_path(directory, name):
    compressed = directory / f"{name}.json.gz"
    return compressed if compressed.is_file() else directory / f"{name}.json"


def _read_json(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8") as source:
        return json.load(source)


class Isa:
    """The instruction set of one architecture, loaded from <arch>.isa.json[.gz]."""

    def __init__(self, db, *, path=None):
        if db.get("schema_version") != 2:
            raise ValueError("unsupported ISA schema; rebuild the database with tools/build_isa.py")
        self.db, self.arch = db, db["arch"]
        self.path, self.source = Path(path).resolve() if path is not None else None, db.get("source", {})
        self._db_dir = self.path.parent if self.path is not None else DB.resolve()
        self._sched = None
        self.enums, self.tables = db["enums"], db["tables"]
        self.enum_values = {t: frozenset(v for _, v in pairs) for t, pairs in self.enums.items()}
        self._values = {t: {n.upper(): v for n, v in pairs} for t, pairs in self.enums.items()}
        self._names = {t: {v: n for n, v in pairs} for t, pairs in self.enums.items()}  # last name wins
        self.max_ureg = db["parameters"]["MAX_UNIFORM_REG_COUNT"]  # UR0 .. UR<max_ureg - 1> exist
        self.zero = {"R": self.enum_value("Register", "RZ"), "UR": self.enum_value("UniformRegister", "URZ"),
                     "P": PT, "UP": PT}
        self.table_outputs = {t: frozenset(out for _, out in rows if isinstance(out, int))
                              for t, rows in self.tables.items()}
        self.table_inverse, self.table_forward = {}, {}
        for t, rows in self.tables.items():
            inverse = self.table_inverse[t] = {}
            for inputs, output in rows:
                inverse.setdefault(output, []).append(inputs)
            if all(all(type(value) is int for value in inputs) for inputs, _ in rows):
                self.table_forward[t] = {tuple(inputs): output for inputs, output in reversed(rows)}  # first row wins
        self.classes = [InstrClass(self, rec) for rec in db["classes"]]
        self.by_name = {c.name: c for c in self.classes}
        layouts = {tuple(f.segs) for c in self.classes for f in c.fields if "opcode" in f.value}
        if len(layouts) != 1:
            raise ValueError(f"{self.arch}: classes place the opcode differently: {layouts}")
        self._opcode_segs = list(layouts.pop())
        self.by_opcode, self.by_mnemonic = {}, {}
        for c in sorted(self.classes, key=lambda c: c.alternate):  # primaries first, then file order
            self.by_opcode.setdefault(c.opcode, []).append(c)
            self.by_mnemonic.setdefault(c.mnemonic, []).append(c)

    def __getattr__(self, mnemonic):
        classes = self.__dict__.get("by_mnemonic", {}).get(mnemonic)
        if classes is None:
            raise AttributeError(mnemonic)
        forms = [c for c in classes if "unsupported" not in c.constructor]
        unsupported = {c.name: c.constructor["unsupported"] for c in classes if "unsupported" in c.constructor}

        def construct(*args, mods=(), guard=None, ctrl=None, form=None):
            selected = [c for c in forms if form is None or c.name == form]
            if not selected:
                detail = unsupported.get(form, "unknown or unsupported instruction form")
                raise EncodeError(f"{mnemonic}: {form!r}: {detail}")
            words, errors = {}, []
            for cls in selected:
                try:
                    for slots in cls.bind(args, mods, guard, form is not None):
                        try:
                            ins = cls._make(slots, ctrl)
                            words.setdefault(ins.encode(), ins)
                        except EncodeError as error:
                            errors.append(str(error))
                except _Mismatch as error:
                    errors.append(str(error))
            if len(words) == 1:
                return next(iter(words.values()))
            if words:
                choices = ", ".join(ins.cls.name for ins in words.values())
                raise EncodeError(f"{mnemonic}: ambiguous encoding ({choices}); specify modifiers or form")
            detail = "; ".join(dict.fromkeys(errors))[:500] or "operands do not match a supported FORMAT"
            raise EncodeError(f"{mnemonic}: no matching encoding: {detail}")

        construct.__name__ = mnemonic
        construct.__doc__ = f"Construct {mnemonic} from typed operands; keywords: mods, guard, ctrl, form."
        setattr(self, mnemonic, construct)
        return construct

    @classmethod
    def load(cls, arch_or_path):
        """Load a bundled architecture or an explicit .isa.json[.gz] path."""
        path = Path(arch_or_path)
        arch = None
        if not path.name.endswith((".json", ".json.gz")):
            arch = str(arch_or_path)
            path = _db_path(DB, f"{arch}.isa")
        isa = cls(_read_json(path), path=path)
        if arch is not None and isa.arch != arch:
            raise ValueError(f"ISA {isa.path} has architecture {isa.arch}, requested {arch}")
        return isa

    @property
    def sched(self):
        """Sibling scheduling database: latency tables and operation sets, loaded on first use."""
        if self._sched is None:
            path = _db_path(self._db_dir, f"{self.arch}.sched")
            sched = Sched(_read_json(path), path=path)
            if sched.arch != self.arch:
                raise ValueError(f"schedule {path} has architecture {sched.arch}; ISA {self.path} uses {self.arch}")
            # The two source hashes identify different files. Current databases have no shared build ID.
            self._sched = sched
        return self._sched

    def make(self, class_name, *, ctrl=None, **slots):
        """Construct an exact class with declared defaults, a true guard and operand flags off.
        Explicit slots override defaults; conflicting ctrl/slot requests raise EncodeError."""
        cls = self.by_name.get(class_name) if isinstance(class_name, str) else None
        if cls is None:
            raise EncodeError(f"{self.arch}: unknown instruction class {class_name!r}")
        return cls._make(slots, ctrl)

    def enum_value(self, etype, name):
        """Encoded value of an enum name (names compare case-insensitively), or None."""
        return self._values.get(etype, {}).get(str(name).upper())

    def enum_name(self, etype, value):
        """Name an enum value prints as: the last name declared for it, or None."""
        return self._names.get(etype, {}).get(value)

    def candidates(self, word):
        """Classes that accept `word`, primaries before alternates."""
        return [c for c in self.by_opcode.get(_extract(word, self._opcode_segs), ()) if c.accepts(word)]

    def decode(self, word):
        """128-bit word -> Instr, or Unknown when no class accepts it."""
        word = _word(word)
        for cls in self.candidates(word):
            slots = cls.decode(word)
            if slots is not None:
                try:
                    if cls._encode(slots) == word:
                        ins = Instr(cls, slots)
                        object.__setattr__(ins, "_word", word)
                        return ins
                except EncodeError:
                    continue
        return Unknown(word)

    def decode_all(self, code):
        """Raw instruction bytes (16 per instruction) -> [Instr | Unknown]."""
        if len(code) % 16:
            raise DecodeError(f"code length {len(code)} is not a multiple of 16")
        return [self.decode(code[i:i + 16]) for i in range(0, len(code), 16)]
