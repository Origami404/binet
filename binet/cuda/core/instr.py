"""Instruction classes, operands, and lossless 128-bit SASS encoding."""
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import NamedTuple

SIGNED = {"SImm", "RSImm"}  # immediates stored two's complement in their field
CALLS = {"ConstBankAddress0", "ConstBankAddress2", "IDENTICAL"}
BRANCH_TYPES = {"BRT_BRANCH": "branch", "BRT_CALL": "call", "BRT_RETURN": "return", "BRT_BRANCHOUT": "exit"}
CODE_ADDRESS_SLOTS = {"BSSY": "Sa", "WARPSYNC": "sImm"}  # reconvergence points: code addresses of non-branches
PT = 7  # the true predicate, PT / UPT
WRITES = {"Rd", "Rd2", "URd", "URd2", "Pu", "Pv", "UPu", "UPv"}  # destination slots (the writer ports of the TRUE tables)
WHOLE_FILE_WRITERS = {"R2P", "UR2UP"}  # their PR / UPR operand is written under a mask; P2R / UP2UR read it


class Reg(NamedTuple):
    """`count` consecutive registers of `file` ("R", "UR", "P", "UP") from `base`: R4:R5 for a 64-bit operand."""
    file: str
    base: int
    count: int = 1

    def names(self):
        return [f"{self.file}{self.base + k}" for k in range(self.count)]

    def __str__(self):
        return self.names()[0] if self.count == 1 else f"{self.file}{self.base}:{self.file}{self.base + self.count - 1}"


def R(index):
    return Reg("R", index)


def UR(index):
    return Reg("UR", index)


def P(index):
    return Reg("P", index)


def UP(index):
    return Reg("UP", index)


@dataclass(frozen=True)
class Mem:
    """[base + uniform + offset], with optional address modifiers."""
    base: Reg | None = None
    offset: int = 0
    uniform: Reg | None = None
    mods: tuple = ()


@dataclass(frozen=True)
class Const:
    """c[bank][base + offset], or cx[UR bank][base + offset]."""
    bank: int | Reg
    offset: int = 0
    base: Reg | None = None
    mods: tuple = ()


class DecodeError(ValueError):
    pass


class EncodeError(ValueError):
    pass


class SemanticError(ValueError):
    """An instruction is encodable, but the requested architectural fact is unavailable."""


class _Mismatch(ValueError):
    pass


def _word(word):
    if isinstance(word, (bytes, bytearray)):
        if len(word) != 16:
            raise DecodeError(f"instruction must be exactly 16 bytes, got {len(word)}")
        word = int.from_bytes(word, "little")
    if type(word) is not int or not 0 <= word < 1 << 128:
        raise DecodeError("instruction must be an integer in [0, 2**128) or exactly 16 bytes")
    return word


def _checked(value, width, signed, label):
    low, high = (-(1 << (width - 1)), 1 << (width - 1)) if signed else (0, 1 << width)
    if type(value) is not int or not low <= value < high:
        raise EncodeError(f"{label}: {value!r} is outside [{low}, {high})")
    return value & ((1 << width) - 1)


def _extract(word, segs):
    value = 0
    for lo, width in segs:
        value = value << width | (word >> lo) & ((1 << width) - 1)
    return value


def _insert(word, segs, total, value):
    rest = total
    for lo, width in segs:
        rest -= width
        mask = (1 << width) - 1
        word = word & ~(mask << lo) | ((value >> rest) & mask) << lo
    return word


def _slot_key(value):
    """Value object -> key of the slot it reads ('Ra', or 'Ra@negate' for an attribute)."""
    return value["slot"] + ("@" + value["attr"] if "attr" in value else "")


class _Field:
    __slots__ = ("name", "segs", "width", "value")

    def __init__(self, rec):
        self.name, self.value = rec["field"], rec["value"]
        self.segs = [(lo, hi - lo + 1) for hi, lo in rec["bits"]]  # MSB first
        self.width = sum(width for _, width in self.segs)
        if any(lo < 0 or width <= 0 or lo + width > 128 for lo, width in self.segs):
            raise ValueError(f"field {self.name}: invalid 128-bit field layout")
        fixed = self.value.get("const", self.value.get("opcode"))
        if ("const" in self.value or "opcode" in self.value) and (type(fixed) is not int or not 0 <= fixed < 1 << self.width):
            raise ValueError(f"field {self.name}: fixed value {fixed!r} does not fit")


class InstrClass:
    """One CLASS / ALTERNATE CLASS of the ISA, prepared for decoding and encoding."""

    def __init__(self, isa, rec):
        self.isa, self.rec = isa, rec
        self.name, self.mnemonic, self.opcode = rec["name"], rec["mnemonic"], rec["opcode"]
        names = rec.get("opcode_names")
        self.opcode_names = frozenset(names) if isinstance(names, list) and names and all(isinstance(n, str) for n in names) else None
        self.alternate = rec["alternate"]
        self.mask, self.value = (int(x, 16) for x in rec["fixed"])
        self.slots, self.constructor = rec["slots"], rec["constructor"]
        self.fields = [_Field(f) for f in rec["encoding"]]
        self.keys = set(self.slots)
        for f in self.fields:
            for arg in ([f.value] if "slot" in f.value else f.value.get("args", ())):
                if "slot" in arg:
                    if arg["slot"] not in self.slots:
                        raise ValueError(f"{self.name}: field {f.name} refers to unknown slot {arg['slot']}")
                    self.keys.add(_slot_key(arg))
        self.direct = [f for f in self.fields if "slot" in f.value]
        self.tables = [f for f in self.fields if "table" in f.value]
        self.calls = {}  # (function, args) -> fields, one per output
        for f in self.fields:
            if "call" in f.value:
                if f.value["call"] not in CALLS:
                    raise ValueError(f"{self.name}: no decoder for {f.value['call']}()")
                self.calls.setdefault((f.value["call"], json.dumps(f.value["args"])), []).append(f)
        self.checks = []  # (field, allowed values) every decoded word must satisfy
        for f in self.fields:
            v = f.value
            if "table" in v:
                if any(len(row) != len(v["args"]) or type(out) is not int or not 0 <= out < 1 << f.width
                       for row, out in isa.tables[v["table"]]):
                    raise ValueError(f"{self.name}: invalid rows for table {v['table']} / field {f.name}")
                self.checks.append((f, isa.table_outputs[v["table"]]))
            elif "slot" in v and "attr" not in v and "convert" not in v:
                allowed = isa.enum_values.get(self.slots.get(v["slot"], {}).get("type"))
                if allowed is not None and "scale" not in v and "multiply" not in v:
                    self.checks.append((f, allowed))
        self.defaults, self.initial = rec["defaults"], rec["initial"]
        props = rec["properties"]
        self.branch = BRANCH_TYPES.get(props.get("BRANCH_TYPE"))
        self.instruction_type = props.get("INSTRUCTION_TYPE")
        self.indexed_register = next((n for n, t in self.slots.items() if t["type"] == "RF"), None)
        self.guard_slot = guard = rec["guard"]
        self.predicates = [n for n, t in self.slots.items() if t["type"] in ("Predicate", "UniformPredicate") and n != guard]
        # the slot holding a code address: the BRANCH_TARGET_INDEX operand (INDEX(sImm) -> "sImm"), or a
        # reconvergence point (BSSY, WARPSYNC.COLLECTIVE); relative when its immediate is signed or .REL is set
        index = props.get("BRANCH_TARGET_INDEX")
        declared = "BRANCH_TARGET_INDEX" in props
        valid = not declared or (isinstance(index, str) and index.startswith("INDEX(")
                                and index.endswith(")") and index[6:-1].isidentifier())
        self.target_slot = index[6:-1] if declared and valid else CODE_ADDRESS_SLOTS.get(self.mnemonic)
        target = self.slots.get(self.target_slot)
        kind = target["type"] if target is not None else None
        self.target_relative = kind in SIGNED or any(t["type"] == "RelOpt" for t in self.slots.values())
        self.target_kind = ("none" if target is None else "direct" if kind in SIGNED | {"UImm"}
                            else "indirect" if target.get("file") in ("R", "UR") or kind in ("C", "CX") else None)
        self.target_error = None
        if not valid:
            self.target_error = f"invalid BRANCH_TARGET_INDEX {index!r}"
        elif declared and target is None:
            self.target_error = f"declared code target {self.target_slot} is unavailable"
        elif self.target_kind is None:
            self.target_error = f"unsupported code target {self.target_slot} type {kind}"
        # register operands: (slot, file, written, size predicate, whole file).  A GPR / UR operand's width comes
        # from ILABEL_<slot>_SIZE, else IDEST[2]_SIZE for a destination and ISRC_<A..E>_SIZE for a source.
        self.operands = []
        for name, t in self.slots.items():
            file = t.get("file")
            if file is None:
                continue
            whole = t["type"] in ("PR", "UPRONLY")
            written = name in WRITES or name == "Pnz" or (whole and self.mnemonic in WHOLE_FILE_WRITERS)
            source = name.removeprefix("UR").removeprefix("R").upper()
            keys = (f"ILABEL_{name}_SIZE", ("IDEST2_SIZE" if name.endswith("2") else "IDEST_SIZE") if written
                    else f"ISRC_{source}_SIZE")
            self.operands.append((name, file, written, next((k for k in keys if k in rec["predicates"]), None), whole))

    def accepts(self, word):
        """Fixed bits match and every field holds a value this class can produce."""
        return word & self.mask == self.value and all(_extract(word, f.segs) in allowed for f, allowed in self.checks)

    # -- bits -> slots -------------------------------------------------------
    def decode(self, word):
        """Slot values of `word`, or None when its fields contradict this class."""
        slots = {}

        def assign(key, value):
            if slots.get(key, value) != value:
                return False
            slots[key] = value
            return True

        for f in self.direct:
            v, raw = f.value, _extract(word, f.segs)
            if "attr" not in v and "convert" not in v:
                if self.slots.get(v["slot"], {}).get("type") in SIGNED and raw >> (f.width - 1):
                    raw -= 1 << f.width
                raw = raw * v.get("scale", 1) // v.get("multiply", 1)
            if not assign(_slot_key(v), raw):
                return None
        for f in self.tables:
            v, raw = f.value, _extract(word, f.segs)
            row = next((r for r in self.isa.table_inverse[v["table"]].get(raw, ())
                        if all(self._cell_fits(cell, arg, slots) for cell, arg in zip(r, v["args"]))), None)
            if row is None:
                return None
            for cell, arg in zip(row, v["args"]):
                if "slot" in arg and isinstance(cell, int) and not assign(_slot_key(arg), cell):
                    return None
        for (fn, _), fields in self.calls.items():
            args = fields[0].value["args"]
            outs = {f.value.get("out", 0): _extract(word, f.segs) for f in fields}
            if fn == "IDENTICAL":  # one field feeds both slots
                ok = all(assign(_slot_key(a), outs[0]) for a in args)
            else:  # ConstBankAddressN(bank, addr): the offset field holds addr >> N
                ok = assign(_slot_key(args[0]), outs[0]) and assign(_slot_key(args[1]), outs[1] << int(fn[-1]))
            if not ok:
                return None
        for name, value in self.defaults.items():
            slots.setdefault(name, value)
        return slots

    @staticmethod
    def _cell_fits(cell, arg, slots):
        if cell is None or not isinstance(cell, int):
            return True
        if "const" in arg:
            return cell == arg["const"]
        return slots.get(_slot_key(arg), cell) == cell

    # -- slots -> bits -------------------------------------------------------
    def _encode(self, slots):
        """Validate slots and encode fields; decoding uses this without decoding its result again."""
        unknown = slots.keys() - self.keys
        if unknown:
            raise EncodeError(f"{self.name}: unknown slots {sorted(unknown, key=str)}")
        for name in self.keys:
            value = slots.get(name)
            if value is None:
                raise EncodeError(f"{self.name}: slot {name} has no value")
            if type(value) is not int:
                raise EncodeError(f"{self.name}: slot {name} requires an integer, got {value!r}")
            spec = self.slots.get(name, {})
            allowed = self.isa.enum_values.get(spec.get("type"))
            if allowed is not None and value not in allowed:
                raise EncodeError(f"{self.name}: invalid {spec['type']} value {value} for slot {name}")
            if "width" in spec:
                _checked(value, spec["width"], spec.get("signed", False), f"{self.name}: slot {name}")
            if name in ("src_rel_sb", "dst_wr_sb") and value != _SB_NONE and not 0 <= value < 6:
                raise EncodeError(f"{self.name}: slot {name} has unsupported scoreboard {value}")
        word = 0
        for f in self.fields:
            word = _insert(word, f.segs, f.width, self._field_value(f, slots))
        return word

    def encode(self, slots):
        slots = {**self.defaults, **slots}
        word = self._encode(slots)
        canonical = self.decode(word)
        if canonical is None or not self.accepts(word):
            raise EncodeError(f"{self.name}: conflicting values for shared encoding fields")
        for name, value in slots.items():
            if canonical.get(name) != value:
                raise EncodeError(f"{self.name}: slot {name}={value} cannot be preserved (decodes as {canonical.get(name)!r})")
        return word

    def _field_value(self, f, slots):
        v = f.value
        if "const" in v or "opcode" in v:
            return v.get("const", v.get("opcode"))
        if "slot" in v:
            value = slots[_slot_key(v)]
            signed = False
            if "attr" not in v and "convert" not in v:
                value, remainder = divmod(value * v.get("multiply", 1), v.get("scale", 1))
                if remainder:
                    raise EncodeError(f"{self.name}: slot {_slot_key(v)} is not aligned for field {f.name}")
                signed = self.slots[v["slot"]].get("signed", False)
            return _checked(value, f.width, signed, f"{self.name}: slot {_slot_key(v)} / field {f.name}")
        if "table" in v:
            key = [a["const"] if "const" in a else slots[_slot_key(a)] for a in v["args"]]
            exact = self.isa.table_forward.get(v["table"])
            if exact is not None:
                output = exact.get(tuple(key))
                if output is not None:
                    return output
            else:
                for inputs, output in self.isa.tables[v["table"]]:
                    if all(c is None or not isinstance(c, int) or c == k for c, k in zip(inputs, key)):
                        return output
            labels = [str(a["const"]) if "const" in a else _slot_key(a) for a in v["args"]]
            raise EncodeError(f"{self.name}: {v['table']} has no row for {dict(zip(labels, key))}")
        args = [slots[_slot_key(a)] for a in v["args"]]
        if v["call"] == "IDENTICAL":
            if len(set(args)) != 1:
                raise EncodeError(f"{self.name}: {', '.join(map(_slot_key, v['args']))} must be equal")
            return _checked(args[0], f.width, False, f"{self.name}: field {f.name}")
        shift = int(v["call"][-1])  # ConstBankAddressN
        if args[1] < 0 or args[1] % (1 << shift):
            raise EncodeError(f"{self.name}: slot {_slot_key(v['args'][1])} must be nonnegative and aligned to {1 << shift}")
        return _checked(args[0] if v.get("out", 0) == 0 else args[1] >> shift,
                        f.width, False, f"{self.name}: field {f.name}")

    def _control_slots(self, ctrl):
        if not isinstance(ctrl, Ctrl):
            raise EncodeError(f"{self.name}: ctrl must be a Ctrl value")
        slots = {}
        for key, value in ctrl.slots().items():
            if key in self.slots:
                slots[key] = value
            elif key not in _CTRL_NONE or value != _CTRL_NONE[key]:
                raise EncodeError(f"{self.name} has no {key} field")
        return slots

    def _make(self, slots, ctrl=None):
        full = {**self.initial, **slots}
        if ctrl is not None:
            for key, value in self._control_slots(ctrl).items():
                if key in slots and slots[key] != value:
                    raise EncodeError(f"{self.name}: slot {key} conflicts with ctrl")
                full[key] = value
        ins = Instr(self, full)
        ins.encode()
        return ins

    def predicate(self, name, slots):
        """Value of one of the class's PREDICATES (IDEST_SIZE, ...) for these slot values; None when it has none."""
        p = self.rec["predicates"].get(name)
        if isinstance(p, dict):
            return p["map"].get(",".join(str(slots.get(b)) for b in p["by"]))
        return p

    def _value(self, name, value):
        spec = self.slots[name]
        file = spec.get("file")
        if file is not None:
            if not isinstance(value, Reg) or value.file != file or type(value.base) is not int or type(value.count) is not int or value.count != 1:
                raise _Mismatch(f"{name} requires one {file} register")
            return value.base
        if "signed" in spec:
            if type(value) is not int:
                raise _Mismatch(f"{name} requires an integer")
            return value
        if isinstance(value, str):
            encoded = self.isa.enum_value(spec["type"], value)
            if encoded is not None:
                return encoded
        raise _Mismatch(f"{name} requires a {spec['type']} enum name")

    def _mods(self, names, requested, explicit):
        if not isinstance(requested, (tuple, list)):
            raise _Mismatch("mods must be a tuple or list of enum names")
        result = {}
        for name in requested:
            if not isinstance(name, str):
                raise _Mismatch("modifiers must be enum names")
            matches = [(n, self.isa.enum_value(self.slots[n]["type"], name)) for n in names]
            matches = [(n, value) for n, value in matches if value is not None]
            if len(matches) != 1:
                raise _Mismatch(f"unknown or ambiguous modifier {name!r}")
            slot, value = matches[0]
            if slot in result and result[slot] != value:
                raise _Mismatch(f"conflicting modifiers for {slot}")
            result[slot] = value
        if not explicit:
            missing = [n for n in names if "default" not in self.slots[n] and n not in result]
            if missing:
                raise _Mismatch(f"explicit modifiers required for {', '.join(missing)}")
        return result

    def _bind_operand(self, operand, value, explicit):
        if operand["kind"] == "scalar":
            result = self._mods(operand["mods"], (), explicit)
            result[operand["slot"]] = self._value(operand["slot"], value)
            return result
        expected = Const if operand["kind"] == "const" else Mem
        if not isinstance(value, expected):
            raise _Mismatch(f"expected {expected.__name__} address")
        result = self._mods(operand["mods"], value.mods, explicit)
        if "bank" in operand:
            bank = operand["bank"]
            result[bank] = self._value(bank, value.bank)
        supplied = [value.base, value.uniform if isinstance(value, Mem) else None]
        for i, name in enumerate(operand["regs"]):
            if supplied[i] is not None:
                result[name] = self._value(name, supplied[i])
            elif name not in self.defaults:
                raise _Mismatch("address base is required")
        if any(v is not None for v in supplied[len(operand["regs"]):]):
            raise _Mismatch("too many address registers")
        offset = operand["offset"]
        if offset is not None:
            result[offset] = self._value(offset, value.offset)
        elif type(value.offset) is not int or value.offset != 0:
            raise _Mismatch("address has no offset")
        return result

    def bind(self, args, mods, guard, explicit=False):
        initial = self._mods(self.constructor["mods"], mods, explicit)
        if guard is not None:
            initial[self.guard_slot] = self._value(self.guard_slot, guard)
        states = [(0, initial)]
        for operand in self.constructor["operands"]:
            following = []
            for i, slots in states:
                if operand["optional"]:
                    following.append((i, slots))
                if i < len(args):
                    try:
                        following.append((i + 1, {**slots, **self._bind_operand(operand, args[i], explicit)}))
                    except _Mismatch:
                        pass
            states = following
        return [slots for i, slots in states if i == len(args)]

    def __repr__(self):
        return f"<InstrClass {self.name}>"


_CTRL_NONE = {"req_bit_set": 0, "src_rel_sb": 7, "dst_wr_sb": 7}  # what a class without the slot encodes
_SB_NONE = 7


class Ctrl(NamedTuple):
    """The scheduling fields of an instruction's control word, [wait, rd, wr, stall, yield]:

    wait    scoreboards (0..5) the instruction waits on before it issues   (slot req_bit_set, bits 121:116)
    rd      scoreboard set when its sources have been read, or None       (src_rel_sb, bits 115:113; 7 = none)
    wr      scoreboard set when its result has been written, or None      (dst_wr_sb, bits 112:110)
    stall   cycles before the next instruction issues, 0..15              (usched_info, bits 108:105)
    yield_  the warp scheduler may switch to another warp after this instruction  (bit 109 clear)

    str(ctrl) prints them in that order, e.g. "[{1}, -, 0, 4, Y]".  The operand reuse flags (bits
    124:122) are not part of this view; they stay in the reuse_src_* slots.
    """
    wait: frozenset
    rd: int | None
    wr: int | None
    stall: int
    yield_: bool

    def __str__(self):
        wait = ",".join(str(i) for i in sorted(self.wait))
        rd, wr = ("-" if v is None else str(v) for v in (self.rd, self.wr))
        return f"[{{{wait}}}, {rd}, {wr}, {self.stall}, {'Y' if self.yield_ else 'N'}]"

    def slots(self):
        """The slot values that encode this control word."""
        if (not isinstance(self.wait, (set, frozenset)) or
                any(type(v) is not int or not 0 <= v < 6 for v in self.wait) or
                any(v is not None and (type(v) is not int or not 0 <= v < 6) for v in (self.rd, self.wr)) or
                type(self.stall) is not int or not 0 <= self.stall < 16 or type(self.yield_) is not bool):
            raise EncodeError(f"invalid control word: {self!r}")
        return {"req_bit_set": sum(1 << i for i in self.wait), "usched_info": self.stall | (0 if self.yield_ else 16),
                "src_rel_sb": _SB_NONE if self.rd is None else self.rd, "dst_wr_sb": _SB_NONE if self.wr is None else self.wr}


@dataclass(frozen=True, slots=True, eq=False)
class Instr:
    """Immutable slots with a cached, validated encoding. Raw construction validates on encode()."""
    cls: InstrClass
    slots: MappingProxyType
    _word: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "slots", MappingProxyType(dict(self.slots)))

    @property
    def ctrl(self):
        s = self.slots
        u = s.get("usched_info", 0)
        sb = lambda v: None if v == _SB_NONE else v
        return Ctrl(frozenset(i for i in range(6) if s.get("req_bit_set", 0) >> i & 1),
                    sb(s.get("src_rel_sb", _SB_NONE)), sb(s.get("dst_wr_sb", _SB_NONE)), u & 15, not u & 16)

    def with_ctrl(self, ctrl):
        """Copy with this control word; unsupported field requests raise EncodeError."""
        return self.replace(**self.cls._control_slots(ctrl))

    mnemonic = property(lambda self: self.cls.mnemonic)
    branch = property(lambda self: self.cls.branch,
                      doc='Control transfer: "branch", "call", "return", "exit", or None.')

    @property
    def guard(self):
        """(predicate register, negated) of the leading @Pg guard, or None."""
        name = self.cls.guard_slot
        return None if name is None else (self._semantic_slot(name), bool(self.slots.get(name + "@not", 0)))

    @property
    def instruction_type(self):
        """The declared instruction type used to interpret scoreboard behavior."""
        value = self.cls.instruction_type
        if not isinstance(value, str) or not value.startswith(("INST_TYPE_COUPLED", "INST_TYPE_DECOUPLED")):
            raise SemanticError(f"{self.cls.name}: INSTRUCTION_TYPE {value!r} interpretation is unavailable")
        return value

    @property
    def conditional(self):
        """Whether the instruction may not take effect: a guard other than PT, a predicate operand other
        than true PT / UPT, or a lane selector (BRA.DIV UR4, BRA.U.NOT_TID0)."""
        g = self.guard
        if g is not None and (g[0] != PT or g[1]):
            return True
        return (any(self._semantic_slot(name) != PT or self.slots.get(name + "@not", 0)
                    for name in self.cls.predicates)
                or "URb" in self.slots or bool(self.slots.get("nottid0", 0)))

    def _semantic_slot(self, name):
        if name not in self.slots:
            raise SemanticError(f"{self.cls.name}: slot {name} is unavailable")
        return self.slots[name]

    def _effects_supported(self):
        if self.cls.indexed_register is not None:
            raise SemanticError(f"{self.cls.name}: indexed RF operand {self.cls.indexed_register} effects are unavailable")
        if self.mnemonic in ("MOV", "MOV32I") and "PixMaskU04" in self.cls.slots \
                and self._semantic_slot("PixMaskU04") != 15:
            raise SemanticError(f"{self.cls.name}: masked PixMaskU04 effects are unavailable")

    def _register(self, name, file, size):
        """Resolve a scalar/width-predicate operand; zero registers need no width interpretation."""
        value, isa = self._semantic_slot(name), self.cls.isa
        if self.cls.slots[name]["type"] in ("PR", "UPRONLY"):
            return Reg(file, 0, 7)
        if value == isa.zero[file]:
            return None
        bits = (size if type(size) is int else self.cls.predicate(size, self.slots)) if file in ("R", "UR") else 32
        if type(bits) is not int or bits < 0:
            raise SemanticError(f"{self.cls.name}: {name} width {size or 'predicate'} is unavailable")
        count = (max(bits, 1) + 31) // 32
        limit = isa.max_ureg if file == "UR" else isa.zero[file]
        if type(value) is not int or value < 0 or value + count > limit:
            raise SemanticError(f"{self.cls.name}: {name} span exceeds the {file} register file")
        return Reg(file, value, count)

    def _regs(self, written, *, definite=False, file=None):
        """Possible effects; definite selects complete destinations assuming execution. Unknown facts raise SemanticError."""
        self._effects_supported()
        out = []
        for name, regfile, w, size, whole in self.cls.operands:
            partial = (whole and w) or name == "Pnz"
            if (file is not None and regfile != file) or (w != written and not (partial and not written)):
                continue
            if definite and partial:
                continue
            reg = self._register(name, regfile, size)
            if reg is not None:
                out.append(reg)
        # PR_PRED's normalized scheduling condition identifies these implicit whole-file reads.
        if not written and file in (None, "P") and self.mnemonic in ("CSMTEST", "VOTE_VTG") \
                and self._semantic_slot("vtgmode") in (2, 3):
            out.append(Reg("P", 0, 7))
        return out

    reads = property(lambda self: self._regs(False), doc="Possible reads, including guards and preservation reads; zero registers omitted.")
    writes = property(lambda self: self._regs(True), doc="Possible writes, including masked destinations; not liveness kills.")
    definite_writes = property(lambda self: self._regs(True, definite=True),
                              doc="Complete destinations IF executed; excludes masked whole-file writes and Pnz. Apply the guard policy separately.")

    @property
    def target_kind(self):
        """'none', 'direct', or 'indirect'; unavailable declared target interpretation raises SemanticError."""
        if self.cls.target_error:
            raise SemanticError(f"{self.cls.name}: {self.cls.target_error}")
        if self.cls.target_kind != "none":
            self._semantic_slot(self.cls.target_slot)
        return self.cls.target_kind

    def target(self, pc):
        """Direct branch/call/reconvergence byte address; relative forms are pc + 16 + imm.
        Returns None for indirect or absent targets."""
        slot = self.cls.target_slot
        if self.target_kind != "direct":
            return None
        return self.slots[slot] + (pc + 16 if self.cls.target_relative else 0)

    def with_target(self, *, pc, target):
        """A validated copy naming `target` at its new `pc`, both nonnegative 16-byte code addresses.
        Register/constant indirect targets are not supported by this operation."""
        if any(type(value) is not int or value < 0 or value % 16 for value in (pc, target)):
            raise EncodeError(f"{self.cls.name}: pc and target must be nonnegative 16-byte aligned integers")
        try:
            direct = self.target_kind == "direct"
        except SemanticError as error:
            raise EncodeError(str(error)) from error
        if not direct:
            raise EncodeError(f"{self.cls.name}: no supported direct code target")
        value = target - (pc + 16) if self.cls.target_relative else target
        return self.replace(**{self.cls.target_slot: value})

    @property
    def modifiers(self):
        """[(slot, printed name)] of the class's modifier slots, in FORMAT order."""
        isa = self.cls.isa
        return [(t["slot"], isa.enum_name(t["type"], self.slots.get(t["slot"])))
                for t in self.cls.rec["format"] if isinstance(t, dict) and t.get("mod")]

    def encode(self):
        if self._word is None:
            object.__setattr__(self, "_word", self.cls.encode(self.slots))
        return self._word

    def to_bytes(self):
        return self.encode().to_bytes(16, "little")

    def replace(self, **slots):
        """A complete validated copy with the requested slot changes; the original is unchanged."""
        return self.cls._make({**self.slots, **slots})

    def __eq__(self, other):
        return isinstance(other, Instr) and self.cls is other.cls and self.slots == other.slots

    def __hash__(self):
        return hash((self.cls.name, frozenset(self.slots.items())))

    def __repr__(self):
        return f"<Instr {self.mnemonic} {self.cls.name} {self.ctrl}>"


class Unknown:
    """A word that no class accepts; it encodes back to itself so code passes through."""
    __slots__ = ("word",)
    cls = mnemonic = None

    def __init__(self, word):
        self.word = _word(word)

    def encode(self):
        return self.word

    def to_bytes(self):
        return self.word.to_bytes(16, "little")

    def __repr__(self):
        return f"<Unknown 0x{self.word:032x}>"
