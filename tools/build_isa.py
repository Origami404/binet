#!/usr/bin/env python3
"""Build a SASS instruction-set database from a captured nvdisasm description."""
import argparse
import functools
import gzip
import hashlib
import itertools
import json
import math
import operator
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
DB = TOOLS.parent / "binet" / "db"
SECTIONS = ("PARAMETERS", "CONSTANTS", "STRING_MAP", "REGISTERS", "TABLES",
            "OPERATION PROPERTIES", "OPERATION PREDICATES", "FUNIT uC", "NOP_ENCODING")
CLASS_RE = re.compile(r'(ALTERNATE )?CLASS "([^"]+)"')
SUBSECTION_RE = re.compile(r"(FORMAT|CONDITIONS|PROPERTIES|PREDICATES|OPCODES|ENCODING)(?: |$)")
INT_RE = re.compile(r"-?(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|\d+)U?")
RANGE_ITEM = re.compile(r'"?([^"(]+)\((\d+)\.\.(\d+)\)"?(?:\s*=\s*\((-?\d+)\.\.(-?\d+)\))?')
VALUE_ITEM = re.compile(r'"?([^"=*]+?)"?\*?\s*(?:\*?=\s*(\S+))?')
ENUM_REF = re.compile(r'`?([A-Za-z_][\w.]*)@"?([^"]+?)"?')
RELOC_TOKEN = re.compile(r'"([^"]*)"|([{},;])|([^\s{},;"]+)')
FORMAT_TOKEN = re.compile(r"""\s+
  | '(?P<lit>[^']*)'
  | \[(?P<pre>!|-|\|\||~)\]
  | (?P<mod>/)?(?P<type>[A-Za-z_][\w.]*)(?:\((?P<arg>[^()]*)\))?(?P<star>\*)?(?P<at>@)?:(?P<name>[A-Za-z_][\w.]*)
  | (?P<word>[A-Za-z_]\w*|\$\(|\)\$|[{}\[\]+@*])""", re.X)
EXPR_TOKEN = re.compile(r"""\s+
  | (?P<enum>`[A-Za-z_][\w.]*@(?:"[^"]*"|[\w.]+))
  | (?P<num>0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)
  | (?P<name>[A-Za-z_][\w.]*)
  | (?P<op>==|!=|<=|>=|<<|>>|&&|\|\||[-+*/%&|^!~?:<>()])""", re.X)
# C operator precedence (higher binds tighter); ?: is handled below all of these.
BINARY = {"||": 2, "&&": 3, "|": 4, "^": 5, "&": 6, "==": 7, "!=": 7, "<": 8, ">": 8, "<=": 8, ">=": 8,
          "<<": 9, ">>": 9, "+": 10, "-": 10, "*": 11, "/": 11, "%": 11}
OPS = {"*": operator.mul, "/": operator.floordiv, "%": operator.mod, "+": operator.add, "-": operator.sub,
       "<<": operator.lshift, ">>": operator.rshift, "&": operator.and_, "^": operator.xor, "|": operator.or_,
       "<": lambda a, b: int(a < b), ">": lambda a, b: int(a > b), "<=": lambda a, b: int(a <= b),
       ">=": lambda a, b: int(a >= b), "==": lambda a, b: int(a == b), "!=": lambda a, b: int(a != b),
       "&&": lambda a, b: int(bool(a) and bool(b)), "||": lambda a, b: int(bool(a) or bool(b))}
UNARY = {"!": lambda a: int(not a), "~": operator.invert, "-": operator.neg}
MAX_COMBINATIONS = 1 << 16  # slot-value combinations a predicate may be tabulated over
REG_FILES = {"Register": "R", "NonZeroRegister": "R", "ZeroRegister": "R", "UniformRegister": "UR",
             "NonZeroUniformRegister": "UR", "ZeroUniformRegister": "UR", "Predicate": "P", "Predicate_vimnmx": "P",
             "UniformPredicate": "UP", "PR": "P", "UPRONLY": "UP"}
IMMEDIATES = {"UImm", "SImm", "RSImm", "BITSET"}

errors = []
context = [""]  # what is being parsed, for error messages
compact = functools.partial(json.dumps, separators=(",", ":"))


def error(msg):
    errors.append(f"{context[0]}: {msg}" if context[0] else msg)


def parse_int(text):
    """Integer literal of the dump (decimal, 0x.., 0b.. with '_', U suffix), else None."""
    text = text.strip()
    if not INT_RE.fullmatch(text):
        return None
    digits = text.rstrip("U").replace("_", "").lstrip("-")
    base = {"0x": 16, "0X": 16, "0b": 2, "0B": 2}.get(digits[:2])
    value = int(digits[2:], base) if base else int(digits)
    return -value if text.startswith("-") else value


def scalar(text):
    value = parse_int(text)
    return text.strip() if value is None else value


def split_top(text, sep):
    """Split on `sep` outside quotes and parentheses; drop empty parts."""
    parts, cur, depth, quote = [], [], 0, None
    for ch in text:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------
# Header sections
# --------------------------------------------------------------------------
def parse_header(text):
    header = {"ARCHITECTURE": re.match(r'ARCHITECTURE "([^"]*)"', text).group(1)}
    for key, value in re.findall(r'^[ \t]+([A-Z_]+)[ \t]*=[ \t]*"(.*)";[ \t]*$', text, re.M):
        header[key] = value
    for key, value in re.findall(r'^[ \t]+([A-Z_]+) ([^;={}"\n]+);[ \t]*$', text, re.M):
        header[key] = scalar(value)
    m = re.search(r"\bOPTIONS\s+([^;]*);", text)
    header["OPTIONS"] = split_top(m.group(1), ",") if m else []
    header["CONDITION_TYPES"] = dict(re.findall(r"^[ \t]+(\w+)[ \t]*:[ \t]*(ERROR|WARNING|INFO)[ \t]*$",
                                                text, re.M))
    m = re.search(r"\bRELOCATORS\b(.*)", text, re.S)
    header["RELOCATORS"] = parse_relocators(m.group(1)) if m else []
    return header


def parse_relocators(text):
    """RELOCATORS block -> one nested list (strings, ints, bools) per entry."""
    entries, stack = [], []
    for m in RELOC_TOKEN.finditer(text):
        string, punct, atom = m.groups()
        if punct == ";" and not stack:
            break
        if punct == "{":
            stack.append([])
        elif punct == "}":
            done = stack.pop()
            (stack[-1] if stack else entries).append(done)
        elif stack and string is not None:
            stack[-1].append(string)
        elif stack and atom:
            stack[-1].append({"True": True, "False": False}.get(atom, scalar(atom)))
    return entries


def parse_assignments(lines):
    matches = (re.match(r"\s*(\w+)\s*=\s*([^;\s]+)", line) for line in lines)
    return {m.group(1): scalar(m.group(2)) for m in matches if m}


def parse_string_map(lines):
    matches = (re.match(r"\s*(\S+)\s*->\s*(\S+)", line) for line in lines)
    return dict(m.groups() for m in matches if m)


def parse_enums(lines):
    """REGISTERS statements -> {type: [[name, value], ...]}."""
    enums, unions = {}, {}
    for stmt in split_top("\n".join(lines), ";"):
        name, _, body = " ".join(stmt.split()).partition(" ")
        context[0] = f"enum {name}"
        if body.startswith("="):  # Integer = Integer8 + Integer16 + ...
            enums[name], unions[name] = None, [part.strip() for part in body[1:].split("+")]
            continue
        members, prev = [], -1
        for item in split_top(body, ","):
            m = RANGE_ITEM.fullmatch(item)
            if m:  # NAME(a..b)=(c..d); a bare NAME(a..b) takes the values a..b
                base, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
                if m.group(4) is None and lo != prev + 1:
                    error(f"range {item!r} does not continue the numbering")
                first = lo if m.group(4) is None else int(m.group(4))
                members += [[f"{base}{i}", first + i - lo] for i in range(lo, hi + 1)]
                prev = members[-1][1]
                continue
            m = VALUE_ITEM.fullmatch(item)  # NAME, "NAME", NAME=v; `*=` and a trailing `*` (REQ req*) are dropped
            value = (parse_int(m.group(2)) if m.group(2) else prev + 1) if m else None
            if value is None:
                error(f"cannot parse item {item!r}")
                continue
            members.append([m.group(1), value])
            prev = value
        enums[name] = members

    def resolve(name, seen=()):
        if name not in enums or name in seen:
            error(f"union refers to unknown or cyclic enum {name!r}")
            return []
        if enums[name] is None:
            enums[name] = [pair for part in unions[name] for pair in resolve(part, seen + (name,))]
        return enums[name]

    for name in unions:
        context[0] = f"enum {name}"
        resolve(name)
    return enums


def enum_resolver(enums):
    """-> enum_value(type, name): the exact name, else case-insensitively (header OPTIONS
    CASE_INSENSITIVE_SYNTAX: tables write DC@noDC for "nodc")."""
    exact = {name: dict(pairs) for name, pairs in enums.items()}
    folded = {name: {k.upper(): v for k, v in pairs} for name, pairs in enums.items()}
    return lambda etype, name: exact.get(etype, {}).get(name, folded.get(etype, {}).get(name.upper()))


def parse_tables(lines, enum_value):
    """TABLES -> {name: [[[inputs...], output], ...]} with cells resolved."""

    def cell(text):
        if text == "-":
            return None
        if text.startswith("'"):
            return text.strip("'")
        value = parse_int(text)
        if value is None:
            m = ENUM_REF.fullmatch(text)
            value = enum_value(m.group(1), m.group(2)) if m else None
            if value is None:
                error(f"cannot resolve cell {text!r}")
        return value

    tables, rows = {}, None
    for line in lines:
        text = line.strip()
        if "->" in text:
            inputs, output = text.split("->", 1)
            rows.append([[cell(c) for c in inputs.split()], cell(output.strip())])
        elif re.fullmatch(r"[A-Za-z_]\w*", text):
            context[0] = f"table {text}"
            rows = tables[text] = []
    for name, rows in tables.items():
        if len({len(inputs) for inputs, _ in rows}) > 1:
            context[0] = f"table {name}"
            error("rows have different input counts")
    return tables


# --------------------------------------------------------------------------
# CLASS blocks
# --------------------------------------------------------------------------
def split_classes(lines):
    """-> [(name, is_alternate, {subsection: text})] in file order."""
    classes, current = [], None
    for line in lines:
        m = CLASS_RE.match(line)
        if m:
            classes.append((m.group(2), bool(m.group(1)), {}))
            current = None
            continue
        m = SUBSECTION_RE.match(line)
        if m:
            current = m.group(1)
            classes[-1][2][current] = line[len(current):]
        elif current:
            classes[-1][2][current] += "\n" + line
    return classes


def format_slot(m, pre):
    slot = {"slot": m.group("name"), "type": m.group("type")}
    arg = m.group("arg")
    if arg is not None:
        parts = split_top(arg, "/")
        if len(parts) > 1 and parts[-1] == "PRINT":
            slot["print"] = True  # print the operand even when it has its default value
            arg = "/".join(parts[:-1])
        sized = re.fullmatch(r"(\d+)(?:/(-?\w+)(\*)?)?", arg)  # UImm(5/0*), BITSET(6/0x0000), SImm(17)
        if sized:
            slot["width"] = int(sized.group(1))
            if sized.group(2) is not None:
                slot["default"] = scalar(sized.group(2))
            if sized.group(3):
                slot["default_star"] = True
        else:  # Predicate(PT), /FTZ("noftz"): an enum name
            slot["default"] = arg.strip('"')
    for flag in ("mod", "star", "at"):
        if m.group(flag):
            slot[flag] = True
    if pre:
        slot["pre"] = pre
    return slot


def parse_format(text):
    """FORMAT -> (flat token list, raw FORMAT_ALIAS statement or None)."""
    template, _, alias = text.partition(";")
    tokens, pre, pos = [], [], 0
    while pos < len(template):
        m = FORMAT_TOKEN.match(template, pos)
        if m is None:
            error(f"cannot tokenize FORMAT at {template[pos:pos + 40]!r}")
            break
        pos = m.end()
        if m.group("pre"):
            pre.append(m.group("pre"))
        elif m.group("name"):
            tokens.append(format_slot(m, pre))
            pre = []
        elif m.group("lit") is not None or m.group("word"):
            tokens += [f"[{p}]" for p in pre] + [m.group("lit") if m.group("lit") is not None else m.group("word")]
            pre = []
    alias = " ".join(alias.split()).rstrip(";").strip()
    return tokens, alias.removeprefix("FORMAT_ALIAS").strip() or None


def parse_conditions(text):
    """-> [[error, predicate, message]]; each message sits on its own line."""
    out, name, predicate = [], None, []
    for line in text.split("\n"):
        line = line.strip()
        if re.fullmatch(r"[A-Z][A-Z0-9_]*_(?:ERROR|WARNING|INFO)", line):
            if name:
                error(f"condition {name} has no message")
            name, predicate = line, []
        elif line.startswith('"') and name:
            out.append([name, " ".join(predicate).rstrip(":").strip(), line.strip('"')])
            name = None
        elif line and name:
            predicate.append(line)
    if name:
        error(f"condition {name} has no message")
    return out


def compile_operand(tokens, slots, defaults, enums):
    mods, clean = [], []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "{":
            if i + 2 >= len(tokens) or not isinstance(tokens[i + 1], dict) or not tokens[i + 1].get("mod") or tokens[i + 2] != "}":
                raise ValueError("unsupported operand annotation")
            mods.append(tokens[i + 1]["slot"])
            i += 3
        elif isinstance(token, dict) and token.get("mod"):
            mods.append(token["slot"])
            i += 1
        else:
            clean.append(token)
            i += 1
    operand = {"mods": mods, "optional": False}
    if len(clean) == 1 and isinstance(clean[0], dict):
        name = clean[0]["slot"]
        spec = slots[name]
        if "file" not in spec and spec["type"] not in IMMEDIATES and spec["type"] not in enums:
            raise ValueError(f"unsupported operand type {spec['type']}")
        return dict(operand, kind="scalar", slot=name, optional=name in defaults)
    if clean and isinstance(clean[0], dict) and clean[0]["type"] in ("C", "CX"):
        if len(clean) < 7 or clean[1] != "[" or not isinstance(clean[2], dict) or clean[3] != "]":
            raise ValueError("unsupported constant address")
        operand.update(kind="const", bank=clean[2]["slot"])
        clean = clean[4:]
        if clean and clean[0] == "*":
            clean = clean[1:]
    else:
        operand["kind"] = "mem"
    if not clean or clean[0] != "[" or clean[-1] != "]":
        raise ValueError("unsupported compound operand")
    terms = clean[1:-1]
    if not terms or len(terms) % 2 == 0 or any(not isinstance(t, dict) if i % 2 == 0 else t != "+" for i, t in enumerate(terms)):
        raise ValueError("unsupported address expression")
    regs = [s["slot"] for s in terms[::2] if slots[s["slot"]].get("file") in ("R", "UR")]
    offsets = [s["slot"] for s in terms[::2] if s["type"] in ("UImm", "SImm")]
    if len(offsets) > 1 or len(regs) > 2 or len(regs) + len(offsets) != len(terms[::2]):
        raise ValueError("unsupported address terms")
    if len(regs) == 2 and [slots[name]["file"] for name in regs] != ["R", "UR"]:
        raise ValueError("unsupported address register combination")
    return dict(operand, regs=regs, offset=offsets[0] if offsets else None)


def compile_constructor(fmt, slots, defaults, enums):
    if len(fmt) < 4 or fmt[:2] != ["PREDICATE", "@"] or not isinstance(fmt[2], dict) or fmt[3] != "Opcode":
        raise ValueError("unsupported guard/opcode format")
    end = fmt.index("$(") if "$(" in fmt else len(fmt)
    controls = {"req", "req_bit_set", "rd", "src_rel_sb", "wr", "wr_early", "dst_wr_sb", "usched_info", "batch_t", "pm_pred"}
    if any(t["slot"] not in controls if isinstance(t, dict) else t not in ("$(", "{", "&", "=", "}", ")$", "?") for t in fmt[end:]):
        raise ValueError("unsupported control annotation")
    tokens, mods = fmt[4:end], []
    while tokens and isinstance(tokens[0], dict) and tokens[0].get("mod"):
        mods.append(tokens[0]["slot"])
        tokens = tokens[1:]
    groups, group, stack = [], [], []
    for token in tokens:
        if token in ("[", "{"):
            stack.append(token)
        elif token in ("]", "}"):
            if not stack or stack.pop() != ("[" if token == "]" else "{"):
                raise ValueError("unbalanced operand format")
        if token == "," and not stack:
            groups.append(group)
            group = []
        else:
            group.append(token)
    if stack:
        raise ValueError("unbalanced operand format")
    if group:
        groups.append(group)
    return {"mods": mods, "operands": [compile_operand(g, slots, defaults, enums) for g in groups]}


def compile_class(cls, enums, enum_value):
    """Compile FORMAT into slot metadata and typed construction rules."""
    fmt = cls["format"]
    slots = {t["slot"]: dict(t) for t in fmt if isinstance(t, dict)}
    defaults = {}
    for name, spec in slots.items():
        kind = spec["type"]
        if kind in REG_FILES:
            spec["file"] = REG_FILES[kind]
        if kind in IMMEDIATES:
            spec["signed"] = kind in ("SImm", "RSImm")
        values = {v for _, v in enums.get(kind, ())}
        value = spec.get("default", next(iter(values)) if len(values) == 1 else None)
        if value is not None:
            if not isinstance(value, int):
                value = enum_value(kind, value)
            if value is None:
                error(f"slot {name}: unknown default {spec['default']!r}")
            else:
                defaults[name] = value
    guard = next((fmt[i + 1]["slot"] for i, t in enumerate(fmt[:-1]) if t == "@" and isinstance(fmt[i + 1], dict)), None)
    initial = {}
    for field in cls["encoding"]:
        value = field["value"]
        for arg in ([value] if "slot" in value else value.get("args", ())):
            if "slot" in arg and "attr" in arg:
                initial[arg["slot"] + "@" + arg["attr"]] = 0
    initial.update(defaults)
    if guard is not None:
        initial.setdefault(guard, 7)  # PT / UPT for a newly constructed instruction.
    try:
        constructor = compile_constructor(fmt, slots, defaults, enums)
    except ValueError as exc:
        constructor = {"unsupported": str(exc)}
    return dict(cls, slots=slots, defaults=defaults, initial=initial, guard=guard, constructor=constructor)


def parse_kv(text):
    pairs = (stmt.partition("=") for stmt in split_top(text, ";"))
    return {key.strip(): scalar(value) for key, eq, value in pairs if eq}


def compile_expr(text, enum_value):
    """C-like expression of the dump -> (function of {slot: value}, sorted slot names)."""
    tokens, pos = [], 0
    while pos < len(text):
        m = EXPR_TOKEN.match(text, pos)
        if m is None:
            raise ValueError(f"cannot tokenize {text[pos:pos + 30]!r}")
        pos = m.end()
        if m.lastgroup:
            tokens.append((m.lastgroup, m.group(m.lastgroup)))
    tokens.append(("end", ""))
    slots, at = set(), [0]

    def take(expected=None):
        kind, tok = tokens[at[0]]
        if expected is not None and tok != expected:
            raise ValueError(f"expected {expected!r}, found {tok!r}")
        at[0] += 1
        return kind, tok

    def unary():
        kind, tok = take()
        if tok == "(":
            inner = ternary()
            take(")")
            return inner
        if kind == "op" and tok in UNARY:
            operand, fn = unary(), UNARY[tok]
            return lambda env: fn(operand(env))
        if kind == "num":
            value = parse_int(tok)
            return lambda env: value
        if kind == "enum":
            etype, _, name = tok[1:].partition("@")
            value = enum_value(etype, name.strip('"'))
            if value is None:
                raise ValueError(f"unknown enum literal {tok}")
            return lambda env: value
        if kind == "name":
            slots.add(tok)
            return lambda env: env[tok]
        raise ValueError(f"unexpected {tok!r}")

    def binary(min_prec):
        left = unary()
        while True:
            kind, op = tokens[at[0]]
            prec = BINARY.get(op) if kind == "op" else None
            if prec is None or prec < min_prec:
                return left
            take()
            right, fn, lhs = binary(prec + 1), OPS[op], left
            left = lambda env, l=lhs, r=right, f=fn: f(l(env), r(env))

    def ternary():
        cond = binary(0)
        if tokens[at[0]][1] != "?":
            return cond
        take("?")
        yes = ternary()
        take(":")
        no = ternary()
        return lambda env: yes(env) if cond(env) else no(env)

    fn = ternary()
    take("")
    return fn, sorted(slots)


def tabulate(value, slots, enums, enum_value):
    """Predicate value -> int, a bare "$CONSTANT", or {"by": [slots], "map": {"v1,v2": int}}
    holding the expression's value for every combination of values its slots can take."""
    if not isinstance(value, str) or re.fullmatch(r"\$\w+", value):
        return value
    fn, names = compile_expr(value, enum_value)
    domains = []
    for name in names:
        slot = slots.get(name)
        if slot is None:
            raise ValueError(f"{name!r} is not a FORMAT slot")
        if slot["type"] in enums:  # enum-typed modifier: its distinct encoded values
            domains.append(sorted({v for _, v in enums[slot["type"]]}))
        elif "width" in slot:  # small immediate such as TEX's 4-bit wmsk
            domains.append(range(1 << slot["width"]))
        else:
            raise ValueError(f"slot {name!r} ({slot['type']}) has no finite set of values")
    if not all(domains) or math.prod(map(len, domains)) > MAX_COMBINATIONS:
        raise ValueError(f"cannot enumerate the values of {names}")
    table = {",".join(map(str, combo)): fn(dict(zip(names, combo))) for combo in itertools.product(*domains)}
    results = set(table.values())
    return results.pop() if len(results) == 1 else {"by": names, "map": table}


def parse_bits(token):
    """BITS_<width>_<hi>_<lo>[_<hi>_<lo>...]_<name> -> (name, [[hi, lo], ...])."""
    parts = token.split("_")[1:]
    width, bits, i = int(parts[0]), [], 1
    while sum(hi - lo + 1 for hi, lo in bits) < width and i + 1 < len(parts) \
            and parts[i].isdigit() and parts[i + 1].isdigit():
        bits.append([int(parts[i]), int(parts[i + 1])])
        i += 2
    name = "_".join(parts[i:])
    if sum(hi - lo + 1 for hi, lo in bits) != width:
        error(f"field {name}: width {width} does not match bits {bits}")
    return name, bits


def parse_value(text, opcode, tables):
    """ENCODING right-hand side -> value object (see the module docstring)."""
    text = text.strip()
    star = text.startswith("*")
    body = text[1:].strip() if star else text
    call = re.fullmatch(r"([A-Za-z_]\w*)\((.*)\)", body, re.S)
    slot = re.fullmatch(r"([A-Za-z_][\w.]*)(?:@(\w+))?((?:\s+(?:SCALE|MULTIPLY)\s+\d+)*)", body)
    convert = re.fullmatch(r"([A-Za-z_][\w.]*)\s+convertFloatType\((.*)\)", body, re.S)
    if parse_int(body) is not None:
        value = {"const": parse_int(body)}
    elif body == "Opcode":
        value = {"opcode": opcode}
    elif call:
        args = [parse_value(arg, opcode, tables) for arg in split_top(call.group(2), ",")]
        value = {"table" if call.group(1) in tables else "call": call.group(1), "args": args}
    elif slot:
        value = {"slot": slot.group(1)}
        if slot.group(2):
            value["attr"] = slot.group(2)
        for op, n in re.findall(r"(SCALE|MULTIPLY)\s+(\d+)", slot.group(3)):
            value[op.lower()] = int(n)
    elif convert:
        value = {"slot": convert.group(1), "convert": " ".join(convert.group(2).split())}
    else:
        value = {"raw": text}
    if star:
        value["star"] = True
    return value


def parse_encoding(text, opcode, tables):
    """ENCODING -> ([{field, bits, value}], raw REMAP template or None)."""
    fields, remap = [], None
    for stmt in split_top(text, ";"):
        if stmt.startswith("!"):  # name of the unused-bits mask
            continue
        if stmt.startswith("REMAP"):
            remap = stmt[len("REMAP"):].strip().strip('"')
            continue
        targets, eq, rhs = stmt.partition("=")
        targets = [t.strip() for t in targets.split(",")]
        if not eq or not all(t.startswith("BITS_") for t in targets):
            error(f"cannot parse ENCODING statement {stmt!r}")
            continue
        value = parse_value(rhs, opcode, tables)
        for i, target in enumerate(targets):
            name, bits = parse_bits(target)
            fields.append({"field": name, "bits": bits, "value": dict(value, out=i) if len(targets) > 1 else value})
    return fields, remap


def fixed_bits(fields):
    """[mask, value] (hex, hi64_lo64) of the bits set by constant fields."""
    mask = value = 0
    for f in fields:
        v = f["value"]
        constant = v.get("const", v.get("opcode"))
        if constant is None or "out" in v:
            continue
        rest = sum(hi - lo + 1 for hi, lo in f["bits"])
        for hi, lo in f["bits"]:
            width = hi - lo + 1
            rest -= width
            mask |= ((1 << width) - 1) << lo
            value |= ((constant >> rest) & ((1 << width) - 1)) << lo
    return [f"0x{x >> 64:016x}_{x & (1 << 64) - 1:016x}" for x in (mask, value)]


def value_slots(value):
    if "slot" in value:
        yield value["slot"]
    for arg in value.get("args", ()):
        yield from value_slots(arg)


def check_class(cls, tables):
    slots = {t["slot"] for t in cls["format"] if isinstance(t, dict)}
    owner = {}
    for f in cls["encoding"]:
        for hi, lo in f["bits"]:
            if not 0 <= lo <= hi <= 127:
                error(f"field {f['field']}: bit range {hi}:{lo} outside the 128-bit word")
            clash = next((b for b in range(lo, hi + 1) if b in owner), None)
            if clash is not None:
                error(f"bit {clash} is set by both {owner[clash]} and {f['field']}")
            owner.update(dict.fromkeys(range(lo, hi + 1), f["field"]))
        missing = sorted(set(value_slots(f["value"])) - slots)
        if missing:
            error(f"field {f['field']} uses slots missing from FORMAT: {missing}")
        table = tables.get(f["value"].get("table"))
        if table and len(f["value"]["args"]) != len(table[0][0]):
            error(f"field {f['field']}: {f['value']['table']} takes {len(table[0][0])} inputs")
    if sum("opcode" in f["value"] for f in cls["encoding"]) != 1:
        error("needs exactly one opcode field")


def parse_class(name, alternate, sections, tables, enums, enum_value):
    pairs = re.findall(r"([\w.]+)\s*=\s*0b([01]+)\s*;", sections.get("OPCODES", ""))
    names, values = [n for n, _ in pairs], {int(b, 2) for _, b in pairs}
    if len(values) != 1:
        error(f"OPCODES {names} must share exactly one value")
    opcode = min(values, default=None)
    fmt, alias = parse_format(sections.get("FORMAT", ""))
    encoding, remap = parse_encoding(sections.get("ENCODING", ""), opcode, tables)
    slots = {t["slot"]: t for t in fmt if isinstance(t, dict)}
    predicates = {}
    for key, value in parse_kv(sections.get("PREDICATES", "")).items():
        try:
            predicates[key] = tabulate(value, slots, enums, enum_value)
        except (ValueError, ZeroDivisionError) as exc:
            error(f"predicate {key}: {exc}")
    cls = {
        "name": name, "alternate": alternate,
        "mnemonic": next((n for n in names if not n.endswith("_pipe")), names[0] if names else None),
        "opcode": opcode, "opcode_names": names, "fixed": fixed_bits(encoding),
        "format": fmt, "encoding": encoding,
        "conditions": parse_conditions(sections.get("CONDITIONS", "")),
        "properties": parse_kv(sections.get("PROPERTIES", "")),
        "predicates": predicates,
    }
    if alias:
        cls["format_alias"] = alias
    if remap:
        cls["remap"] = remap
    check_class(cls, tables)
    return compile_class(cls, enums, enum_value)


# --------------------------------------------------------------------------
def pinned_sha256(arch):
    """sha256 that intercept_nvdisasm.py pins for this arch's instructions, if any."""
    sys.path.insert(0, str(TOOLS))
    try:
        from intercept_nvdisasm import EXPECTED
    except ImportError:
        return None
    return EXPECTED.get(arch.removeprefix("sm_"), (None,))[0]


def build(path):
    raw = path.read_bytes()
    lines = raw.decode("utf-8").replace("\r\n", "\n").split("\n")
    first_class = next(i for i, line in enumerate(lines) if CLASS_RE.match(line))
    heads = {line.rstrip(): i for i, line in enumerate(lines[:first_class]) if line.rstrip() in SECTIONS}
    bounds = sorted(heads.values()) + [first_class]

    def section(name):
        if name not in heads:
            error(f"missing section {name}")
            return []
        start = heads[name]
        return lines[start + 1:next(b for b in bounds if b > start)]

    header = parse_header("\n".join(lines[:bounds[0]]))
    enums = parse_enums(section("REGISTERS"))
    enum_value = enum_resolver(enums)
    tables = parse_tables(section("TABLES"), enum_value)
    classes = []
    for name, alternate, sections in split_classes(lines[first_class:]):
        context[0] = f"class {name}"
        classes.append(parse_class(name, alternate, sections, tables, enums, enum_value))
    context[0] = ""

    arch = path.name.removesuffix("_instructions.txt")
    sha256 = hashlib.sha256(raw).hexdigest()
    return {
        "schema_version": 2,
        "arch": arch,
        "source": {"file": path.name, "sha256": sha256, "elf_version": header.get("ELF_VERSION"),
                   "verified": sha256 == pinned_sha256(arch)},
        "header": header,
        "parameters": parse_assignments(section("PARAMETERS")),
        "constants": parse_assignments(section("CONSTANTS")),
        "string_map": parse_string_map(section("STRING_MAP")),
        "enums": enums,
        "tables": tables,
        "classes": classes,
    }


def write_json(db, path):
    """One line per top-level key, enum, table and class: compact yet diffable."""
    chunks = []
    for key, value in db.items():
        if key in ("enums", "tables"):
            body = "{\n" + ",\n".join(f"{json.dumps(k)}:{compact(v)}" for k, v in value.items()) + "\n}"
        elif key == "classes":
            body = "[\n" + ",\n".join(compact(c) for c in value) + "\n]"
        else:
            body = compact(value)
        chunks.append(f"{json.dumps(key)}:{body}")
    data = ("{\n" + ",\n".join(chunks) + "\n}\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".gz":
        with tmp.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as compressed:
                compressed.write(data)
    else:
        tmp.write_bytes(data)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=" ".join(__doc__.split("\n\n")[0].split()))
    ap.add_argument("instructions", type=Path, help="sm_XX_instructions.txt written by intercept_nvdisasm.py")
    ap.add_argument("--dir", type=Path, default=DB, help="output directory (default: binet/db)")
    args = ap.parse_args()

    db = build(args.instructions)
    for msg in errors[:40]:
        print(f"error: {msg}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} error(s); nothing written", file=sys.stderr)
        return 1
    if db["source"]["elf_version"] != 129 or not db["source"]["verified"]:
        print(f"warning: {args.instructions.name} is not a verified CUDA 12.9 capture", file=sys.stderr)
    args.dir.mkdir(parents=True, exist_ok=True)
    out = args.dir / f"{db['arch']}.isa.json.gz"
    write_json(db, out)
    constructors = sum("unsupported" not in c["constructor"] for c in db["classes"])
    print(f"{out}: {len(db['classes'])} classes, {constructors} constructor forms, {len(db['enums'])} enums, {len(db['tables'])} tables, "
          f"{out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
