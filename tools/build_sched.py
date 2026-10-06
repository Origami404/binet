#!/usr/bin/env python3
"""Build a scheduling database from a captured nvdisasm latency description."""
import argparse
import functools
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
DB = TOOLS.parent / "binet" / "db"
HEADERS = {"OPERATION SETS", "HARD RESOURCE", "RESOURCE", "CONNECTOR NAMES", "CONNECTOR NAME",
           "CONNECTOR CONDITIONS", "CONNECTOR SETS", "OPERATION PIPELINE RESOURCES"}
RANGE_RE = re.compile(r"\(\(\(\(\(MD_PRED\((\w+)\)\)>=\(1\)\)\?\(MD_PRED\(\1\)\):\(1\)\)-1\)>>5\)\+1")
CONDITION_RE = re.compile(r"\(\((.+)\)_OR_0\)")  # both matched with all whitespace removed
TERM_RE = re.compile(r"([\w.]+)(?:\[(\w+)\])?(?:`\{([^}]*)\})?")
TOKEN_RE = re.compile(r"\s+|(HARD\(\d+\)|[\w.]+(?:\[\w+\])?(?:`\{[^}]*\})?|[+:-])")
TABLE_RE = re.compile(r"TABLE_(\w+)\((\w+)\)\s*:\s*(.*?)\s*=\s*\{(.*)\}")

errors = []
compact = functools.partial(json.dumps, separators=(",", ":"))


def error(msg):
    errors.append(msg)


def statements(text):
    """-> [(section, statement)], statements ';'-terminated with whitespace collapsed."""
    out, section, buf = [], None, []
    for line in text.split("\n"):
        line = line.strip()
        if line in HEADERS and not buf:
            section = line
            continue
        if line:
            buf.append(line)
        while buf and ";" in buf[-1]:
            head, _, rest = " ".join(buf).partition(";")
            out.append((section, " ".join(head.split())))
            buf = [rest.strip()] if rest.strip() else []
    if buf:
        error(f"unterminated statement {' '.join(buf)[:60]!r}")
    return out


def value(text):
    """Cell / default value -> int, "ORDERED_ZERO", {"hard": n}, or None for '-'."""
    if text == "-":
        return None
    if text.isdigit():
        return int(text)
    m = re.fullmatch(r"HARD\((\d+)\)", text)
    if m:
        return {"hard": int(m.group(1))}
    if text != "ORDERED_ZERO":
        error(f"unknown value {text!r}")
    return text


def elements(text):
    """'R(0..254), RZ' -> ['R0', ..., 'R254', 'RZ']."""
    out = []
    for item in filter(None, (i.strip() for i in text.split(","))):
        m = re.fullmatch(r"(\w+?)\((\d+)\.\.(\d+)\)", item)
        out += [f"{m.group(1)}{i}" for i in range(int(m.group(2)), int(m.group(3)) + 1)] if m else [item]
    return out


def tokens(text):
    out, pos = [], 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if m is None:
            error(f"cannot tokenize {text[pos:pos + 40]!r}")
            break
        pos = m.end()
        if m.group(1):
            out.append(m.group(1))
    return out


class Sched:
    def __init__(self):
        self.resources, self.connectors, self.sets = {}, {}, {}
        self.ranges, self.conditions, self.groups = {}, {}, {}
        self.tables, self.capacity, self.uses = [], {}, []

    # -- definitions -------------------------------------------------------
    def add_set(self, stmt):
        name, _, expr = (s.strip() for s in stmt.partition("="))
        members, sign = set(), "+"
        for part in re.findall(r"\{[^}]*\}|[+-]|[\w.]+", expr):
            if part in "+-":
                sign = part
                continue
            if part.startswith("{"):
                names = {n.strip() for n in part[1:-1].split(",") if n.strip()}
            elif part in self.sets:
                names = set(self.sets[part])
            else:
                error(f"set {name}: unknown set {part!r}")
                continue
            members = members | names if sign == "+" else members - names
        if name in self.sets and set(self.sets[name]) != members:
            error(f"set {name} is defined twice with different members")
        self.sets[name] = sorted(members)

    def add_resource(self, stmt, hard):
        m = re.fullmatch(r"(\w+)\s*(?:\((\w+)\))?\s*(?:=\s*\{([^}]*)\})?\s*(.*)", stmt)
        defaults = dict(re.findall(r"DEFAULT_(\w+)=(\S+)", m.group(4)))
        self.resources[m.group(1)] = {
            "hard": hard, "unit": m.group(2), "elements": elements(m.group(3) or ""),
            "default": {kind: value(defaults[kind]) if kind in defaults else None for kind in ("ANTI", "OUTPUT")},
        }

    def add_connectors(self, stmt):
        names, _, resource = stmt.rpartition(":")
        resource = resource.strip()
        if resource not in self.resources:
            error(f"connectors {names.strip()!r}: unknown resource {resource!r}")
        for item in re.findall(r"\w+(?:\s*\{[^}]*\})?", names):
            m = re.fullmatch(r"(\w+)(?:\s*\{([^}]*)\})?", item)
            self.connectors[m.group(1)] = {"resource": resource}
            if m.group(2) is not None:  # PR_PRED { P(0..6) }: a fixed set of elements
                self.connectors[m.group(1)]["elements"] = elements(m.group(2))

    def add_condition(self, stmt):
        name, _, expr = (s.strip() for s in stmt.partition("="))
        expr = re.sub(r"\s+", "", expr)
        m = RANGE_RE.fullmatch(expr)
        if m:  # registers = ceil(max(size, 1) / 32): keep only the size predicate
            self.ranges[name] = m.group(1)
            return
        m = CONDITION_RE.fullmatch(expr)
        tests = [re.fullmatch(r"(\w+)(==|!=)(\d+)", t) for t in m.group(1).split("||")] if m else [None]
        if None in tests or len({(t.group(1), t.group(2)) for t in tests}) != 1 \
                or (tests[0].group(2) == "!=" and len(tests) > 1):
            error(f"condition {name}: unsupported form {expr!r}")
            return
        self.conditions[name] = {"slot": tests[0].group(1),
                                 "in" if tests[0].group(2) == "==" else "not_in": [int(t.group(3)) for t in tests]}

    def term(self, text):
        """'OP_WARPGROUP[MODE_ARV]`{GMMA_GPR}' -> [term]; a bare CONNECTOR SETS name -> its terms."""
        m = TERM_RE.fullmatch(text)
        name, cond, ports = m.groups() if m else (text, None, None)
        if cond is None and ports is None and name in self.groups:
            return self.groups[name]
        if name not in self.sets:
            error(f"group {text!r}: unknown set {name!r}")
        term = {"set": name, "ports": []}
        for port in filter(None, (p.strip() for p in (ports or "").split(","))):
            connector, _, rng = port.partition("@")
            connector, rng = connector.strip(), rng.strip()
            if connector not in self.connectors:
                error(f"group {text!r}: unknown connector {connector!r}")
            if rng and rng not in self.ranges:
                error(f"group {text!r}: unknown range {rng!r}")
            term["ports"].append([connector, self.ranges.get(rng)])
        if cond is not None:
            if cond not in self.conditions:
                error(f"group {text!r}: unknown condition {cond!r}")
            term["when"] = self.conditions.get(cond)
        return [term]

    def groups_of(self, toks):
        """Header / row tokens -> [{label, terms}]; '+' joins terms into one group."""
        groups, join = [], False
        for tok in toks:
            if tok == "+":
                join = True
                continue
            if join and groups:
                groups[-1]["label"] += " + " + tok
                groups[-1]["terms"] += self.term(tok)
            else:
                groups.append({"label": tok, "terms": self.term(tok)})
            join = False
        return groups

    def add_group(self, stmt):
        name, _, expr = (s.strip() for s in stmt.partition("="))
        found = self.groups_of(tokens(expr))
        if len(found) != 1:
            error(f"connector set {name}: expected one group")
        self.groups[name] = found[0]["terms"] if found else []

    def add_table(self, stmt):
        m = TABLE_RE.fullmatch(stmt)
        if m is None:
            error(f"cannot parse table {stmt[:60]!r}")
            return
        dep, resource, header, body = m.groups()
        if resource not in self.resources:
            error(f"TABLE_{dep}({resource}): unknown resource")
        cols = self.groups_of(tokens(header))
        rows, cells, toks, i = [], [], tokens(body), 0
        is_value = lambda tok: tok.isdigit() or tok in ("ORDERED_ZERO", "-") or tok.startswith("HARD(")
        while i < len(toks):
            j = toks.index(":", i) if ":" in toks[i:] else len(toks)
            k = j + 1
            while k < len(toks) and is_value(toks[k]):
                k += 1
            values = toks[j + 1:k]
            if len(values) == 1:  # one value applies to every column
                values *= len(cols)
            if j == len(toks) or len(values) != len(cols):
                error(f"TABLE_{dep}({resource}): row {' '.join(toks[i:j])[:40]!r} needs {len(cols)} values")
                break
            rows += self.groups_of(toks[i:j])
            cells.append([value(v) for v in values])
            i = k
        for group in rows + cols:
            for t in group["terms"]:
                for connector, _ in t["ports"]:
                    if self.connectors.get(connector, {}).get("resource", resource) != resource:
                        error(f"TABLE_{dep}({resource}): connector {connector} belongs to another resource")
        self.tables.append({"dep": dep, "resource": resource, "rows": rows, "cols": cols, "cells": cells})

    def add_pipeline(self, stmt, section):
        if stmt.startswith("PIPELINE RESOURCE"):
            m = re.fullmatch(r"PIPELINE RESOURCE (\w+) : (\d+)", stmt)
            self.capacity[m.group(1)] = int(m.group(2))
            return
        m = re.fullmatch(r"(\w+) : (\w+) \[(\d+)\]", stmt)
        if m is None or m.group(1) not in self.sets or m.group(2) not in self.capacity:
            error(f"{section}: cannot use {stmt!r}")
            return
        self.uses.append({"set": m.group(1), "resource": m.group(2), "cycles": int(m.group(3))})

    def parse(self, text):
        for section, stmt in statements(text):
            if stmt.startswith("TABLE_"):
                self.add_table(stmt)
            elif stmt.startswith("PIPELINE RESOURCE") or section == "OPERATION PIPELINE RESOURCES":
                self.add_pipeline(stmt, section)
            elif section == "OPERATION SETS":
                self.add_set(stmt)
            elif section in ("HARD RESOURCE", "RESOURCE"):
                self.add_resource(stmt, section == "HARD RESOURCE")
            elif section in ("CONNECTOR NAMES", "CONNECTOR NAME"):
                self.add_connectors(stmt)
            elif section == "CONNECTOR CONDITIONS":
                self.add_condition(stmt)
            elif section == "CONNECTOR SETS":
                self.add_group(stmt)
            else:
                error(f"statement outside any section: {stmt[:60]!r}")


def pinned_sha256(arch):
    """sha256 that intercept_nvdisasm.py pins for this arch's latencies, if any."""
    sys.path.insert(0, str(TOOLS))
    try:
        from intercept_nvdisasm import EXPECTED
    except ImportError:
        return None
    return EXPECTED.get(arch.removeprefix("sm_"), (None, None))[1]


def build(path):
    raw = path.read_bytes()
    sched = Sched()
    sched.parse(raw.decode("utf-8").replace("\r\n", "\n"))
    arch = path.name.removesuffix("_latencies.txt")
    sha256 = hashlib.sha256(raw).hexdigest()
    return {
        "arch": arch,
        "source": {"file": path.name, "sha256": sha256, "verified": sha256 == pinned_sha256(arch)},
        "resources": sched.resources,
        "connectors": sched.connectors,
        "sets": sched.sets,
        "tables": sched.tables,
        "pipeline": {"capacity": sched.capacity, "uses": sched.uses},
    }


def write_json(db, path):
    """One line per top-level key, resource, connector, set and table: compact yet diffable."""
    chunks = []
    for key, value in db.items():
        if key in ("resources", "connectors", "sets"):
            body = "{\n" + ",\n".join(f"{json.dumps(k)}:{compact(v)}" for k, v in value.items()) + "\n}"
        elif key == "tables":
            body = "[\n" + ",\n".join(compact(t) for t in value) + "\n]"
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
    ap.add_argument("latencies", type=Path, help="sm_XX_latencies.txt written by intercept_nvdisasm.py")
    ap.add_argument("--dir", type=Path, default=DB, help="output directory (default: binet/db)")
    args = ap.parse_args()

    db = build(args.latencies)
    for msg in errors[:40]:
        print(f"error: {msg}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} error(s); nothing written", file=sys.stderr)
        return 1
    if not db["source"]["verified"]:
        print(f"warning: {args.latencies.name} is not a verified CUDA 12.9 capture", file=sys.stderr)
    args.dir.mkdir(parents=True, exist_ok=True)
    out = args.dir / f"{db['arch']}.sched.json.gz"
    write_json(db, out)
    print(f"{out}: {len(db['resources'])} resources, {len(db['sets'])} sets, {len(db['tables'])} tables, "
          f"{out.stat().st_size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
