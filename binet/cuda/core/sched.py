"""Architecture scheduling rules for decoded instructions; no execution-time model."""
from pathlib import Path
from typing import NamedTuple

from binet.cuda.core.instr import Instr, SemanticError, Unknown


class SchedRequirement(NamedTuple):
    """A distance in issue-to-issue cycles, ORDERED_ZERO, or an uninterpreted vendor HARD(value) constraint."""
    kind: str
    value: int | None = None


class SchedResult(NamedTuple):
    """kind: distance / ordered_zero / hard / none / unknown. Only distance and hard carry a value.
    Unknown retains known requirements and its reason; neither it nor none means zero cycles."""
    kind: str
    value: int | None = None
    reason: str | None = None
    requirements: tuple = ()


class Sched:
    """Evaluate structured dependency rules for decoded instructions, without modelling execution time."""
    FILE_RESOURCE = {"R": "GPR", "UR": "UGPR", "P": "PRED", "UP": "UPRED"}
    RESOURCE_FILE = {resource: file for file, resource in FILE_RESOURCE.items()}
    ZERO_ELEMENT = {"GPR": "RZ", "UGPR": "URZ", "PRED": "PT", "UPRED": "UPT"}

    def __init__(self, db, *, path=None):
        self.db, self.arch = db, db["arch"]
        self.path, self.source = Path(path).resolve() if path is not None else None, db.get("source", {})
        self.sets = {name: frozenset(members) for name, members in db["sets"].items()}
        self.resources, self.connectors = db.get("resources", {}), db.get("connectors", {})
        self.defaults = {name: {dep: self._requirement(rec.get("default", {}).get(dep)) for dep in ("ANTI", "OUTPUT")}
                         for name, rec in self.resources.items()}
        self.fixed = {}
        for name, port in self.connectors.items():
            if port.get("resource") not in self.resources or ("elements" in port and
                    (not isinstance(port["elements"], list) or any(not isinstance(e, str) for e in port["elements"]))):
                raise ValueError(f"invalid scheduling connector {name}")
            if "elements" in port:
                elements = set(port["elements"])
                if not elements <= set(self.resources[port["resource"]]["elements"]):
                    raise ValueError(f"connector {name} elements are outside resource {port['resource']}")
                self.fixed[name] = elements - {self.ZERO_ELEMENT.get(port["resource"])}
        self.matrices, self.fixed_groups = {}, {}
        for t in db["tables"]:
            dep, resource = t["dep"], t["resource"]
            if dep not in ("TRUE", "OUTPUT", "ANTI") or resource not in self.resources:
                raise ValueError(f"invalid scheduling table {dep}/{resource}")
            if len(t["cells"]) != len(t["rows"]) or any(len(row) != len(t["cols"]) for row in t["cells"]):
                raise ValueError(f"invalid scheduling matrix dimensions for {dep}/{resource}")
            for written, groups in ((dep != "ANTI", t["rows"]), (dep != "TRUE", t["cols"])):
                self.fixed_groups.setdefault((resource, written), []).extend(groups)
                for group in groups:
                    for term in group["terms"]:
                        self._validate_term(term, resource)
            cells = [[self._requirement(cell) for cell in row] for row in t["cells"]]
            self.matrices.setdefault((dep, resource), []).append((t["rows"], t["cols"], cells))

    @staticmethod
    def _requirement(value):
        if value is None:
            return None
        if type(value) is int and value >= 0:
            return SchedRequirement("distance", value)
        if value == "ORDERED_ZERO":
            return SchedRequirement("ordered_zero")
        if isinstance(value, dict) and set(value) == {"hard"} and type(value["hard"]) is int and value["hard"] >= 0:
            return SchedRequirement("hard", value["hard"])
        raise ValueError(f"invalid scheduling constraint {value!r}")

    def _validate_term(self, term, resource):
        if term.get("set") not in self.sets:
            raise ValueError(f"unknown scheduling operation set {term.get('set')!r}")
        if "when" in term:
            cond = term["when"]
            if not isinstance(cond, dict) or set(cond) not in ({"slot", "in"}, {"slot", "not_in"}) \
                    or not isinstance(cond["slot"], str):
                raise ValueError(f"invalid scheduling condition {cond!r}")
            values = cond.get("in", cond.get("not_in"))
            if not isinstance(values, list) or any(type(value) is not int for value in values):
                raise ValueError(f"invalid scheduling condition values {cond!r}")
        for name, size in term["ports"]:
            if name not in self.connectors or self.connectors[name]["resource"] != resource \
                    or (size is not None and not isinstance(size, str)):
                raise ValueError(f"invalid scheduling port {name!r} for {resource}")

    def _ports(self, ins, group):
        names = ins.cls.opcode_names
        if names is None:
            raise SemanticError(f"{ins.cls.name}: opcode_names are unavailable")
        out = []
        for term in group["terms"]:
            if not self.sets[term["set"]].intersection(names):
                continue
            if "when" in term:
                cond = term["when"]
                if cond["slot"] not in ins.slots or (ins.slots[cond["slot"]] in cond.get("in", cond.get("not_in"))) != ("in" in cond):
                    continue
            for name, size in term["ports"]:
                if (bool(self.fixed[name]) if name in self.fixed else name in ins.cls.slots):
                    out.append((name, size))
        return out

    def _fixed(self, ins, resource, written):
        """Implicit fixed ports contribute even when FORMAT has no ordinary register operand."""
        return {element for group in self.fixed_groups.get((resource, written), ())
                for name, _ in self._ports(ins, group) for element in self.fixed.get(name, ())}

    def _elements(self, ins, ports, file):
        out, errors = set(), []
        for name, size in ports:
            if name in self.fixed:
                out.update(self.fixed[name])
                continue
            try:
                if ins.cls.slots[name].get("file") != file:
                    raise SemanticError(f"{ins.cls.name}: unsupported register connector {name}")
                # No range predicate on a connector means one resource element, not an absent width fact.
                reg = ins._register(name, file, 32 if size is None else size)
                if reg is not None:
                    out.update(reg.names())
            except SemanticError as error:
                errors.append(str(error))
        return out, errors

    @staticmethod
    def _result(requirements, errors=()):
        known = {r for r in requirements if r.kind != "distance"}
        distances = [r.value for r in requirements if r.kind == "distance"]
        if distances:
            known.add(SchedRequirement("distance", max(distances)))
        known = tuple(sorted(known, key=lambda r: (r.kind, -1 if r.value is None else r.value)))
        if errors:
            return SchedResult("unknown", reason="; ".join(dict.fromkeys(errors)), requirements=known)
        if len(known) == 1:
            return SchedResult(known[0].kind, known[0].value, requirements=known)
        return SchedResult("unknown", reason="no supported combination of scheduling constraints", requirements=known)

    def query(self, producer, consumer, dep, file):
        """Constraint for TRUE/OUTPUT/ANTI dependencies on R/UR/P/UP or a declared resource name.
        Numeric distances combine by maximum. Special constraints retain their kind; unsupported mixes,
        missing rule coverage and pseudo-resource semantics return unknown, never a fabricated distance."""
        if not all(isinstance(ins, (Instr, Unknown)) for ins in (producer, consumer)):
            raise TypeError("scheduling queries require decoded instructions")
        resource = self.FILE_RESOURCE.get(file, file) if isinstance(file, str) else None
        if dep not in ("TRUE", "OUTPUT", "ANTI") or resource not in self.resources:
            raise ValueError(f"invalid scheduling query {dep}/{file}")
        if any(isinstance(ins, Instr) and ins.cls.isa.arch != self.arch for ins in (producer, consumer)):
            raise ValueError(f"scheduling query requires architecture {self.arch}")
        if any(isinstance(ins, Unknown) for ins in (producer, consumer)):
            return SchedResult("unknown", reason="unknown instruction effects")
        if resource not in self.RESOURCE_FILE:
            return SchedResult("unknown", reason=f"unsupported scheduling resource {resource}")
        regfile = self.RESOURCE_FILE[resource]
        try:
            def effects(ins, written):
                return {name for reg in ins._regs(written, file=regfile) for name in reg.names()} | self._fixed(ins, resource, written)
            shared = effects(producer, dep != "ANTI") & effects(consumer, dep != "TRUE")
        except SemanticError as error:
            return SchedResult("unknown", reason=str(error))
        if not shared:
            return SchedResult("none")
        covered, requirements, errors = set(), [], []
        for rows, cols, cells in self.matrices.get((dep, resource), ()):
            for i, row in enumerate(rows):
                pports = self._ports(producer, row)
                if not pports:
                    continue
                for j, col in enumerate(cols):
                    cell = cells[i][j]
                    if cell is None:
                        continue
                    cports = self._ports(consumer, col)
                    if not cports:
                        continue
                    pe, perr = self._elements(producer, pports, regfile)
                    ce, cerr = self._elements(consumer, cports, regfile)
                    if (not pe & shared and not perr) or (not ce & shared and not cerr):
                        continue
                    errors.extend(perr + cerr)
                    overlap = pe & ce & shared
                    if overlap:
                        covered.update(overlap)
                        requirements.append(cell)
        # An unresolved matching port might cover a missing element; its default is not established.
        if shared - covered and not errors:
            default = self.defaults[resource].get(dep)
            if default is None:
                errors.append(f"no {dep} rule covers {', '.join(sorted(shared - covered))}")
            else:
                requirements.append(default)
        return self._result(requirements, errors)
