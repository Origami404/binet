"""Annotate the warps that may reach each original instruction."""
from collections import deque

from binet.cuda.core.instr import Reg, SemanticError, Unknown

ALL = (1 << 32) - 1


def _operand(ins, state, name):
    spec = ins.cls.slots.get(name)
    if spec is None:
        return None
    value, file = ins.slots[name], spec.get("file")
    if any(v for k, v in ins.slots.items() if k.startswith(name + "@") and k != name + "@not"):
        return None
    if file:
        result = ((1, 1) if file in ("P", "UP") else (0, 0)) if value == ins.cls.isa.zero[file] else state.get(Reg(file, value))
    else:
        result = (value & ALL, value & ALL) if spec["type"] in ("UImm", "SImm") else None
    if result is not None and ins.slots.get(name + "@not"):
        result = (1 - result[1], 1 - result[0])
    return result


def _truth(value):
    return bool(value[0]) if value is not None and value[0] == value[1] else None


def _guard(ins, state):
    return True if ins.cls.guard_slot is None else _truth(_operand(ins, state, ins.cls.guard_slot))


def _compare(a, b, op, signed):
    if a is None or b is None:
        return None
    if signed:
        def convert(v):
            return (v[0] - (1 << 32), v[1] - (1 << 32)) if v[0] >= 1 << 31 else v if v[1] < 1 << 31 else None
        a, b = convert(a), convert(b)
        if a is None or b is None:
            return None
    if op in ("GT", "LE"):
        return _compare(b, a, "LT" if op == "GT" else "GE", False)
    if op == "LT":
        return True if a[1] < b[0] else False if a[0] >= b[1] else None
    if op == "GE":
        v = _compare(a, b, "LT", False)
    elif op in ("EQ", "NE"):
        v = False if a[1] < b[0] or b[1] < a[0] else True if a[0] == a[1] == b[0] == b[1] else None
        if op == "EQ":
            return v
    else:
        return None
    return None if v is None else not v


def _result(ins, state, tids, entry):
    """Known complete destination values; all other writes are killed by the caller."""
    mods, slots = dict(ins.modifiers), ins.slots
    def operand(*names):
        return next((_operand(ins, state, n) for n in names if n in ins.cls.slots), None)
    a, b, c = operand("Ra", "URa", "Sa"), operand("Rb", "URb", "Sb"), operand("Rc", "URc", "Sc")
    value = None
    name = ins.mnemonic
    if name == "S2R":
        sr = ins.cls.isa.enum_name(ins.cls.slots["SRa"]["type"], slots["SRa"])
        value = tids.get(sr)
    elif name in ("MOV", "UMOV"):
        value = b
    elif name == "R2UR" and ins.cls.name == "r2ur__noOR":
        value = a
    elif name == "SHFL" and entry and mods.get("shflmd") == "IDX" and b == (0, 0) and c == (31, 31):
        # Lane zero is active at entry, including in a partial final warp.
        value = a
    elif name in ("SHF", "USHF") and mods.get("fmt") == "U32" and mods.get("xor", "noxor") == "noxor":
        source = c if mods.get("hilo") == "HI" and a == (0, 0) else a if mods.get("hilo") == "LO" and c == (0, 0) else None
        if source is not None and b is not None and b[0] == b[1] and b[1] < 32:
            if mods.get("dir") == "R":
                value = (source[0] >> b[0], source[1] >> b[0])
            elif mods.get("dir") == "L" and source[1] << b[0] <= ALL:
                value = (source[0] << b[0], source[1] << b[0])
    elif name in ("IMAD", "UIMAD", "IADD3", "UIADD3") and mods.get("wide", "LO") == "LO" and "X" not in mods:
        if name.endswith("MAD"):
            a = (0, 0) if a == (0, 0) or b == (0, 0) else (a[0] * b[0], a[1] * b[1]) if a is not None and b is not None else None
        else:
            a = (a[0] + b[0], a[1] + b[1]) if a is not None and b is not None else None
        if a is not None and c is not None:
            lo, hi = a[0] + c[0], a[1] + c[1]
            value = (lo & ALL, hi & ALL) if lo == hi or hi <= ALL else None
    elif name in ("LOP3", "ULOP3") and "imm8" in slots and all(v is not None and v[0] == v[1] for v in (a, b, c)):
        x, y, z = a[0], b[0], c[0]
        bits = 0
        for i in range(8):
            if slots["imm8"] >> i & 1:
                bits |= (x if i & 4 else ~x) & (y if i & 2 else ~y) & (z if i & 1 else ~z)
        value = (bits & ALL, bits & ALL)
    elif name in ("ISETP", "UISETP") and "ex" not in mods:
        # Limit boolean composition to the common AND PT form.
        if mods.get("bop") == "AND" and operand("Pp", "UPp") == (1, 1):
            test = _compare(a, b, mods.get("icmp"), mods.get("fmt") == "S32")
            if test is not None:
                file = "UP" if name == "UISETP" else "P"
                return {Reg(file, slots[file + "u"]): (int(test), int(test)),
                        Reg(file, slots[file + "v"]): (int(not test), int(not test))}
    dst = "URd" if "URd" in slots else "Rd"
    return {Reg("UR" if dst == "URd" else "R", slots[dst]): value} if value is not None and dst in slots else {}


def _step(ins, state, tids, entry):
    try:
        guard = _guard(ins, state)
        if guard is False:
            return state
        writes = ins.writes
        known = _result(ins, state, tids, entry) if guard is True else {}
        complete = set(ins.definite_writes)
    except SemanticError:
        return {}
    result = dict(state)
    for r in writes:
        for index in range(r.base, r.base + r.count):
            result.pop(Reg(r.file, index), None)
    result.update((r, value) for r, value in known.items() if r in complete)
    return result


def _branch(ins, state):
    # Only plain predicate branches; lane selectors retain both edges.
    if (ins.mnemonic != "BRA" or "URb" in ins.slots or ins.slots.get("nottid0", 0)
            or dict(ins.modifiers).get("depth", "nodepth") != "nodepth"):
        return None
    values = [_guard(ins, state)] + [_truth(_operand(ins, state, n)) for n in ins.cls.predicates]
    return False if False in values else True if all(v is True for v in values) else None


def annotate(cfg, *, block=(1024, 1, 1)):
    """Set conservative warp masks without changing resource annotations or probes.

    block supplies the coordinate mapping only. For n launched warps, callers
    interpret a site's mask as site.warp_mask & ((1 << n) - 1).
    """
    block = tuple(block)
    if len(block) != 3 or any(type(x) is not int or x <= 0 for x in block):
        raise ValueError("block dimensions must be three positive integers")
    incomplete = bool(cfg.unresolved) or any(isinstance(i, Unknown) for i in cfg.instrs)
    for site in cfg.sites:
        site.warp_mask = ALL if incomplete else 0
    if not cfg.sites or incomplete:
        return
    edges = [[] for _ in cfg.blocks]
    for edge in cfg.edges:
        edges[cfg.block_of[edge.src]].append(edge)
    for warp in range(32):
        tids = {"SR_LANEID": (0, 31)}
        x, y, _ = block
        coordinates = [(t % x, t // x % y, t // (x * y)) for t in range(warp * 32, (warp + 1) * 32)]
        tids.update(("SR_TID." + axis, (min(v[j] for v in coordinates), max(v[j] for v in coordinates))) for j, axis in enumerate("XYZ"))
        incoming = [None] * len(cfg.blocks)
        queue = deque()
        for function in cfg.functions:
            bid = cfg.block_of[function]
            incoming[bid] = {}
            queue.append(bid)
        while queue:
            bid = queue.popleft()
            blk, state = cfg.blocks[bid], incoming[bid]
            before = state
            for i in range(blk.start, blk.end):
                cfg.sites[i].warp_mask |= 1 << warp
                before = state
                state = _step(cfg.instrs[i], state, tids, blk.start == 0 and not blk.preds)
            ins = cfg.instrs[blk.end - 1]
            taken = _branch(ins, before) if ins.branch == "branch" else None
            for edge in edges[bid]:
                if ins.branch == "branch" and ((edge.kind == "fall" and taken is True) or (edge.kind != "fall" and taken is False)):
                    continue
                outgoing = state
                if ins.branch in ("call", "return"):
                    outgoing = {}  # No interprocedural value summaries.
                elif ins.branch == "branch" and taken is None:
                    # Divergent siblings can write shared uniform registers.
                    outgoing = {r: v for r, v in state.items() if r.file not in ("UR", "UP")}
                dst = cfg.block_of[edge.dst]
                old = incoming[dst]
                joined = dict(outgoing) if old is None else {r: v for r, v in old.items() if outgoing.get(r) == v}
                if old is None or joined != old:
                    incoming[dst] = joined
                    queue.append(dst)
