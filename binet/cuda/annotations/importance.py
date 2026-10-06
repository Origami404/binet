"""Reading-priority annotations over original instructions, independent of probe placement."""
from binet.cuda.core.instr import Unknown
from binet.utils import demangle


# Include this mapping once as metadata["importance_levels"]. JSON converts the
# integer keys to strings; each Site stores only its integer importance score.
LEVELS = {
    0: "No rule matched",
    1: "Supporting control flow or memory operations",
    2: "Main computation: matrix-multiply families, MUFU",
    3: "Synchronization or work scheduling: barrier wait/arrive, getNextWork*",
}

_COMPUTE = {
    "MUFU", "BMMA", "DMMA", "HMMA", "IMMA", "HFMA2.MMA",
    "BGMMA", "HGMMA", "IGMMA", "QGMMA",
    "UTCHMMA", "UTCIMMA", "UTCMXQMMA", "UTCOMMA", "UTCQMMA",
    "MXQMMA", "OMMA", "QMMA",
}
_SYNC = {
    "BAR", "BSYNC", "WARPSYNC", "MEMBAR", "DEPBAR", "LDGDEPBAR",
    "UCGABARARV", "UCGABARWAIT", "UCGABAR_ARV", "UCGABAR_WAIT",
}
_MEMORY = {
    "LDG", "STG", "LDS", "STS", "LDGSTS", "ATOM", "ATOMS", "RED", "REDS",
    "UTMALDG", "UTMASTG", "UTMAREDG", "UTMAPF",
    "UCGABARGET", "UCGABARSET", "UCGABAR_GET", "UCGABAR_SET",
}


def _score(ins):
    if isinstance(ins, Unknown):
        return 0
    name = ins.mnemonic
    if name == "SYNCS":
        # A barrier's LD/EXCH/CCTL forms are supporting operations; PHASECHK
        # (ONCE or TRYWAIT) and ARRIVE are the synchronization landmarks.
        return 3 if dict(ins.modifiers).get("op") in ("ARRIVE", "PHASECHK") else 1
    if name == "UTCBAR":
        return 1 if dict(ins.modifiers).get("flush") == "FLUSH" else 3
    if name in _SYNC:
        return 3
    if name in _COMPUTE:
        return 2
    return int(name in _MEMORY or ins.branch == "call")


def _work_function(signature):
    # Keep the declared function's name, ignoring template arguments, parameter
    # types and return types that might themselves mention getNextWork.
    head, depth = [], 0
    for char in signature:
        if char == "<":
            depth += 1
        elif char == ">":
            depth = max(0, depth - 1)
        elif not depth:
            if char == "(":
                break
            head.append(char)
    name = "".join(head).rsplit("::", 1)[-1].split()
    return bool(name) and name[-1].startswith("getNextWork")


def annotate(cfg, *, functions=()):
    """Set Site.importance to 0..3; the highest matching rule wins.

    No other annotations are required or changed. All instruction sites are
    scored, including sites that cannot be probed; scores are reading hints,
    not estimates of execution cost or proof that an instruction executes.

    functions accepts Kernel.functions: symbols in this CFG's code section,
    with byte offsets in .value and names in .name. A getNextWork* function's
    entry and resolved call sites score 3. Matching uses demangled names;
    stripped/inlined names and unavailable demangling cannot supply that hint.

    Other calls and backward branch/indirect edges score at least 1. Scores
    stay on their anchors; context windows/grouping belong to the reader.
    Reapplying the pass recomputes scores without retaining old symbol hints.
    """
    functions = list(functions)
    entries = {symbol.value // 16
               for symbol, signature in zip(functions, demangle([s.name for s in functions]))
               if symbol.value % 16 == 0 and 0 <= symbol.value < 16 * len(cfg.sites)
               and _work_function(signature)}
    for site, ins in zip(cfg.sites, cfg.instrs):
        site.importance = _score(ins)
    for index in entries:
        cfg.sites[index].importance = 3
    for edge in cfg.edges:
        if edge.kind == "call" and edge.dst in entries:
            cfg.sites[edge.src].importance = 3
        elif edge.kind in ("branch", "indirect") and edge.dst <= edge.src:
            site = cfg.sites[edge.src]
            site.importance = max(site.importance, 1)
