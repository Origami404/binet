"""Inject probes before selected original instructions in a cubin.

Usage: binet mv kernel.cubin --sites 12 48 77 --output prepared.cubin
Sites are positive original instruction indices. Use --kernel-name NAME for an
exact symbol or unique demangled substring; omit only for a single-kernel cubin.
Optional --ring-depth N sets records per warp (positive power of two; default 16).
Creates prepared.cubin and prepared.meta.json; both paths must be new.
Prints the cubin path; pass it to binet profile.
"""
import hashlib
import json
from pathlib import Path

from binet.cuda.core.cubin import Cubin
from binet.cuda.inject import RING_DEPTH, inject
from binet.utils import demangle


def _select_kernel(cubin, selector=None):
    """Select an exact symbol or a unique substring of its demangled signature."""
    if selector is not None and (not isinstance(selector, str) or not selector):
        raise ValueError("kernel selector must be a nonempty string")
    if selector in cubin.kernels:
        return cubin.kernels[selector]
    found = list(cubin.kernels.values())
    if selector is not None:
        found = [k for k, signature in zip(found, demangle([k.name for k in found])) if selector in signature]
    if len(found) != 1:
        raise ValueError(f"expected one kernel for {selector!r}, found {len(found)}; "
                         "select an exact symbol or a unique demangled signature")
    return found[0]


def mv(cubin, sites, output, *, kernel=None, ring_depth=RING_DEPTH):
    """Inject raw sites and write a new cubin plus adjacent .meta.json."""
    cb = cubin if isinstance(cubin, Cubin) else Cubin.load(cubin)
    k = _select_kernel(cb, kernel)
    built = inject(cb, k.name, sites, ring_depth=ring_depth)
    metadata = {"kernel": built.kernel, "arch": built.arch,
                "source_cubin_sha256": built.source_cubin_sha256,
                "image_sha256": hashlib.sha256(built.image).hexdigest(),
                "param_count": built.param_count, "param_offset": built.param_offset,
                "sites": list(built.sites), "ring_depth": built.ring_depth}
    text = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    output = Path(output)
    sidecar = output.with_suffix(".meta.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with output.open("xb") as file:
            created.append(output)
            file.write(built.image)
        with sidecar.open("x", encoding="utf-8") as file:
            created.append(sidecar)
            file.write(text)
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    return output
