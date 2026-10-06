"""Capture one loaded cubin containing a kernel and annotate that kernel."""
from contextlib import redirect_stdout
from functools import partial
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


def get(command, output, kernel_name, *, num_threads=1024):
    """Run python SCRIPT [ARGS...] and save a cubin plus .info.json for num_threads per block."""
    if len(command) < 2 or str(command[1]).startswith("-"):
        raise ValueError("get requires python SCRIPT [ARGS...]; interpreter flags, -m and -c are not supported")
    if not isinstance(kernel_name, str) or not kernel_name:
        raise ValueError("kernel name must be a nonempty string")
    if type(num_threads) is not int or not 1 <= num_threads <= 1024:
        raise ValueError("num_threads must be an integer between 1 and 1024 (threads per block)")
    python, script, *args = command
    status = subprocess.call([python, str(Path(__file__).resolve()), kernel_name,
                              str(Path(output).resolve()), str(num_threads), script, *args])
    return status if status >= 0 else 128 - status


def _analyse(cubin, name, *, num_threads=1024):
    from binet.cuda.annotations import hazard, importance, liveness, warps
    from binet.cuda.core.cfg import CFG
    from binet.cuda.inject import check_site
    from binet.cuda.isa import Isa

    kernel, isa = cubin.kernel(name), Isa.load(cubin.arch)
    cfg = CFG.of_kernel(kernel, isa).apply(
        partial(liveness.annotate, regcount=kernel.regcount),
        partial(hazard.annotate, isa=isa),
        warps.annotate,
        partial(importance.annotate, functions=kernel.functions),
    )
    warp_mask = (1 << ((num_threads + 31) // 32)) - 1
    for site in cfg.sites:
        site.warp_mask &= warp_mask
    sites = [{"site": site.index, "mnemonic": "?" if ins.cls is None else ins.mnemonic,
              "probeable": check_site(cfg, site.index)[0], "warp_mask": site.warp_mask,
              "importance": site.importance}
             for ins, site in zip(cfg.instrs, cfg.sites)]
    return {"version": 1, "cubin_sha256": hashlib.sha256(cubin.image).hexdigest(), "kernel": name,
            "importance_levels": importance.LEVELS.copy(), "sites": sites}


def _run(kernel_name, output, script, args, *, num_threads=1024):
    from binet.cuda.core.cubin import Cubin
    from binet.cuda.hook import module_loads
    from binet.utils import demangle

    seen, matches = set(), {}

    def collect(context, module_id, image):
        digest = hashlib.sha256(image).hexdigest()
        if digest in seen:
            return
        seen.add(digest)
        cubin = Cubin(image)
        names = list(cubin.kernels)
        selected = ([kernel_name] if kernel_name in cubin.kernels else
                    [name for name, signature in zip(names, demangle(names)) if signature == kernel_name])
        for name in selected:
            matches[(digest, name)] = cubin

    sys.argv = [script, *args]
    sys.path[0] = str(Path(script).resolve().parent)
    with module_loads(collect), redirect_stdout(sys.stderr):
        try:
            runpy.run_path(script, run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise

    if not matches:
        print(f"binet: kernel {kernel_name!r} not found in loaded modules")
        return 3
    if len(matches) != 1:
        print(f"binet: kernel {kernel_name!r} is ambiguous across {len(matches)} candidates:")
        for digest, name in sorted(matches):
            print(f"  {digest} {name}")
        return 2

    (_, name), cubin = next(iter(matches.items()))
    annotations = _analyse(cubin, name, num_threads=num_threads)
    path = Path(output)
    sidecar = path.with_suffix(".info.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with path.open("xb") as stream:
            created.append(path)
            stream.write(cubin.image)
        with sidecar.open("x", encoding="utf-8") as stream:
            created.append(sidecar)
            json.dump(annotations, stream, indent=2)
            stream.write("\n")
    except BaseException:
        for entry in reversed(created):
            entry.unlink(missing_ok=True)
        raise
    print(path)
    print(sidecar)
    return 0


if __name__ == "__main__":
    # This file is the explicit child entry point, including for another Python
    # environment where Binet itself has not been installed.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(_run(sys.argv[1], sys.argv[2], sys.argv[4], sys.argv[5:], num_threads=int(sys.argv[3])))
