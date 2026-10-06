"""List CUDA kernel names and launch dimensions using the Torch profiler.

Usage: binet ls -- python workload.py [ARGS...]
Requires Torch with CUDA in the chosen Python interpreter; interpreter flags,
-m and -c are unsupported. Prints sorted, unique name/grid/block rows to stdout;
workload stdout is redirected to stderr. Exits 0 if kernels were observed,
3 if none were observed, otherwise propagates the child failure status.
"""
from contextlib import redirect_stdout
import json
from pathlib import Path
import runpy
import subprocess
import sys


def ls(command):
    """Run python SCRIPT [ARGS...] in a child profiler; return its exit status."""
    if len(command) < 2 or str(command[1]).startswith("-"):
        raise ValueError("ls requires python SCRIPT [ARGS...]; interpreter flags, -m and -c are not supported")
    python, script, *args = command
    status = subprocess.call([python, str(Path(__file__).resolve()), script, *args])
    return status if status >= 0 else 128 - status


def _run(script, args):
    import torch
    from torch.profiler import profile, ProfilerActivity, _ExperimentalConfig

    sys.argv = [script, *args]
    sys.path[0] = str(Path(script).resolve().parent)
    with profile(activities=[ProfilerActivity.CUDA], experimental_config=_ExperimentalConfig(
            expose_kineto_event_metadata=True)) as prof, redirect_stdout(sys.stderr):
        try:
            runpy.run_path(script, run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        torch.cuda.synchronize()

    # Kineto exposes launch metadata directly; no trace export is needed.
    launches = set()
    for event in prof.profiler.kineto_results.events():
        if event.activity_type() == "kernel":
            metadata = event.extra_meta()
            grid = tuple(json.loads(metadata["grid"]))
            block = tuple(json.loads(metadata["block"]))
            launches.add((event.name(), grid, block))
    for name, grid, block in sorted(launches):
        print(f"{name}\tgrid={grid}\tblock={block}")
    if not launches:
        print("binet: no CUDA kernels observed", file=sys.stderr)
    return 0 if launches else 3


if __name__ == "__main__":
    raise SystemExit(_run(sys.argv[1], sys.argv[2:]))
