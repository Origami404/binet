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


def _launch_metadata(event):
    """Return grid/block metadata for one GPU kernel event, or None.

    Torch 2.10 removed activity_type()/extra_meta(); fall back to device_type()
    plus metadata_json, whose pretty-printed fragment carries grid and block.
    """
    import json

    activity = getattr(event, "activity_type", None)
    if activity is not None:
        if activity() != "kernel":
            return None
        metadata = event.extra_meta()
    else:
        from torch.autograd import DeviceType

        if event.device_type() != DeviceType.CUDA or event.duration_ns() <= 0:
            return None
        raw = event.metadata_json()
        if '"grid"' not in raw or '"block"' not in raw:
            return None
        metadata = json.loads("{" + raw + "}")
    return metadata


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
        metadata = _launch_metadata(event)
        if metadata is None:
            continue
        grid = tuple(json.loads(metadata["grid"]) if isinstance(metadata["grid"], str) else metadata["grid"])
        block = tuple(json.loads(metadata["block"]) if isinstance(metadata["block"], str) else metadata["block"])
        launches.add((event.name(), grid, block))
    for name, grid, block in sorted(launches):
        print(f"{name}\tgrid={grid}\tblock={block}")
    if not launches:
        print("binet: no CUDA kernels observed", file=sys.stderr)
    return 0 if launches else 3


if __name__ == "__main__":
    raise SystemExit(_run(sys.argv[1], sys.argv[2:]))
