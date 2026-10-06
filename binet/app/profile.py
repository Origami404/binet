"""Run a prepared kernel patch and capture the first matching CUDA launch.

Usage: binet profile --cubin prepared.cubin --output run-001 -- python workload.py [ARGS...]
Input must come from binet mv, with adjacent prepared.meta.json.
The output directory must be new; omitting --output uses ~/.binet/profile-TIMESTAMP.
Writes profile.json (session report) and trace.npz on capture; prints the trace path.
Runs the workload in the chosen Python interpreter, redirecting its stdout to stderr;
interpreter flags, -m and -c are unsupported. Exits 0 on capture, 3 on no match,
or nonzero on workload/profiling failure.
"""
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback


def profile(cubin, command, output):
    """Run python SCRIPT [ARGS...] in an explicit child and return its session report."""
    command = [os.fspath(value) for value in command]
    if len(command) < 2 or str(command[1]).startswith("-"):
        raise ValueError("profile requires python SCRIPT [ARGS...]; interpreter flags, -m and -c are not supported")
    cubin, output = Path(cubin).resolve(), Path(output).resolve()
    output.mkdir(parents=True)
    python, script, *args = command
    status = subprocess.call([python, str(Path(__file__).resolve()), str(cubin), str(output), script, *args])
    status = status if status >= 0 else 128 - status
    report = output / "profile.json"
    try:
        result = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        result = dict(command=list(command), instrumented_image=str(cubin), traces=[], warnings=[],
                      errors=[f"child did not produce a complete profile report: {error}"],
                      target_exit_code=status, exit_code=status or 1)
        from binet.utils import write_json
        write_json(report, result)
    # A signal or abrupt exit must not be hidden by a report written earlier.
    if status:
        result["exit_code"] = status
    return result


def _load(cubin):
    from binet.cuda.inject import InjectedKernel
    from binet.records import depth_checked, sites_checked

    image = cubin.read_bytes()
    metadata = json.loads(cubin.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if hashlib.sha256(image).hexdigest() != metadata["image_sha256"]:
        raise ValueError("prepared cubin SHA256 differs from its metadata")
    sites = sites_checked(metadata["sites"])
    if tuple(metadata["sites"]) != sites:
        raise ValueError("prepared cubin sites must be unique and sorted")
    built = InjectedKernel(image=image, kernel=metadata["kernel"], arch=metadata["arch"],
                           source_cubin_sha256=metadata["source_cubin_sha256"],
                           param_count=metadata["param_count"], param_offset=metadata["param_offset"],
                           sites=sites, ring_depth=depth_checked(metadata["ring_depth"]))
    return built


def _save_capture(capture, output, injected):
    from binet.trace import Trace

    path = output / "trace.npz"
    metadata = injected.metadata(capture.grid, capture.block)
    # The completed CPU tensor exposes the device layout without another raw copy.
    trace = Trace.from_buffer(capture.data.numpy(), metadata)
    temporary = path.with_suffix(".tmp")
    try:
        trace.save(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(file=path.name, kernel=injected.kernel, record_count=len(trace.records))


def _run(cubin, output, script, args):
    from binet.cuda.profiler import CUDAProfiler
    from binet.utils import write_json

    cubin, output = Path(cubin), Path(output)
    result = dict(command=[sys.executable, script, *args], instrumented_image=str(cubin),
                  traces=[], errors=[], warnings=[], target_exit_code=0)
    traces, collector = [], None
    try:
        injected = _load(cubin)
        result.update(kernel=injected.kernel, sites=list(injected.sites), ring_depth=injected.ring_depth)
        collector = CUDAProfiler(injected, on_capture=lambda capture: traces.append(_save_capture(
            capture, output, injected)))
        sys.argv = [script, *args]
        sys.path[0] = str(Path(script).resolve().parent)
        with collector, redirect_stdout(sys.stderr):
            try:
                runpy.run_path(script, run_name="__main__")
            except SystemExit as error:
                if error.code is not None:
                    if isinstance(error.code, int):
                        result["target_exit_code"] = error.code % 256
                    else:
                        print(error.code, file=sys.stderr)
                        result["target_exit_code"] = 1
            except KeyboardInterrupt:
                traceback.print_exc()
                result["target_exit_code"] = 130
            except BaseException:
                traceback.print_exc()
                result["target_exit_code"] = 1
    except Exception as error:
        result["errors"].append(f"profile failed: {error}")
    finally:
        if collector is not None:
            result["errors"].extend(collector.errors)
    result["traces"] = traces
    result["exit_code"] = (result["target_exit_code"] or (1 if result["errors"] else 0 if traces else 3))
    temporary = output / "profile.tmp"
    try:
        write_json(temporary, result)
        temporary.replace(output / "profile.json")
    finally:
        temporary.unlink(missing_ok=True)
    return result["exit_code"]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(_run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]))
