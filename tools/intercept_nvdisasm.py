#!/usr/bin/env python3
"""Dump built-in SASS instruction and scheduling descriptions from CUDA 12.9 nvdisasm."""
import argparse
import hashlib
import io
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
RAW = TOOLS.parent / "assets" / "raw"
CACHE = TOOLS / ".nvdisasm-12.9.88"
SHIM_SRC = TOOLS / "intercept_sass.cc"

PINNED_BUILD = "12.9.88"
REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvdisasm/"
# platform.machine() -> (archive under REDIST, sha256), from redistrib_12.9.1.json
ARCHIVES = {
    "x86_64": ("linux-x86_64/cuda_nvdisasm-linux-x86_64-12.9.88-archive.tar.xz",
               "49296dd550e05434185a8588ec639f1325b2de413e2321ddd7e56c5182a476ff"),
    "aarch64": ("linux-sbsa/cuda_nvdisasm-linux-sbsa-12.9.88-archive.tar.xz",
                "28b2597f0901cfafcd050cba0877c1eb5edcd7ebd8164aea356cec832e636ee3"),
}
# sha256 of (instructions, latencies) as produced with nvdisasm 12.9.88.
EXPECTED = {
    "80": ("142bfb2c9aebc7caa13913c52a28d04a5a347bf188640a852a599c7da05251ef",
           "68f1f43ec557286bfca7b73bf0c0257ec3fa917b278417aa7d898bcd55f6850f"),
    "90": ("f4f95102a2e07e931b5ec9cbcd3f6264d3c64b9173f2da4efc3ebd07bcde7048",
           "81f021795de5a58b76932e2bb117bf368e8e5146909f7f5e9c0ab35306b1427e"),
    "100": ("5825ca097637b7f4f19442e37a46efc6e80276dd3984381c91eb06369fc7d72a",
            "8edced09da9ae25e8d576391e605c718a28b2594a649416ba62cc596d1e85292"),
    "103": ("445038bc6806a4ee9c1bc27966db467b57af8528ec9a9c83e7f4e667d4a50acf",
            "bfdd87e83c26c5e8d6b19061f845565d2d578545789a66f262694237f6eab657"),
    "120": ("78a1e10e710eb6493aed57141b96378c12ab2cd4f8fc217dceb3eb3c8c3271fc",
            "6298d9ca8a05f75f76c25bac8da17d6586175261ebbb607bd49835b6879e961b"),
}


def nvdisasm_build(path):
    """Return the build ('12.9.88') of an nvdisasm binary, or None."""
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True, text=True).stdout
    except OSError:
        return None
    m = re.search(r"\bV(\d+\.\d+\.\d+)", out)
    return m.group(1) if m else None


def download_nvdisasm():
    machine = platform.machine()
    if machine not in ARCHIVES:
        sys.exit(f"no nvdisasm {PINNED_BUILD} download for {machine}")
    rel, sha256 = ARCHIVES[machine]
    print(f"downloading {REDIST}{rel}")
    with urllib.request.urlopen(REDIST + rel) as resp:
        data = resp.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        sys.exit(f"sha256 mismatch for {rel}")
    exe = CACHE / "bin" / "nvdisasm"
    exe.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("/bin/nvdisasm"))
        exe.write_bytes(tar.extractfile(member).read())
    exe.chmod(0o755)
    return exe


def find_nvdisasm():
    """Use a CUDA 12.9 nvdisasm from $CUDA_HOME or the tools/ cache, else download one."""
    candidates = [CACHE / "bin" / "nvdisasm"]
    if os.environ.get("CUDA_HOME"):
        candidates.insert(0, Path(os.environ["CUDA_HOME"]) / "bin" / "nvdisasm")
    for path in candidates:
        build = nvdisasm_build(path) if path.is_file() else None
        if build and build.startswith("12.9."):
            return path, build
        if build:
            print(f"skipping {path}: nvdisasm {build}, need 12.9")
    return download_nvdisasm(), PINNED_BUILD


def build_shim():
    """Compile intercept_sass.cc into the cache unless an up-to-date build exists."""
    lib = CACHE / "intercept_sass.so"
    if not lib.exists() or lib.stat().st_mtime < SHIM_SRC.stat().st_mtime:
        CACHE.mkdir(parents=True, exist_ok=True)
        cxx = os.environ.get("CXX", "c++")
        subprocess.run([cxx, "-std=c++17", "-O2", "-fPIC", "-shared", "-o", str(lib), str(SHIM_SRC), "-ldl"],
                       check=True)
    return lib


def trim(raw, head):
    """Drop the NUL chunk terminators; keep `head` up to the last ';', plus a newline."""
    text = raw.replace(b"\0", b"")
    start, end = text.find(head), text.rfind(b";")
    return text[start:end + 1] + b"\n" if 0 <= start < end else None


def capture(nvdisasm, shim, arch):
    """Run nvdisasm under the shim; return (instructions, latencies), None if missing."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zero = tmp / "zero.bin"
        zero.write_bytes(bytes(16))
        env = dict(os.environ, LD_PRELOAD=str(shim), INTERCEPT_SASS_DIR=str(tmp), OMP_NUM_THREADS="1")
        # nvdisasm rejects the all-zero instruction ("Illegal instruction found")
        # only after loading both descriptions, so its exit status is ignored.
        subprocess.run([str(nvdisasm), "-b", f"SM{arch}", "-json", str(zero)], env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        texts = []
        for name, head in (("instructions.raw", b"ARCHITECTURE"), ("latencies.raw", b"OPERATION SETS")):
            raw = tmp / name
            texts.append(trim(raw.read_bytes(), head) if raw.exists() else None)
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("archs", nargs="+", help="target architectures, e.g. 90 100 103 (or sm_90, SM90a)")
    ap.add_argument("--dir", type=Path, default=RAW, help="output directory (default: assets/raw)")
    args = ap.parse_args()
    if platform.system() != "Linux":
        sys.exit("intercept_nvdisasm.py needs Linux (LD_PRELOAD)")

    nvdisasm, build = find_nvdisasm()
    shim = build_shim()
    print(f"nvdisasm {build}: {nvdisasm}")
    args.dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for arch in args.archs:
        arch = re.sub(r"(?i)^sm_?", "", arch)
        instructions, latencies = capture(nvdisasm, shim, arch)
        if instructions is None or latencies is None or not re.search(rb"ELF_VERSION 129\b", instructions):
            print(f"sm_{arch}: capture failed (instructions {'ok' if instructions else 'missing'}, "
                  f"latencies {'ok' if latencies else 'missing'})")
            failed += 1
            continue
        digests = tuple(hashlib.sha256(t).hexdigest() for t in (instructions, latencies))
        expected = EXPECTED.get(arch)
        if build == PINNED_BUILD and expected and digests != expected:
            print(f"sm_{arch}: output differs from the verified {PINNED_BUILD} capture: {digests}")
            failed += 1
            continue
        for kind, text in (("instructions", instructions), ("latencies", latencies)):
            (args.dir / f"sm_{arch}_{kind}.txt").write_bytes(text)
        classes = len(re.findall(rb'(?:^|;)[ \t]*(?:ALTERNATE )?CLASS[ \t]+"', instructions, re.M))
        note = "verified" if build == PINNED_BUILD and expected else f"unverified: {digests}"
        print(f"sm_{arch}: {classes} classes, {len(instructions)} + {len(latencies)} bytes ({note})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
