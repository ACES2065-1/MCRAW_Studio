#!/usr/bin/env python3
"""
Phase J verification: the ProRes encoder farm is bit-identical to serial
encoding and actually faster.

For prores4444 and prores422hq, renders the same 16-frame range with
encoder_instances=1 (serial) and =0 (auto farm), in BOTH MCRAW_GPU_YUV
states (planar submits vs float submits):
  - decoded raw video must hash identical (prores_ks is intra-only with
    per-frame bit targeting -> the farm must not change a single bit);
  - ffprobe frame count, duration and audio codec must match;
  - the farm must be > 1.8x faster for prores4444 (hard gate; measured
    on the GPU-fed pass where the encoder is the only bottleneck).

Renders run in subprocesses so the MCRAW_GPU_YUV gate is evaluated
freshly per render.

Usage:  python test/prores_farm_verify.py [clip.mcraw]
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BUILD = Path(os.environ.get("MCRAW_BUILD_DIR", PROJECT / "build"))
VCPKG_BIN = Path(os.environ.get("MCRAW_DLL_DIR",
                                r"C:\dev\vcpkg\installed\x64-windows\bin"))
if VCPKG_BIN.is_dir() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(VCPKG_BIN))
if BUILD.is_dir() and str(BUILD) not in sys.path:
    sys.path.insert(0, str(BUILD))

import mcraw  # noqa: E402  (also validates the module loads)

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
N = 16
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:36s} {detail}")
    if not ok:
        FAILURES.append(name)


def render(clip: Path, out: Path, codec: str, instances: int,
           gpu: bool) -> float:
    env = dict(os.environ)
    env.pop("MCRAW_GPU_YUV", None)
    if gpu:
        env["MCRAW_GPU_YUV"] = "1"
    code = (
        "import os, sys;"
        f"sys.path.insert(0, r'{BUILD}');"
        f"os.add_dll_directory(r'{VCPKG_BIN}');"
        "import mcraw;"
        f"mcraw.render(input=r'{clip}', output=r'{out}', "
        f"colorspace='acescg', codec='{codec}', start=0, end={N}, "
        f"encoder_instances={instances})"
    )
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"render {codec} instances={instances} gpu={gpu} failed:\n"
            f"{proc.stderr}")
    return dt


def decoded_hash(path: Path) -> str:
    out = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-map", "0:v:0",
         "-f", "rawvideo", "-"],
        capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode(errors="replace"))
    return hashlib.sha256(out.stdout).hexdigest()


def probe(path: Path) -> str:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,nb_frames,duration",
         "-of", "csv=p=0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", str(path)],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()


def main() -> int:
    clip = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        PROJECT / "VIDEO_20250114_130136.0.mcraw"
    if not clip.is_file():
        print(f"ERROR: clip not found: {clip}", file=sys.stderr)
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="farm_"))

    try:
        for gpu in (True, False):
            feed = "gpu" if gpu else "cpu"
            if gpu and not (mcraw.cuda_built() and mcraw.cuda_available()):
                print(f"  [SKIP] {feed} feed — no CUDA device.")
                continue
            for codec in ("prores4444", "prores422hq"):
                serial = tmp / f"{codec}_{feed}_s.mov"
                farm = tmp / f"{codec}_{feed}_f.mov"
                t_serial = render(clip, serial, codec, 1, gpu)
                t_farm = render(clip, farm, codec, 0, gpu)

                check(f"{codec} {feed} bit-identical",
                      decoded_hash(serial) == decoded_hash(farm),
                      f"decoded hash equal")
                check(f"{codec} {feed} probe equal",
                      probe(serial) == probe(farm), probe(farm))
                speed = t_serial / max(t_farm, 1e-9)
                if codec == "prores4444" and gpu:
                    check(f"{codec} {feed} farm speedup", speed > 1.8,
                          f"{N/t_serial:.1f} -> {N/t_farm:.1f} fps "
                          f"({speed:.1f}x)")
                else:
                    print(f"  [INFO] {codec} {feed} farm speedup       "
                          f"{N/t_serial:.1f} -> {N/t_farm:.1f} fps "
                          f"({speed:.1f}x)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
