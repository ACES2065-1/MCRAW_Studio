"""One-shot A/B bench: render a 60-frame H.265 NVENC clip via the Python
binding's `mcraw.render()` with MCRAW_GPU_YUV unset, then again with it set
to 1. Compares wallclock so we can verify the GUI path picks up the Phase
C.2 speedup the CLI showed.

Skip if no NVIDIA GPU."""
from __future__ import annotations
import os
import sys
import time
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BUILD = PROJECT / "build"
if str(BUILD) not in sys.path:
    sys.path.insert(0, str(BUILD))
VCPKG_BIN = Path(r"C:\dev\vcpkg\installed\x64-windows\bin")
if VCPKG_BIN.is_dir() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(VCPKG_BIN))

import mcraw  # noqa: E402

if not mcraw.cuda_available():
    print("No CUDA GPU; skipping.")
    sys.exit(0)

fixture = sys.argv[1] if len(sys.argv) > 1 else "VIDEO_20250114_130136.0.mcraw"
NFRAMES = 60
CODEC = "h265_nvenc"
CS = "rec709-display"

def bench(label: str) -> float:
    with tempfile.TemporaryDirectory(prefix="mcraw_bench_") as td:
        out = Path(td) / f"out_{label}.mp4"
        t0 = time.time()
        mcraw.render(input=fixture, output=str(out),
                     colorspace=CS, codec=CODEC,
                     start=0, end=NFRAMES, bitrate=80)
        dt = time.time() - t0
        sz = out.stat().st_size
    fps = NFRAMES / dt
    print(f"  {label:18s}: {dt:.2f}s  ({fps:5.1f} fps, {sz:,} bytes)")
    return dt

print(f"Fixture: {fixture}, {NFRAMES} frames, {CODEC}, {CS}")

# Force CPU path first.
os.environ["MCRAW_GPU_YUV"] = "0"
print(f"MCRAW_GPU_YUV=0  (CPU pipeline + NVENC)")
t_cpu = bench("cpu_path")

# Then GPU path (Phase C.2).
os.environ["MCRAW_GPU_YUV"] = "1"
print(f"MCRAW_GPU_YUV=1  (Phase C.2 GPU pipeline)")
t_gpu = bench("gpu_path")

if t_gpu > 0:
    print(f"\nSpeedup: {t_cpu / t_gpu:.2f}x")
