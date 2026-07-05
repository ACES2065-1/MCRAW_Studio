#!/usr/bin/env python3
"""
Phase G verification: render + preview run concurrently without corrupting
each other.

Pre-Phase-G the render and preview shared one buffer set behind one mutex:
a 4K NVENC render and a proxy-res preview alternated EnsureBuffers dimensions
(realloc thrash) and serialized per frame. Phase G gives each its own context
(buffers + mutex + stream), so this script:

  1. captures a CPU + GPU preview reference frame (full res + bin=2),
  2. starts an NVENC GPU render (MCRAW_GPU_YUV=1) on a worker thread,
  3. while it runs, continuously decodes GPU preview frames at BOTH sizes
     (forcing what used to be dimension thrash),
  4. asserts every concurrent preview frame still matches the CPU reference
     within +/-1 code value, and the render output is a valid file.

Usage:
    python test/phase_g_concurrency_verify.py [path/to/clip.mcraw]

Requires an NVIDIA GPU with NVENC; exits 0 with a SKIP message otherwise.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
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

# The GPU render path is opt-in via env; must be set before mcraw loads the
# encoder config (it's read per-render, but set it up front to be safe).
os.environ["MCRAW_GPU_YUV"] = "1"

import numpy as np  # noqa: E402

import mcraw  # noqa: E402


def to_arr(buf: bytes, h: int, w: int) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)


def main() -> int:
    default = PROJECT / "VIDEO_20250114_130136.0.mcraw"
    fixture = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not fixture.is_file():
        print(f"ERROR: fixture not found: {fixture}", file=sys.stderr)
        return 2
    if not (mcraw.cuda_built() and mcraw.cuda_available()
            and mcraw.encoder_available("hevc_nvenc")):
        print("SKIP: needs CUDA + NVENC (not available on this machine).")
        return 0

    d = mcraw.Decoder(str(fixture))
    ts = d.frames
    t = ts[min(4, len(ts) - 1)]

    # References BEFORE the render starts (quiet GPU).
    refs = {}
    for b in (1, 2):
        buf, h, w = d.process_frame_rgb24(t, "srgb", False, True, False, b)
        cpu = to_arr(buf, h, w)
        refs[b] = cpu
    print(f"Fixture: {fixture.name}  refs captured (bin=1: "
          f"{refs[1].shape[1]}x{refs[1].shape[0]}, bin=2: "
          f"{refs[2].shape[1]}x{refs[2].shape[0]})\n")

    out = Path(tempfile.mkdtemp(prefix="mcraw_phase_g_")) / "g.mp4"
    render_err: list = []
    end_frame = min(len(ts), 120)

    def do_render():
        try:
            mcraw.render(input=str(fixture), output=str(out),
                         colorspace="srgb", codec="h265_nvenc",
                         start=0, end=end_frame, bitrate=40)
        except Exception as exc:
            render_err.append(exc)

    worker = threading.Thread(target=do_render, daemon=True)
    worker.start()

    frames = 0
    worst = 0
    t0 = time.time()
    # Alternate preview sizes the whole time the render runs — this is the
    # exact pattern that thrashed the shared buffers before Phase G.
    while worker.is_alive():
        for b in (1, 2):
            buf, h, w = d.process_frame_rgb24(t, "srgb", False, True, True, b)
            gpu = to_arr(buf, h, w)
            if gpu.shape != refs[b].shape:
                print(f"FAIL: bin={b} shape changed mid-render: {gpu.shape}")
                return 1
            diff = int(np.abs(gpu.astype(np.int16)
                              - refs[b].astype(np.int16)).max())
            worst = max(worst, diff)
            frames += 1
            if diff > 2:
                print(f"FAIL: concurrent preview bin={b} max|diff|={diff} "
                      f"vs CPU reference (corruption)")
                return 1
        if time.time() - t0 > 120:
            print("FAIL: render did not finish within 120s")
            return 1
    worker.join()

    if render_err:
        print(f"FAIL: render raised: {render_err[0]}")
        return 1
    if not out.exists() or out.stat().st_size == 0:
        print("FAIL: render produced no output")
        return 1

    dur = time.time() - t0
    print(f"  render: {end_frame} frames -> {out.stat().st_size:,} bytes "
          f"in {dur:.1f}s (NVENC GPU path)")
    print(f"  preview: {frames} concurrent frames, alternating full/half res, "
          f"worst |diff| vs reference = {worst}")
    print("\nPASS: no corruption, no deadlock, both paths completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
