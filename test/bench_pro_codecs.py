#!/usr/bin/env python3
"""
Pro-codec render benchmark (Phase I). Run before/after each speedup lands
and append the numbers to the RESULTS block below.

RESULTS (dev box: RTX 4090, 12-core, 4032x1696, 48 frames, acescg):
  2026-07-05 baseline (pre Phase I):
    GPU chain 60.1 | CPU pipeline 12.0 | prores4444 1.2 | prores422hq 1.7
    | dnxhr_hqx 2.3 | cineform 5.9  (fps)
  2026-07-06 after A1 (encoder slice threading):
    prores4444 3.1 | prores422hq 4.2 | dnxhr_hqx 9.5 | cineform 12.9  (fps)
  2026-07-06 after A2, MCRAW_GPU_YUV=1 (GPU pack feeds the encoders):
    prores4444 3.4 | prores422hq 4.8 | dnxhr_hqx 14.4 | cineform 27.1  (fps)
    (env unset, CPU path: 2.9 / 4.0 / 9.2 / 12.0 — sws now pinned BT.709)
    prores_ks remains encoder-bound; frame-parallel encoding (Approach C
    in the spec) is the documented next step if ProRes needs more.

Usage:  python test/bench_pro_codecs.py [clip.mcraw] [--frames N]
Respects MCRAW_GPU_YUV; run it in both states when benchmarking A2.
"""
from __future__ import annotations

import argparse
import os
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

import mcraw  # noqa: E402

CODECS = ("prores4444", "prores422hq", "dnxhr_hqx", "cineform")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?",
                    default=str(PROJECT / "VIDEO_20250114_130136.0.mcraw"))
    ap.add_argument("--frames", type=int, default=48)
    args = ap.parse_args()
    clip = Path(args.clip)
    if not clip.is_file():
        print(f"ERROR: clip not found: {clip}", file=sys.stderr)
        return 2

    n = args.frames
    tmp = Path(tempfile.mkdtemp(prefix="bench_pro_"))
    d = mcraw.Decoder(str(clip))
    ts = d.frames[:n]
    gpu_env = os.environ.get("MCRAW_GPU_YUV", "")
    print(f"clip={clip.name}  frames={n}  MCRAW_GPU_YUV={gpu_env!r}\n")

    if mcraw.cuda_built() and mcraw.cuda_available():
        t0 = time.perf_counter()
        for t in ts:
            d.process_frame_gl(t, "srgb", False, True, 1, False)
        print(f"GPU chain (no readback):      {n/(time.perf_counter()-t0):6.1f} fps")

    t0 = time.perf_counter()
    for t in ts:
        d.process_frame(t, "acescg", False, True)
    print(f"CPU pipeline (process_frame): {n/(time.perf_counter()-t0):6.1f} fps")

    for codec in CODECS:
        out = tmp / f"b_{codec}.mov"
        t0 = time.perf_counter()
        mcraw.render(input=str(clip), output=str(out), colorspace="acescg",
                     codec=codec, start=0, end=n)
        dt = time.perf_counter() - t0
        print(f"render {codec:12s}          {n/dt:6.1f} fps  ({dt:.1f}s, "
              f"{out.stat().st_size//1_000_000} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
