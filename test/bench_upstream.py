#!/usr/bin/env python3
"""
Microbenchmark for the upstream colour pipeline (decode + normalise + LSM +
debayer + matrix + OCIO + sRGB encode). Skips video encoding entirely so we
measure only the per-pixel work that AVX2 affects.

Usage:
    python test/bench_upstream.py <path/to/sample.mcraw> [--frames N] [--repeat R]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BUILD = PROJECT / "build"
VCPKG_BIN = Path(r"C:\dev\vcpkg\installed\x64-windows\bin")

if VCPKG_BIN.is_dir() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(VCPKG_BIN))
if BUILD.is_dir() and str(BUILD) not in sys.path:
    sys.path.insert(0, str(BUILD))

import mcraw  # noqa: E402


def run_pass(fixture: str, frames: int, colorspace: str,
             highlight_recovery: bool) -> float:
    """Render `frames` frames through the upstream pipeline (no encode).
    Returns total wall-clock seconds."""
    dec = mcraw.Decoder(fixture)
    timestamps = list(dec.frames)[:frames]
    t0 = time.perf_counter()
    for ts in timestamps:
        # process_frame_rgb24 runs: decode → normalise → debayer → matrix
        # → OCIO → sRGB encode (float→u8). Same as the GUI thumbnail path.
        dec.process_frame_rgb24(ts, colorspace, highlight_recovery)
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture")
    ap.add_argument("--frames", type=int, default=24,
                    help="How many frames to process per run (default: 24)")
    ap.add_argument("--repeat", type=int, default=5,
                    help="How many times to repeat each scenario (default: 5)")
    args = ap.parse_args()

    if not Path(args.fixture).is_file():
        print(f"ERROR: fixture not found: {args.fixture}", file=sys.stderr)
        return 2

    scenarios = [
        ("srgb,            recovery=off", "srgb", False),
        ("srgb,            recovery=on ", "srgb", True),
        ("acescg,          recovery=off", "acescg", False),
        ("rec709-display,  recovery=off", "rec709-display", False),
    ]

    print(f"Fixture: {args.fixture}")
    print(f"Frames per run: {args.frames}   Runs per scenario: {args.repeat}")
    print(f"mcraw module:   {mcraw.__file__}")
    print()
    print(f"{'scenario':40s}  median (s)   per-frame (ms)   fps")
    print("-" * 80)

    for name, cs, hr in scenarios:
        # Warm up once (caches, lazy OCIO context).
        run_pass(args.fixture, min(4, args.frames), cs, hr)
        times = [run_pass(args.fixture, args.frames, cs, hr)
                 for _ in range(args.repeat)]
        med = statistics.median(times)
        per_frame_ms = (med / args.frames) * 1000.0
        fps = args.frames / med
        print(f"{name:40s}  {med:8.3f}     {per_frame_ms:8.1f}      {fps:6.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
