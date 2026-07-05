#!/usr/bin/env python3
"""
Scopes verification: GPU and CPU histograms/clip stats must describe the
same image.

  1. Self-consistency (both backends): the histogram must sum to
     3 * pixel count, and per-channel bin sums to the pixel count.
  2. Histogram-vs-image: the R/G/B histograms must exactly match numpy
     histograms of the returned u8 image (the bin formula IS the u8
     quantise, so this is an equality check per backend).
  3. GPU-vs-CPU: same frame through both backends — clip fractions must
     agree closely and the histograms may differ only by the +/-1-bin
     migration of borderline pixels (bounded L1 distance).

Usage:  python test/scopes_verify.py [clip.mcraw]
"""
from __future__ import annotations

import os
import sys
from array import array
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

import numpy as np  # noqa: E402

import mcraw  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:30s} {detail}")
    if not ok:
        FAILURES.append(name)


def run_backend(d, t, gpu: bool):
    buf, h, w, hist_b, lo, hi = d.process_frame_rgb24_scopes(
        t, "srgb", False, True, gpu, 1)
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    hist = np.asarray(array("I", hist_b), dtype=np.uint64).reshape(3, 256)
    return img, hist, lo, hi


def main() -> int:
    default = PROJECT / "VIDEO_20250114_130136.0.mcraw"
    fixture = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not fixture.is_file():
        print(f"ERROR: fixture not found: {fixture}", file=sys.stderr)
        return 2

    d = mcraw.Decoder(str(fixture))
    t = d.frames[min(4, len(d.frames) - 1)]
    cuda = mcraw.cuda_built() and mcraw.cuda_available()

    backends = [("cpu", False)] + ([("gpu", True)] if cuda else [])
    results = {}
    for name, gpu in backends:
        img, hist, lo, hi = run_backend(d, t, gpu)
        results[name] = (img, hist, lo, hi)
        n = img.shape[0] * img.shape[1]
        check(f"{name} bin totals", all(int(hist[c].sum()) == n for c in range(3)),
              f"each channel sums to {n}")
        exact = all(
            np.array_equal(hist[c],
                           np.bincount(img[:, :, c].ravel(), minlength=256))
            for c in range(3))
        check(f"{name} hist == image", exact, "3x256 bins equal numpy bincount")
        check(f"{name} clip range", 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0,
              f"lo={lo:.4f} hi={hi:.4f}")

    if cuda:
        (ci, ch, clo, chi), (gi, gh, glo, ghi) = results["cpu"], results["gpu"]
        l1 = int(np.abs(ch.astype(np.int64) - gh.astype(np.int64)).sum())
        n = ci.shape[0] * ci.shape[1]
        # Borderline pixels may land one bin apart between backends; each
        # migrant costs 2 in L1. The u8 parity check bounds migrants at
        # ~frac(>0) of pixels, so allow a small fraction.
        check("gpu-vs-cpu hist L1", l1 <= max(64, int(0.001 * n)),
              f"L1={l1} over {n} px")
        check("gpu-vs-cpu clip", abs(clo - glo) < 1e-3 and abs(chi - ghi) < 1e-3,
              f"lo {clo:.5f}/{glo:.5f}  hi {chi:.5f}/{ghi:.5f}")
    else:
        print("  [SKIP] gpu-vs-cpu — no CUDA device.")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
