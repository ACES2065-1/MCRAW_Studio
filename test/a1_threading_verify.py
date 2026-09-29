#!/usr/bin/env python3
"""
A1 verification: threaded pro-codec encodes are deterministic.

Renders the same 12-frame range with encoder_threads=1 and =0 (auto) for
prores4444, dnxhr_hqx and cineform, decodes both to raw video, and compares:
  - primary bar: decoded bytes identical (slice layout is thread-invariant);
  - downgrade bar (spec): if a codec is NOT thread-count-deterministic,
    mean abs decoded diff must be tiny, with a printed warning.
Also checks the auto run is faster than the 1-thread run (threading engaged).
The speedup check is a hard gate for ProRes/DNxHR; warn-only for CineForm
(FFmpeg's cfhd encoder may not support threading at all — it was already
the fastest of the four and is not a spec target).

Usage:  python test/a1_threading_verify.py [clip.mcraw]
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

import numpy as np  # noqa: E402

import mcraw  # noqa: E402

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
N = 12
FAILURES: list[str] = []


def decode_raw(path: Path) -> bytes:
    out = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-map", "0:v:0",
         "-f", "rawvideo", "-"],
        capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode(errors="replace"))
    return out.stdout


def check(name: str, ok: bool, detail: str, warn_only: bool = False) -> None:
    marker = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{marker}] {name:28s} {detail}")
    if not ok and not warn_only:
        FAILURES.append(name)


def main() -> int:
    clip = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        PROJECT / "VIDEO_20250114_130136.0.mcraw"
    if not clip.is_file():
        print(f"ERROR: clip not found: {clip}", file=sys.stderr)
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="a1_"))

    for codec in ("prores4444", "dnxhr_hqx", "cineform"):
        times = {}
        raws = {}
        for label, threads in (("t1", 1), ("auto", 0)):
            out = tmp / f"{codec}_{label}.mov"
            t0 = time.perf_counter()
            mcraw.render(input=str(clip), output=str(out),
                         colorspace="acescg", codec=codec,
                         start=0, end=N, encoder_threads=threads)
            times[label] = time.perf_counter() - t0
            raws[label] = decode_raw(out)

        h1 = hashlib.sha256(raws["t1"]).hexdigest()
        ha = hashlib.sha256(raws["auto"]).hexdigest()
        if h1 == ha:
            check(f"{codec} deterministic", True, f"decoded hash {ha[:12]}…")
        else:
            a = np.frombuffer(raws["t1"], dtype=np.uint8).astype(np.int16)
            b = np.frombuffer(raws["auto"], dtype=np.uint8).astype(np.int16)
            m = min(len(a), len(b))
            mad = float(np.abs(a[:m] - b[:m]).mean())
            check(f"{codec} numeric bar", mad < 0.15,
                  f"NOT bit-identical (warn); mean abs byte diff {mad:.4f}")
        speedup = times["t1"] / max(times["auto"], 1e-9)
        check(f"{codec} threading engaged", speedup > 1.5,
              f"1-thread {N/times['t1']:.1f} fps -> auto "
              f"{N/times['auto']:.1f} fps ({speedup:.1f}x)",
              warn_only=(codec == "cineform"))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
