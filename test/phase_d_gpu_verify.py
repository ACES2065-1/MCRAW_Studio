"""Phase D end-to-end GPU check.

Runs the real CUDA pipeline (bayer -> cam->ACEScg -> baked OCIO 3D LUT) via
cuda_process_frame_phase_d and compares it to process_frame() (which uses the
OCIO CPU processor) on the same frame. This exercises the whole device path
including the hardware-trilinear texture fetch and its weight quantization.

Pass bar: the GPU LUT should match the OCIO CPU reference to within a few
10-bit code values (the inherent CPU-vs-GPU-LUT difference, same as Resolve).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "build"
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
dec = mcraw.Decoder(fixture)
ts = dec.frames[0]

# OCIO targets (no baked transform) exposed by the module.
targets = ["acescct", "slog3-sgamut3cine"]

print(f"Fixture: {fixture}  ({dec.frame_count} frames)\n")
worst = 0.0
for cs in targets:
    ref = np.asarray(dec.process_frame(ts, cs, False)).reshape(-1, 3)
    gpu = np.asarray(mcraw.cuda_process_frame_phase_d(dec, ts, cs)).reshape(-1, 3)
    err = np.abs(ref - gpu)
    mx, mn, p999 = float(err.max()), float(err.mean()), float(np.percentile(err, 99.9))
    worst = max(worst, mx)
    print(f"  {cs:22s}  max={mx:.5f}  p99.9={p999:.5f}  mean={mn:.6f}  "
          f"(~{mx*1023:.1f} cv max,  ref range [{ref.min():.3f},{ref.max():.3f}])")

print(f"\nWorst max abs error: {worst:.5f}  (~{worst*1023:.1f} cv @10-bit)")
sys.exit(0 if worst < 0.01 else 1)  # < ~10 cv
