"""Phase E.3 check: GPU highlight recovery vs the CPU reference.

Compares cuda_process_frame_phase_c(..., highlight_recovery=True) against
process_frame(..., highlight_recovery=True) for:
  * a display-encoded target (srgb) -> exercises BOTH the pre-matrix
    NeutraliseClippedHighlights and the post-transform rolloff,
  * a scene-referred target (acescg) -> NeutraliseClippedHighlights only.

Both should be bit-identical within float epsilon (same math, GPU vs CPU).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np

BUILD = Path(__file__).resolve().parent.parent / "build"
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

worst = 0.0
for cs in ("srgb", "acescg"):
    for hr in (False, True):
        ref = np.asarray(dec.process_frame(ts, cs, hr)).reshape(-1, 3)
        gpu = np.asarray(mcraw.cuda_process_frame_phase_c(dec, ts, cs, hr)).reshape(-1, 3)
        err = float(np.abs(ref - gpu).max())
        worst = max(worst, err)
        print(f"  {cs:7s} highlight_recovery={str(hr):5s}  max_abs_diff={err:.7f}")

print(f"\nWorst: {worst:.7f}")
sys.exit(0 if worst < 1e-4 else 1)
