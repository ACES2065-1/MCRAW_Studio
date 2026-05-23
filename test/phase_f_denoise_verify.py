"""Phase F check: GPU output-space denoise vs the CPU DenoiseRgb reference.

GPU path: cuda_process_frame_phase_c(cs, denoise_chroma=, denoise_luma=)
CPU ref : denoise_rgb(process_frame(cs), chroma, luma)

Both denoise the same output-space RGB. The GPU computes the bilateral range
weight with expf directly while the CPU uses a 1024-bin LUT, so they won't be
bit-identical, but a correct port lands far below a code value.
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
cs = "srgb"  # display-encoded; denoise runs in this output space

worst = 0.0
for chroma, luma in [(0, 0), (50, 0), (0, 50), (60, 40), (100, 100)]:
    ref = np.asarray(dec.process_frame(ts, cs, False)).copy()
    if chroma or luma:
        ref = np.asarray(mcraw.denoise_rgb(ref, chroma, luma))
    ref = ref.reshape(-1, 3)
    gpu = np.asarray(mcraw.cuda_process_frame_phase_c(
        dec, ts, cs, False, chroma, luma)).reshape(-1, 3)
    err = np.abs(ref - gpu)
    mx, p999 = float(err.max()), float(np.percentile(err, 99.9))
    worst = max(worst, mx)
    print(f"  chroma={chroma:3d} luma={luma:3d}  max={mx:.6f}  p99.9={p999:.6f}  "
          f"(~{mx*1023:.2f} cv)")

print(f"\nWorst max abs diff: {worst:.6f}  (~{worst*1023:.2f} cv @10-bit)")
sys.exit(0 if worst < 2e-3 else 1)  # < ~2 cv
