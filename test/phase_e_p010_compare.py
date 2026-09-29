"""Validate the GPU P010 (10-bit) kernel against the already-verified GPU
NV12 (8-bit) kernel. Both use the same BT.709 limited-range math, so the
*normalised* decoded luma must agree — isolating the only thing new in the
P010 kernel: the 10-bit scaling (x4) and the high-bit storage (<<6).

(We deliberately don't compare against the CPU path here: libswscale uses
BT.601 luma coefficients by default while our kernels use BT.709, so that
comparison carries a systematic offset unrelated to the 10-bit math.)

Decodes frame 0 of each .mov, range-expanded to full-range gray by ffmpeg,
normalised to [0,1] per bit depth, then compares.
"""
from __future__ import annotations
import subprocess
import sys
import numpy as np

FF = "ffmpeg"
W, H = 4032, 1696


def luma(path: str, bits: int) -> np.ndarray:
    pix = "gray16le" if bits == 10 else "gray"
    out = subprocess.run(
        [FF, "-v", "error", "-i", path, "-vframes", "1",
         "-f", "rawvideo", "-pix_fmt", pix, "-"],
        capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode(errors="replace"))
    if bits == 10:
        a = np.frombuffer(out.stdout, dtype="<u2").astype(np.float64) / 65535.0
    else:
        a = np.frombuffer(out.stdout, dtype=np.uint8).astype(np.float64) / 255.0
    return a[: W * H]


p010 = luma("p010_gpu.mov", 10)   # GPU 10-bit
nv12 = luma("nv12_gpu.mov", 8)    # GPU 8-bit
diff = (p010 - nv12) * 1023.0     # express error in 10-bit code values
print(f"GPU P010 (norm): mean={p010.mean():.4f}  min={p010.min():.4f}  max={p010.max():.4f}")
print(f"GPU NV12 (norm): mean={nv12.mean():.4f}  min={nv12.min():.4f}  max={nv12.max():.4f}")
print(f"diff vs 8-bit: mean={diff.mean():+.2f}  abs-mean={np.abs(diff).mean():.2f}  "
      f"p99.9={np.percentile(np.abs(diff),99.9):.1f}  max={np.abs(diff).max():.0f} (10-bit cv)")
# Same BT.709 math at two bit depths through HEVC: expect agreement within
# codec + 8-bit-quantization noise (a few cv mean). A wrong P010 scale/shift
# would diverge by 4x / 64x.
sys.exit(0 if abs(diff.mean()) < 6.0 else 1)
