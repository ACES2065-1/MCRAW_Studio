"""Phase E.2 + sws-fix verification.

For a BT.709 target (acescg) and a BT.2020 target (rec2020-pq), render the
same clip on GPU (kernel matrix) and CPU (now-pinned sws matrix) and compare
decoded luma. With the sws matrix pinned to match the kernel, GPU and CPU
should now agree within codec noise (the pre-fix BT.601-vs-709 path showed a
~3.5 cv systematic offset). Also confirms the BT.2020 file carries the right
nclc tags.
"""
from __future__ import annotations
import subprocess
import sys
import numpy as np

FF = r"ffmpeg.exe"
FFP = r"ffprobe.exe"
W, H = 4032, 1696


def luma10(path: str) -> np.ndarray:
    out = subprocess.run(
        [FF, "-v", "error", "-i", path, "-vframes", "1",
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
        capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode(errors="replace"))
    return np.frombuffer(out.stdout, dtype="<u2").astype(np.float64)[: W * H] / 64.0


def tags(path: str) -> str:
    out = subprocess.run(
        [FFP, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=color_space,color_primaries,color_transfer", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    return out.stdout.strip()


def compare(label: str, gpu: str, cpu: str) -> float:
    g, c = luma10(gpu), luma10(cpu)
    d = g - c
    print(f"  {label}: GPU/CPU luma mean {g.mean():.1f}/{c.mean():.1f}  "
          f"diff mean={d.mean():+.2f} abs-mean={np.abs(d).mean():.2f} (10-bit cv)")
    print(f"      GPU tags (space,prim,trc): {tags(gpu)}")
    return abs(d.mean())


if __name__ == "__main__":
    worst = max(
        compare("BT.709 (acescg)", "e2_gpu_709.mov", "e2_cpu_709.mov"),
        compare("BT.2020 (rec2020-pq)", "e2_gpu_2020.mov", "e2_cpu_2020.mov"),
    )
    print(f"\nWorst mean offset: {worst:.2f} cv  "
          f"(pre-sws-fix BT.709 was ~3.5)")
    sys.exit(0 if worst < 1.5 else 1)
