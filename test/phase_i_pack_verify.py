#!/usr/bin/env python3
"""
Phase I (A2) verification, part 1: the GPU YUV pack kernels against a
numpy implementation of the exact spec formula (BT.709 limited 10-bit,
low-bit u16; 4:2:2 chroma = horizontal pair average).

Part 2 (added by the MovEncoder wiring task) renders GPU-packed vs
CPU-sws files and compares decoded planes to the Phase E.2 bar.

Usage:  python test/phase_i_pack_verify.py [clip.mcraw]
"""
from __future__ import annotations

import os
import sys
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


def numpy_pack(rgb: np.ndarray, subsample422: bool):
    """Reference implementation of the Global Constraints formula."""
    c = np.clip(rgb.astype(np.float64), 0.0, 1.0)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    yp = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - yp) / 1.8556
    cr = (r - yp) / 1.5748
    y10 = np.rint(64.0 + 876.0 * yp).astype(np.uint16)
    if subsample422:
        cb = (cb[:, 0::2] + cb[:, 1::2]) * 0.5
        cr = (cr[:, 0::2] + cr[:, 1::2]) * 0.5
    cb10 = np.rint(512.0 + 896.0 * cb).astype(np.uint16)
    cr10 = np.rint(512.0 + 896.0 * cr).astype(np.uint16)
    return y10, cb10, cr10


def part2(clip: Path) -> None:
    """End-to-end: GPU-packed render vs CPU-sws render, decoded plane
    means within the Phase E.2 bar (< 1.5 cv @10-bit), identical tags."""
    import shutil
    import subprocess
    import tempfile
    FFMPEG = shutil.which("ffmpeg") or r"ffmpeg.exe"
    FFPROBE = shutil.which("ffprobe") or r"ffprobe.exe"
    tmp = Path(tempfile.mkdtemp(prefix="phase_i_e2e_"))
    N = 8

    def render(codec: str, gpu: bool) -> Path:
        out = tmp / f"{codec}_{'gpu' if gpu else 'cpu'}.mov"
        env = dict(os.environ)
        env.pop("MCRAW_GPU_YUV", None)
        if gpu:
            env["MCRAW_GPU_YUV"] = "1"
        # Subprocess so the env gate is evaluated freshly per render.
        code = (
            "import os, sys;"
            f"sys.path.insert(0, r'{BUILD}');"
            f"os.add_dll_directory(r'{VCPKG_BIN}');"
            "import mcraw;"
            f"mcraw.render(input=r'{clip}', output=r'{out}', "
            f"colorspace='acescg', codec='{codec}', start=0, end={N})"
        )
        proc = subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"render {codec} gpu={gpu} failed:\n{proc.stderr}")
        if gpu and "Phase I GPU pack pipeline enabled" not in proc.stderr:
            raise RuntimeError(
                f"render {codec} gpu=True did not take the pack path:\n{proc.stderr}")
        return out

    def planes(path: Path, pixfmt: str, cw_div: int, w: int, h: int):
        outp = subprocess.run(
            [FFMPEG, "-v", "error", "-i", str(path), "-vframes", "1",
             "-f", "rawvideo", "-pix_fmt", pixfmt, "-"],
            capture_output=True, check=True)
        a = np.frombuffer(outp.stdout, dtype="<u2")
        nY = w * h
        nC = (w // cw_div) * h
        return a[:nY], a[nY:nY + nC], a[nY + nC:nY + 2 * nC]

    def tags(path: Path) -> str:
        outp = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=pix_fmt,color_space,color_primaries,color_transfer",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True)
        return outp.stdout.strip()

    d = mcraw.Decoder(str(clip))
    buf, h, w = d.process_frame_rgb24(d.frames[0], "srgb", False, True)
    del buf

    for codec, pixfmt, cw_div in (("prores4444", "yuv444p10le", 1),
                                  ("dnxhr_hqx", "yuv422p10le", 2)):
        gpu_f = render(codec, True)
        cpu_f = render(codec, False)
        check(f"{codec} tags equal", tags(gpu_f) == tags(cpu_f),
              tags(gpu_f))
        gp = planes(gpu_f, pixfmt, cw_div, w, h)
        cp = planes(cpu_f, pixfmt, cw_div, w, h)
        for name, gv, cv in zip(("Y", "Cb", "Cr"), gp, cp):
            off = abs(float(gv.astype(np.float64).mean())
                      - float(cv.astype(np.float64).mean()))
            check(f"{codec} {name} mean offset", off < 1.5,
                  f"{off:.3f} cv @10-bit (limit 1.5)")


def main() -> int:
    clip = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        PROJECT / "VIDEO_20250114_130136.0.mcraw"
    if not clip.is_file():
        print(f"ERROR: clip not found: {clip}", file=sys.stderr)
        return 2
    if not (mcraw.cuda_built() and mcraw.cuda_available()):
        print("SKIP: needs CUDA.")
        return 0

    d = mcraw.Decoder(str(clip))
    t = d.frames[min(4, len(d.frames) - 1)]

    for cs in ("acescg", "srgb"):
        # CPU float reference through the same chain the GPU runs.
        ref_rgb = np.asarray(d.process_frame(t, cs, False, True))
        for sub in (False, True):
            tag = f"{cs} {'422' if sub else '444'}"
            y, cb, cr = mcraw.cuda_pack_yuv_planar(d, t, cs, sub, False, True)
            ry, rcb, rcr = numpy_pack(ref_rgb, sub)
            for name, got, want in (("Y", y, ry), ("Cb", cb, rcb),
                                    ("Cr", cr, rcr)):
                got = np.asarray(got)
                if got.shape != want.shape:
                    check(f"{tag} {name}", False,
                          f"shape {got.shape} vs {want.shape}")
                    continue
                diff = np.abs(got.astype(np.int32) - want.astype(np.int32))
                worst = int(diff.max())
                frac = float((diff > 1).mean())
                check(f"{tag} {name}", worst <= 2 and frac < 1e-4,
                      f"max|diff|={worst} cv, frac(>1)={frac:.2e}")

    part2(clip)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
