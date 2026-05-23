"""Phase D pre-flight (no build needed).

Bakes the OCIO ACEScg->target transform into an N^3 float 3D LUT using a
log2 input shaper, then checks trilinear+shaper sampling against the
direct OCIO CPU processor over a sweep of ACEScg values. The point is to
pick the LUT size + shaper range BEFORE writing any CUDA.

Pass/fail is judged on the *output-encoded* signal (log/PQ targets live in
~[0,1]), so an abs error of ~1e-3 is roughly one 10-bit code value.
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

# ---- asinh shaper (maps ACEScg scene-linear -> [0,1] LUT domain) ---------
# Unlike a log2 allocation, asinh is defined over all reals: ~linear near 0
# (so sub-black / small negatives from the debayer survive) and log-like in
# the highlights (so HDR values compress). The device side only needs the
# forward s(L)->t (asinh + affine); the inverse (sinh) is host-only at bake.
#
#   s(L) = asinh(L / K)
#   t    = (s(L) - s_lo) / (s_hi - s_lo),  clamped [0,1]
#   L    = K * sinh(s_lo + t*(s_hi - s_lo))           (inverse)
#
# K sets the width of the linear toe; L_LO/L_HI set the represented range.
K = 0.01
L_LO = -0.35
L_HI = 16.0
_S_LO = float(np.arcsinh(L_LO / K))
_S_HI = float(np.arcsinh(L_HI / K))


def lin_to_t(L: np.ndarray) -> np.ndarray:
    return np.clip((np.arcsinh(L / K) - _S_LO) / (_S_HI - _S_LO), 0.0, 1.0)


def t_to_lin(t: np.ndarray) -> np.ndarray:
    return K * np.sinh(_S_LO + t * (_S_HI - _S_LO))


def set_shaper(k: float, lo: float, hi: float) -> None:
    global K, L_LO, L_HI, _S_LO, _S_HI
    K, L_LO, L_HI = k, lo, hi
    _S_LO = float(np.arcsinh(lo / k))
    _S_HI = float(np.arcsinh(hi / k))


def bake_lut(dst: str, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    grid = np.stack(np.meshgrid(t, t, t, indexing="ij"), axis=-1)  # (n,n,n,3)
    lin = t_to_lin(grid).astype(np.float32)
    flat = np.ascontiguousarray(lin.reshape(-1, 1, 3))
    tr = mcraw.OcioTransform("ACEScg", dst)
    out = tr.apply(flat)  # in-place ACEScg -> dst
    return np.asarray(out).reshape(n, n, n, 3).astype(np.float32)


def trilinear(lut: np.ndarray, coords: np.ndarray) -> np.ndarray:
    n = lut.shape[0]
    p = coords * (n - 1)
    i0 = np.clip(np.floor(p).astype(np.int64), 0, n - 2)
    f = p - i0
    i, j, k = i0[:, 0], i0[:, 1], i0[:, 2]
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]

    def g(di, dj, dk):
        return lut[i + di, j + dj, k + dk]

    c00 = g(0, 0, 0) * (1 - fx) + g(1, 0, 0) * fx
    c01 = g(0, 0, 1) * (1 - fx) + g(1, 0, 1) * fx
    c10 = g(0, 1, 0) * (1 - fx) + g(1, 1, 0) * fx
    c11 = g(0, 1, 1) * (1 - fx) + g(1, 1, 1) * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def evaluate_on(dst: str, n: int, test: np.ndarray) -> tuple[float, float, float]:
    """Returns (max, mean, 99.9th-percentile) abs error on the given
    ACEScg test colors, shape (M,1,3) or (M,3)."""
    test = np.ascontiguousarray(test.reshape(-1, 1, 3).astype(np.float32))
    lut = bake_lut(dst, n)
    ref = np.asarray(mcraw.OcioTransform("ACEScg", dst).apply(test.copy())).reshape(-1, 3)
    approx = trilinear(lut, lin_to_t(test.reshape(-1, 3)))
    err = np.abs(ref - approx)
    return float(err.max()), float(err.mean()), float(np.percentile(err, 99.9))


def real_frame_acescg(fixture: str, ts_index: int = 0) -> np.ndarray | None:
    try:
        dec = mcraw.Decoder(fixture)
    except Exception as exc:
        print(f"  (no fixture: {exc})")
        return None
    frames = dec.frames
    if not frames:
        return None
    arr = np.asarray(dec.process_frame(frames[min(ts_index, len(frames) - 1)],
                                       "acescg", False))
    return arr.reshape(-1, 3)


if __name__ == "__main__":
    targets = {
        "ACEScct": "ACEScct",
        "S-Log3 S-Gamut3.Cine": "S-Log3 S-Gamut3.Cine",
    }

    fixture = sys.argv[1] if len(sys.argv) > 1 else "VIDEO_20250114_130136.0.mcraw"
    real = real_frame_acescg(fixture)
    if real is None:
        print("No fixture frame available; aborting.")
        sys.exit(0)
    print(f"Real decoded ACEScg pixels: {real.shape[0]:,} px;  range "
          f"[{real.min():.4f}, {real.max():.4f}]\n")

    # Lock toe-width K=0.01, sweep the highlight ceiling L_hi to confirm a
    # safe headroom doesn't cost midtone accuracy. Add a few synthetic bright
    # speculars to the real pixels so the ceiling actually gets exercised.
    speculars = np.array([[s, s, s] for s in (4, 8, 16, 32, 64)], dtype=np.float32)
    real_hi = np.vstack([real, np.repeat(speculars, 50, axis=0)])
    for lhi in (16.0, 32.0, 64.0):
        set_shaper(0.01, -0.35, lhi)
        print(f"=== asinh K=0.01  L_lo=-0.35  L_hi={lhi} ===")
        for n in (33, 65):
            for label, dst in targets.items():
                mx, mn, p999 = evaluate_on(dst, n, real_hi)
                print(f"  {n}^3  {label:24s}  max={mx:.5f}  p99.9={p999:.5f}  "
                      f"mean={mn:.6f}  (~{mx*1023:.1f} cv max)")
