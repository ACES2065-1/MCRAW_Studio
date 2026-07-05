#!/usr/bin/env python3
"""
Verification for the real-time player's decode backend (v0.7.0).

Three properties are checked, following the Phase C/D "GPU must match CPU"
pattern:

  1. GPU u8 parity (full res): process_frame_rgb24(prefer_gpu=True) must match
     the CPU path within +/-1 8-bit code value for (nearly) every pixel. The
     GPU chain is the same math verified in Phase C/E; the only new step is
     the RgbFloatToU8 clamp kernel, so any disagreement beyond float rounding
     is a bug in that kernel or the readback.

  2. GPU u8 parity (proxy): same comparison at preview_bin=2 and 4. Both
     paths run SubsampleBayerCells first, so this pins the GPU chain on the
     decimated mosaic AND confirms the proxy keeps a valid CFA phase (a
     phase error swaps chroma and blows the diff far past 1 code value).

  3. Proxy geometry + plausibility: the binned output dims must equal
     2*floor(W/2/bin) x 2*floor(H/2/bin), and the proxy image must be
     statistically close to the full-res render sampled at the matching
     grid positions (gross mean error means wrong cells were copied).

Usage:
    python test/player_preview_verify.py [path/to/clip.mcraw]

Exit 0 = pass. GPU checks are skipped (with a warning) on machines without
CUDA; geometry checks always run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BUILD = PROJECT / "build"
VCPKG_BIN = Path(r"C:\dev\vcpkg\installed\x64-windows\bin")

if VCPKG_BIN.is_dir() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(VCPKG_BIN))
if BUILD.is_dir() and str(BUILD) not in sys.path:
    sys.path.insert(0, str(BUILD))

import numpy as np  # noqa: E402

import mcraw  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    marker = "PASS" if ok else "FAIL"
    print(f"  [{marker}] {name:34s} {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def rgb24_to_array(buf: bytes, h: int, w: int) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)


def main() -> int:
    default = PROJECT / "VIDEO_20250114_130136.0.mcraw"
    fixture = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not fixture.is_file():
        print(f"ERROR: fixture not found: {fixture}", file=sys.stderr)
        return 2

    d = mcraw.Decoder(str(fixture))
    ts = d.frames
    if not ts:
        print("ERROR: clip has no frames", file=sys.stderr)
        return 2
    t = ts[min(4, len(ts) - 1)]   # skip the very first frame (AE settling)

    cuda = mcraw.cuda_built() and mcraw.cuda_available()
    print(f"Fixture: {fixture.name}   frames={len(ts)}   cuda={cuda}\n")

    # CPU references at every bin factor under test.
    cpu = {}
    for b in (1, 2, 4):
        buf, h, w = d.process_frame_rgb24(t, "srgb", False, True, False, b)
        cpu[b] = rgb24_to_array(buf, h, w)

    # ---- 3. proxy geometry + plausibility (CPU only, always runs) ---------
    fh, fw = cpu[1].shape[:2]
    for b in (2, 4):
        bh, bw = cpu[b].shape[:2]
        eh, ew = 2 * ((fh // 2) // b), 2 * ((fw // 2) // b)
        check(f"proxy dims bin={b}", (bh, bw) == (eh, ew),
              f"got {bw}x{bh}, expected {ew}x{eh}")
        # Sample the full-res render at each proxy cell's source position and
        # compare. Not exact (debayer neighbourhoods differ) but a CFA phase
        # bug or off-by-one cell copy pushes this far beyond the threshold.
        cy = np.arange(bh // 2) * (2 * b)
        cx = np.arange(bw // 2) * (2 * b)
        ref = cpu[1][np.ix_(cy, cx)].astype(np.int16)
        got = cpu[b][::2, ::2].astype(np.int16)
        mad = float(np.mean(np.abs(ref - got)))
        check(f"proxy content bin={b}", mad < 10.0,
              f"mean |diff| vs full-res grid = {mad:.2f} (limit 10)")

    # ---- 1 + 2. GPU u8 parity ---------------------------------------------
    if not cuda:
        print("\n  [SKIP] GPU parity checks — no CUDA device visible.")
    else:
        for b in (1, 2, 4):
            buf, h, w = d.process_frame_rgb24(t, "srgb", False, True, True, b)
            gpu = rgb24_to_array(buf, h, w)
            if gpu.shape != cpu[b].shape:
                check(f"gpu parity bin={b}", False,
                      f"shape {gpu.shape} != cpu {cpu[b].shape}")
                continue
            diff = np.abs(gpu.astype(np.int16) - cpu[b].astype(np.int16))
            worst = int(diff.max())
            frac_gt1 = float(np.mean(diff > 1))
            # +/-1 covers float rounding at the u8 quantise; a tiny tail of
            # 2 is tolerated (fp contraction differences at curve knees).
            ok = worst <= 2 and frac_gt1 < 1e-4
            check(f"gpu parity bin={b}", ok,
                  f"max|diff|={worst}, frac(>1)={frac_gt1:.2e}")

    # ---- audio bytes sanity (player feeds these to QAudioSink) ------------
    pcm = d.load_audio_bytes()
    rate, ch = int(d.audio_sample_rate), int(d.audio_channels)
    if rate > 0 and ch > 0:
        arr = d.load_audio()
        check("audio bytes", len(pcm) == arr.size * 2,
              f"{len(pcm)} bytes vs numpy {arr.size} samples")
    else:
        print("  [SKIP] audio bytes — clip has no audio.")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
