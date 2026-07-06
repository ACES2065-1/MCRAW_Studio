# Phase I: Pro-Codec Speedup (Encoder Threading + GPU Pixel Packing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take ProRes/DNxHR/CineForm renders from encoder-bound 1.2–5.9 fps to ≥10 fps at 4K by (A1) enabling slice threading on the CPU encoders and (A2) feeding them GPU-packed `yuv42{2,4}p10le` planes instead of CPU sws_scale.

**Architecture:** A1 sets `thread_count`/`thread_type` on the `AVCodecContext` for `prores_ks`/`dnxhd`/`cineform` only, plumbed as `encoder_threads` (0=auto) through `EncodeSettings` → `mcraw.render()` → CLI/GUI. A2 adds two pack kernels to the existing GPU bayer chain (render context), a `ProcessBayerToYuvPlanarHost` entry point with pinned readback, and extends `MovEncoder::EnableGpuBayerPipeline` eligibility to the seven pro MOV codecs behind the existing `MCRAW_GPU_YUV=1` opt-in; DoRender's producer thread packs on the GPU while the consumer thread encodes.

**Tech Stack:** C++17, CUDA 12.x (nvcc), FFmpeg (libavcodec/libswscale via vcpkg), pybind11, Python 3.12 + numpy for verification.

**Spec:** `docs/superpowers/specs/2026-07-05-pro-codec-gpu-speedup-design.md`

## Global Constraints

- Build command (must be used verbatim, cmd via PowerShell): `cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && ninja -C build'`
- All verify/bench scripts run with the dev-box python: `python test\<script>.py` from the repo root; they self-locate `build/` and the vcpkg DLLs and accept `MCRAW_BUILD_DIR`/`MCRAW_DLL_DIR` env overrides (copy the bootstrap block from `test/scopes_verify.py`).
- Test clip: `VIDEO_20250114_130136.0.mcraw` (repo root, 250 frames, 4032x1696). CI fixture: `test/fixtures/smoke2.mcraw` (2 frames).
- Baselines to beat (measured 2026-07-05): prores4444 1.2 fps, prores422hq 1.7, dnxhr_hqx 2.3, cineform 5.9. CPU pipeline 12.0 fps, GPU chain 60.1 fps. Success: prores4444 ≥ 10 fps.
- Commit messages: NO `Co-Authored-By` trailer (repo rule). Use single-quoted here-strings for multi-line messages in PowerShell.
- `MCRAW_GPU_YUV=1` is the opt-in gate for ALL GPU render paths including the new pack path. CPU path must remain byte-for-byte what it is today when the env var is unset (aside from A1 threading).
- GPU-vs-CPU numeric bar (from Phase E.2): decoded mean luma AND chroma offsets < 1.5 cv @10-bit. Kernel-level bar: ±1 code value vs the numpy reference formula.
- BT.709 limited-range 10-bit quantisation (the single source of truth for kernels AND numpy references): `Y' = 0.2126 R + 0.7152 G + 0.0722 B`, `Cb = (B − Y')/1.8556`, `Cr = (R − Y')/1.5748`, on RGB clamped to [0,1]; `Y10 = round(64 + 876·Y')`, `C10 = round(512 + 896·C)`, stored as u16 with the 10-bit value in the LOW bits (that is the `yuv42xp10le` convention — NOT the P010 high-bits convention).
- The existing smoke suite must stay 12/12 in BOTH `MCRAW_GPU_YUV` states after every task that touches the render path.

---

### Task 1: Committed benchmark script with recorded baselines

**Files:**
- Create: `test/bench_pro_codecs.py`

**Interfaces:**
- Produces: `python test/bench_pro_codecs.py [clip]` printing an fps table; later tasks re-run it and append result lines to its header comment.

- [ ] **Step 1: Write the benchmark script**

```python
#!/usr/bin/env python3
"""
Pro-codec render benchmark (Phase I). Run before/after each speedup lands
and append the numbers to the RESULTS block below.

RESULTS (dev box: RTX 4090, 12-core, 4032x1696, 48 frames, acescg):
  2026-07-05 baseline (pre Phase I):
    GPU chain 60.1 | CPU pipeline 12.0 | prores4444 1.2 | prores422hq 1.7
    | dnxhr_hqx 2.3 | cineform 5.9  (fps)

Usage:  python test/bench_pro_codecs.py [clip.mcraw] [--frames N]
Respects MCRAW_GPU_YUV; run it in both states when benchmarking A2.
"""
from __future__ import annotations

import argparse
import os
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

import mcraw  # noqa: E402

CODECS = ("prores4444", "prores422hq", "dnxhr_hqx", "cineform")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?",
                    default=str(PROJECT / "VIDEO_20250114_130136.0.mcraw"))
    ap.add_argument("--frames", type=int, default=48)
    args = ap.parse_args()
    clip = Path(args.clip)
    if not clip.is_file():
        print(f"ERROR: clip not found: {clip}", file=sys.stderr)
        return 2

    n = args.frames
    tmp = Path(tempfile.mkdtemp(prefix="bench_pro_"))
    d = mcraw.Decoder(str(clip))
    ts = d.frames[:n]
    gpu_env = os.environ.get("MCRAW_GPU_YUV", "")
    print(f"clip={clip.name}  frames={n}  MCRAW_GPU_YUV={gpu_env!r}\n")

    if mcraw.cuda_built() and mcraw.cuda_available():
        t0 = time.perf_counter()
        for t in ts:
            d.process_frame_gl(t, "srgb", False, True, 1, False)
        print(f"GPU chain (no readback):      {n/(time.perf_counter()-t0):6.1f} fps")

    t0 = time.perf_counter()
    for t in ts:
        d.process_frame(t, "acescg", False, True)
    print(f"CPU pipeline (process_frame): {n/(time.perf_counter()-t0):6.1f} fps")

    for codec in CODECS:
        out = tmp / f"b_{codec}.mov"
        t0 = time.perf_counter()
        mcraw.render(input=str(clip), output=str(out), colorspace="acescg",
                     codec=codec, start=0, end=n)
        dt = time.perf_counter() - t0
        print(f"render {codec:12s}          {n/dt:6.1f} fps  ({dt:.1f}s, "
              f"{out.stat().st_size//1_000_000} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it and confirm it reproduces the baseline within noise**

Run: `python test\bench_pro_codecs.py`
Expected: prores4444 in 1.0–1.5 fps range, others near the RESULTS line. (If numbers differ wildly, STOP — the machine state changed and every later comparison is invalid.)

- [ ] **Step 3: Commit**

```powershell
git add test/bench_pro_codecs.py
git commit -m "Phase I: pro-codec benchmark script with recorded baselines"
```

---

### Task 2: A1 — encoder threading (`encoderThreads` end to end)

**Files:**
- Modify: `lib/include/motioncam/MovEncoder.hpp` (EncodeSettings, after `tenBit`)
- Modify: `lib/MovEncoder.cpp` (constructor, after the CineForm `quality` block ~line 288; add `#include <thread>` to the include list at the top)
- Modify: `python/mcraw_py.cpp` (`DoRender` signature + `es` assignment + `m.def("render", ...)` args)
- Create: `test/a1_threading_verify.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `EncodeSettings::encoderThreads` (int, 0 = auto = all logical cores) and `mcraw.render(..., encoder_threads=0)`. Task 3 (CLI/GUI) and Task 5 (bench comparisons) rely on the kwarg name `encoder_threads` exactly.

- [ ] **Step 1: Write the failing verify script**

```python
#!/usr/bin/env python3
"""
A1 verification: threaded pro-codec encodes are deterministic.

Renders the same 12-frame range with encoder_threads=1 and =0 (auto) for
prores4444 and dnxhr_hqx, decodes both to raw video, and compares:
  - primary bar: decoded bytes identical (slice layout is thread-invariant);
  - downgrade bar (spec): if a codec is NOT thread-count-deterministic,
    mean abs decoded diff < 0.5 cv @10-bit, with a printed warning.
Also asserts the auto run is not slower than the 1-thread run (sanity that
threading engaged at all).

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

FFMPEG = shutil.which("ffmpeg") or r"ffmpeg.exe"
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


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:28s} {detail}")
    if not ok:
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
            # 0.5 cv @10-bit on 16-bit-container bytes ≈ 0.5*64/256 per byte;
            # use the spec's intent: mean abs diff tiny.
            check(f"{codec} numeric bar", mad < 0.15,
                  f"NOT bit-identical (warn); mean abs byte diff {mad:.4f}")
        speedup = times["t1"] / max(times["auto"], 1e-9)
        check(f"{codec} threading engaged", speedup > 1.5,
              f"1-thread {N/times['t1']:.1f} fps -> auto "
              f"{N/times['auto']:.1f} fps ({speedup:.1f}x)")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it to verify it fails for the right reason**

Run: `python test\a1_threading_verify.py`
Expected: `TypeError: render() got an unexpected keyword argument 'encoder_threads'` — the plumbing doesn't exist yet.

- [ ] **Step 3: Add the settings field**

In `lib/include/motioncam/MovEncoder.hpp`, inside `struct EncodeSettings`, directly after the `bool tenBit = false;` member:

```cpp
    // Phase I (A1): worker threads for the CPU intermediate encoders
    // (prores_ks / dnxhd / cineform). 0 = auto (all logical cores).
    // libx264/x265 manage their own pools and NVENC is hardware — both
    // ignore this field.
    int encoderThreads = 0;
```

- [ ] **Step 4: Configure threading in the MovEncoder constructor**

In `lib/MovEncoder.cpp`: add `#include <thread>` next to the existing standard includes. Then insert AFTER the CineForm `quality` block (`av_opt_set(p->videoCtx->priv_data, "quality", "film3", 0);` + closing brace) and BEFORE the "Codecs with explicit user-controlled bitrate" block:

```cpp
    // Phase I (A1): slice-thread the CPU intermediate encoders. libavcodec
    // defaults to ONE thread unless asked. prores_ks / dnxhd parallelise
    // across slices within a frame (bitstream identical to single-thread);
    // cfhd additionally accepts frame threading. libx264/x265 self-thread
    // and NVENC is hardware — neither is touched here.
    {
        const bool isProCpu =
            s.codec == Codec::ProRes422  || s.codec == Codec::ProRes422HQ ||
            s.codec == Codec::ProRes4444 || s.codec == Codec::ProRes4444XQ ||
            s.codec == Codec::DNxHR_HQX  || s.codec == Codec::DNxHR_444 ||
            s.codec == Codec::CineForm;
        if (isProCpu) {
            int threads = s.encoderThreads;
            if (threads <= 0) {
                threads = static_cast<int>(std::thread::hardware_concurrency());
                if (threads <= 0) threads = 1;
            }
            p->videoCtx->thread_count = threads;
            p->videoCtx->thread_type = (s.codec == Codec::CineForm)
                ? (FF_THREAD_SLICE | FF_THREAD_FRAME)
                : FF_THREAD_SLICE;
        }
    }
```

- [ ] **Step 5: Plumb through the Python binding**

In `python/mcraw_py.cpp`:

1. `DoRender` signature — add a parameter after `bool bake_vignette`:

```cpp
    bool bake_vignette,
    int encoder_threads)
```

2. Where `EncodeSettings es{}` is filled (after `es.tenBit = ten_bit;`):

```cpp
        es.encoderThreads = encoder_threads;
```

3. In `m.def("render", ...)` add after `py::arg("bake_vignette") = true,`:

```cpp
        py::arg("encoder_threads") = 0,
```

and extend the docstring's option list with one line: `encoder_threads: worker threads for ProRes/DNxHR/CineForm (0 = all cores).`

- [ ] **Step 6: Build**

Run: `cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && ninja -C build'`
Expected: clean build ending with `Linking CXX shared module mcraw.cp312-win_amd64.pyd`.

- [ ] **Step 7: Run the verify script — expect pass**

Run: `python test\a1_threading_verify.py`
Expected: all checks PASS; prores4444 speedup line shows ≥ 1.5x (anticipate ~5–8x). If `cineform` reports NOT bit-identical, the numeric-bar downgrade prints a warning — that is acceptable per spec, only a FAIL is a blocker.

- [ ] **Step 8: Re-run smoke + bench, append the A1 numbers**

Run: `python test\smoke.py test\fixtures\smoke2.mcraw` → Expected: `12/12 passed`.
Run: `$env:MCRAW_GPU_YUV = "1"; python test\smoke.py test\fixtures\smoke2.mcraw; Remove-Item Env:MCRAW_GPU_YUV` → Expected: `12/12 passed`.
Run: `python test\bench_pro_codecs.py` and append a `2026-07-06 after A1:` line to the RESULTS block in `test/bench_pro_codecs.py` with the printed fps values.

- [ ] **Step 9: Commit**

```powershell
git add lib/include/motioncam/MovEncoder.hpp lib/MovEncoder.cpp python/mcraw_py.cpp test/a1_threading_verify.py test/bench_pro_codecs.py
git commit -m @'
Phase I.A1 - slice-thread the CPU intermediate encoders

libavcodec defaults to one thread; prores_ks/dnxhd/cineform were
encoding 4K on a single core. EncodeSettings.encoderThreads (0 = all
logical cores) now drives thread_count/thread_type for those codecs
only; libx264/x265/NVENC untouched. Exposed as encoder_threads= on
mcraw.render(). Verified deterministic vs single-thread by decoded
hash (test/a1_threading_verify.py); bench numbers in
test/bench_pro_codecs.py header; smoke 12/12 both GPU env states.
'@
```

---

### Task 3: A1 — CLI `--threads` and GUI thread budget

**Files:**
- Modify: `mcraw_render.cpp` (args struct ~line 42, `PrintUsage` ~line 67, `ParseArgs` ~line 124, `es` fill ~line 265)
- Modify: `gui/motioncam_tools.py` (`RenderWorker._run_one` kwargs block; `MainWindow._render_all` settings dict)

**Interfaces:**
- Consumes: `EncodeSettings::encoderThreads` and `mcraw.render(encoder_threads=)` from Task 2.
- Produces: CLI flag `--threads <n>`; GUI settings key `"encoder_threads"`.

- [ ] **Step 1: CLI flag**

In `mcraw_render.cpp`:
1. Args struct (next to `int bitrateMbps = 80;`): `int encoderThreads = 0;`
2. `PrintUsage()` after the `--bitrate` line: `"  --threads <n>           ProRes/DNxHR/CineForm encoder threads (default: all cores)\n"`
3. `ParseArgs` next to the `--bitrate` branch: `else if (a == "--threads") out.encoderThreads = std::stoi(needValue());`
4. Where `es.bitrateMbps = args.bitrateMbps;` is set: `es.encoderThreads = args.encoderThreads;`

- [ ] **Step 2: GUI thread budget**

In `gui/motioncam_tools.py`:

1. In `MainWindow._render_all`, in the `if is_video:` block that already sets `settings["codec"]`, add:

```python
            # Phase I (A1): divide encoder threads across concurrent renders
            # so 3 workers x 12 threads don't oversubscribe the machine.
            settings["encoder_threads"] = max(
                1, (os.cpu_count() or 8) // int(self.concurrentSpin.value()))
```

2. In `RenderWorker._run_one`, in the `kwargs` assembly (next to the `ten_bit` pass-through):

```python
            if "encoder_threads" in self.settings:
                kwargs["encoder_threads"] = int(self.settings["encoder_threads"])
```

- [ ] **Step 3: Build + verify CLI and GUI paths**

Run: `cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && ninja -C build'` → clean.
Run: `.\build\mcraw_render.exe test\fixtures\smoke2.mcraw -o "$env:TEMP\t_cli.mov" --codec prores4444 -c acescg --threads 4` → Expected: renders without error (2 frames), exit 0.
Run: `python -c "import sys; sys.path.insert(0,'gui'); import motioncam_tools as mt; w = mt.RenderWorker([], {'encoder_threads': 4, 'colorspace': 'acescg'}); print('settings pass-through OK')"` → prints OK (constructor only; no render).
Run GUI syntax check: `python -c "import ast; ast.parse(open('gui/motioncam_tools.py', encoding='utf-8').read()); print('syntax OK')"`.

- [ ] **Step 4: Commit**

```powershell
git add mcraw_render.cpp gui/motioncam_tools.py
git commit -m @'
Phase I.A1 - encoder-thread budget on the CLI (--threads) and GUI

GUI passes cores // concurrent_workers so parallel batch renders share
the machine instead of oversubscribing it; CLI takes an explicit
--threads override. Default everywhere stays 0 = all cores.
'@
```

---

### Task 4: A2 — pack kernels + `ProcessBayerToYuvPlanarHost` + verify binding

**Files:**
- Modify: `lib/include/motioncam/CudaHwHandoff.hpp` (declare entry point, after the Phase H block)
- Modify: `lib/cuda/BayerPipeline.cu` (PipelineCtx fields, ReleaseCtxLocked, two kernels, entry point)
- Modify: `python/mcraw_py.cpp` (verify-only binding `cuda_pack_yuv_planar`, inside the existing `#if MCRAW_HAVE_CUDA` block next to `cuda_process_frame_phase_c`)
- Create: `test/phase_i_pack_verify.py`

**Interfaces:**
- Consumes: `RunBayerChainLocked`, `PipelineCtx`/`gRenderCtx`, `BuildBakedBayerConstants` (mcraw_py.cpp) — all existing.
- Produces (Task 5 relies on these exact signatures):

```cpp
// CudaHwHandoff.hpp
// Phase I (A2): run the render-context bayer chain, then pack BT.709
// limited-range 10-bit planar YUV (low-bit u16, yuv42xp10le convention)
// and copy the planes to host. subsample422: true = 4:2:2 (chroma width
// w/2, horizontal pair average), false = 4:4:4. Host pointers must hold
// h rows of tightly-packed u16 (chroma rows are cw = subsample422 ?
// width/2 : width elements). Returns false on any CUDA error.
bool ProcessBayerToYuvPlanarHost(
    const uint16_t* bayer_host,
    const float wb[3],
    const BayerPipelineConstants& consts,
    bool subsample422,
    uint16_t* y_host,
    uint16_t* cb_host,
    uint16_t* cr_host);
```

- [ ] **Step 1: Write the failing verify script (kernel-level, numpy reference)**

```python
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
                if got.shape != want.shape:
                    check(f"{tag} {name}", False,
                          f"shape {got.shape} vs {want.shape}")
                    continue
                diff = np.abs(got.astype(np.int32) - want.astype(np.int32))
                worst = int(diff.max())
                frac = float((diff > 1).mean())
                check(f"{tag} {name}", worst <= 2 and frac < 1e-4,
                      f"max|diff|={worst} cv, frac(>1)={frac:.2e}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it to verify it fails for the right reason**

Run: `python test\phase_i_pack_verify.py`
Expected: `AttributeError: module 'mcraw' has no attribute 'cuda_pack_yuv_planar'`.

- [ ] **Step 3: Header declaration**

In `lib/include/motioncam/CudaHwHandoff.hpp`, after the Phase H block (`void ReleaseGlInterop();`), add the `ProcessBayerToYuvPlanarHost` declaration exactly as shown in **Interfaces** above.

- [ ] **Step 4: Context fields + release**

In `lib/cuda/BayerPipeline.cu`, add to `struct PipelineCtx` after the Phase H display-buffer fields:

```cpp
    // Phase I (A2): packed planar 10-bit YUV output for the pro-codec
    // render path (render ctx only). Device planes + pinned host staging.
    uint16_t* packY        = nullptr;
    uint16_t* packCb       = nullptr;
    uint16_t* packCr       = nullptr;
    size_t    packYCount   = 0;   // elements, not bytes
    size_t    packCCount   = 0;
    uint16_t* pinnedPack   = nullptr;   // Y then Cb then Cr, contiguous
    size_t    pinnedPackCount = 0;
```

And in `ReleaseCtxLocked`, after the `rgbaDev` line:

```cpp
    if (ctx.packY)      { cudaFree(ctx.packY);          ctx.packY  = nullptr; }
    if (ctx.packCb)     { cudaFree(ctx.packCb);         ctx.packCb = nullptr; }
    if (ctx.packCr)     { cudaFree(ctx.packCr);         ctx.packCr = nullptr; }
    ctx.packYCount = 0; ctx.packCCount = 0;
    if (ctx.pinnedPack) { cudaFreeHost(ctx.pinnedPack); ctx.pinnedPack = nullptr;
                          ctx.pinnedPackCount = 0; }
```

- [ ] **Step 5: Kernels + entry point**

In `lib/cuda/BayerPipeline.cu`, after `RgbFloatToRgba8Kernel` / `ProcessBayerToRgbaDevice` and before the `#ifdef _WIN32` GL-interop block, add:

```cpp
// ============================================================================
// Phase I (A2): float RGB -> planar 10-bit YUV for the pro-codec encoders.
// BT.709 limited range; 10-bit value in the LOW bits (yuv42xp10le
// convention, unlike P010's high-bit packing). Formulas match the numpy
// reference in test/phase_i_pack_verify.py — keep them in lockstep.
// ============================================================================

__device__ __forceinline__ float3 RgbClamp01(const float* rgb, size_t i) {
    float r = rgb[i * 3 + 0], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
    r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
    g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
    b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);
    return make_float3(r, g, b);
}

__global__ void RgbFloatToYuv444P10Kernel(
    const float* __restrict__ rgb,
    uint16_t* __restrict__ Y, uint16_t* __restrict__ Cb,
    uint16_t* __restrict__ Cr, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const size_t i = size_t(y) * size_t(width) + size_t(x);
    const float3 c = RgbClamp01(rgb, i);
    const float yp = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    const float cb = (c.z - yp) / 1.8556f;
    const float cr = (c.x - yp) / 1.5748f;
    Y[i]  = uint16_t(__float2int_rn(64.0f + 876.0f * yp));
    Cb[i] = uint16_t(__float2int_rn(512.0f + 896.0f * cb));
    Cr[i] = uint16_t(__float2int_rn(512.0f + 896.0f * cr));
}

// One thread per CHROMA sample (pixel pair). Writes two luma samples and
// one pair-averaged Cb/Cr, matching the numpy reference's 0::2/1::2 mean.
__global__ void RgbFloatToYuv422P10Kernel(
    const float* __restrict__ rgb,
    uint16_t* __restrict__ Y, uint16_t* __restrict__ Cb,
    uint16_t* __restrict__ Cr, int width, int height)
{
    const int cx = blockIdx.x * blockDim.x + threadIdx.x;   // pair index
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int cw = width / 2;
    if (cx >= cw || y >= height) return;
    const size_t i0 = size_t(y) * size_t(width) + size_t(cx) * 2;
    const float3 a = RgbClamp01(rgb, i0);
    const float3 b = RgbClamp01(rgb, i0 + 1);
    const float ypA = 0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z;
    const float ypB = 0.2126f * b.x + 0.7152f * b.y + 0.0722f * b.z;
    const float cb = ((a.z - ypA) / 1.8556f + (b.z - ypB) / 1.8556f) * 0.5f;
    const float cr = ((a.x - ypA) / 1.5748f + (b.x - ypB) / 1.5748f) * 0.5f;
    Y[i0]     = uint16_t(__float2int_rn(64.0f + 876.0f * ypA));
    Y[i0 + 1] = uint16_t(__float2int_rn(64.0f + 876.0f * ypB));
    const size_t ci = size_t(y) * size_t(cw) + size_t(cx);
    Cb[ci] = uint16_t(__float2int_rn(512.0f + 896.0f * cb));
    Cr[ci] = uint16_t(__float2int_rn(512.0f + 896.0f * cr));
}

bool ProcessBayerToYuvPlanarHost(
    const uint16_t* bayer_host,
    const float wb[3],
    const BayerPipelineConstants& C,
    bool subsample422,
    uint16_t* y_host,
    uint16_t* cb_host,
    uint16_t* cr_host)
{
    if (!IsCudaAvailable()) return false;
    if (!bayer_host || !wb || !y_host || !cb_host || !cr_host) return false;
    if (C.width <= 0 || C.height <= 0) return false;

    const int W = C.width, H = C.height;
    const int CW = subsample422 ? (W / 2) : W;
    const size_t nY = size_t(W) * size_t(H);
    const size_t nC = size_t(CW) * size_t(H);
    const size_t total = nY + 2 * nC;

    PipelineCtx& ctx = gRenderCtx;
    std::lock_guard<std::mutex> lock(ctx.mtx);
    if (!EnsureBuffers(ctx, W, H)) return false;
    if (ctx.packYCount != nY || ctx.packCCount != nC ||
        !ctx.packY || !ctx.packCb || !ctx.packCr) {
        if (ctx.packY)  { cudaFree(ctx.packY);  ctx.packY  = nullptr; }
        if (ctx.packCb) { cudaFree(ctx.packCb); ctx.packCb = nullptr; }
        if (ctx.packCr) { cudaFree(ctx.packCr); ctx.packCr = nullptr; }
        ctx.packYCount = 0; ctx.packCCount = 0;
        if (cudaMalloc(reinterpret_cast<void**>(&ctx.packY),  nY * 2) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&ctx.packCb), nC * 2) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&ctx.packCr), nC * 2) != cudaSuccess)
            return false;
        ctx.packYCount = nY;
        ctx.packCCount = nC;
    }
    if (ctx.pinnedPackCount != total || !ctx.pinnedPack) {
        if (ctx.pinnedPack) { cudaFreeHost(ctx.pinnedPack); ctx.pinnedPack = nullptr; }
        ctx.pinnedPackCount = 0;
        if (cudaMallocHost(reinterpret_cast<void**>(&ctx.pinnedPack),
                           total * 2) == cudaSuccess)
            ctx.pinnedPackCount = total;
        // Pinned staging is an optimisation; on failure we copy direct.
    }

    if (!RunBayerChainLocked(ctx, bayer_host, wb, C, W, H)) return false;

    dim3 block(32, 8);
    if (subsample422) {
        dim3 grid((CW + block.x - 1) / block.x, (H + block.y - 1) / block.y);
        RgbFloatToYuv422P10Kernel<<<grid, block, 0, ctx.stream>>>(
            ctx.rgbFloat, ctx.packY, ctx.packCb, ctx.packCr, W, H);
    } else {
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
        RgbFloatToYuv444P10Kernel<<<grid, block, 0, ctx.stream>>>(
            ctx.rgbFloat, ctx.packY, ctx.packCb, ctx.packCr, W, H);
    }
    if (cudaGetLastError() != cudaSuccess) return false;

    if (ctx.pinnedPack && ctx.pinnedPackCount == total) {
        uint16_t* pY  = ctx.pinnedPack;
        uint16_t* pCb = pY + nY;
        uint16_t* pCr = pCb + nC;
        if (cudaMemcpyAsync(pY,  ctx.packY,  nY * 2, cudaMemcpyDeviceToHost, ctx.stream) != cudaSuccess ||
            cudaMemcpyAsync(pCb, ctx.packCb, nC * 2, cudaMemcpyDeviceToHost, ctx.stream) != cudaSuccess ||
            cudaMemcpyAsync(pCr, ctx.packCr, nC * 2, cudaMemcpyDeviceToHost, ctx.stream) != cudaSuccess)
            return false;
        if (cudaStreamSynchronize(ctx.stream) != cudaSuccess) return false;
        std::memcpy(y_host,  pY,  nY * 2);
        std::memcpy(cb_host, pCb, nC * 2);
        std::memcpy(cr_host, pCr, nC * 2);
    } else {
        if (cudaMemcpy(y_host,  ctx.packY,  nY * 2, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(cb_host, ctx.packCb, nC * 2, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(cr_host, ctx.packCr, nC * 2, cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
    }
    return true;
}
```

Note: the render ctx `stream` is 0 (legacy default), so the async copies + `cudaStreamSynchronize(0)` are correct and keep ordering with the chain kernels.

- [ ] **Step 6: Verify-only pybind binding**

In `python/mcraw_py.cpp`, inside the `#if MCRAW_HAVE_CUDA` module section (immediately after the `cuda_process_frame_phase_c` def), add:

```cpp
    // Phase I (A2) correctness binding: run the GPU chain + YUV pack on one
    // frame; returns (Y, Cb, Cr) uint16 numpy arrays (10-bit values, low
    // bits). For test/phase_i_pack_verify.py only.
    m.def("cuda_pack_yuv_planar",
        [](PyDecoder& pyDec, int64_t timestamp,
           const std::string& target_colorspace, bool subsample422,
           bool highlight_recovery, bool bake_vignette) -> py::tuple {
            mc::Decoder& dec = *pyDec.underlying();
            mcc::OutputColorSpace cs;
            if (!mcc::ParseOutputColorSpace(target_colorspace, cs))
                throw std::runtime_error("unknown colorspace: " + target_colorspace);

            std::vector<uint8_t> rawBuf;
            nlohmann::json frameMeta;
            mcc::FrameParams params;
            {
                py::gil_scoped_release release;
                dec.loadFrame(timestamp, rawBuf, frameMeta);
                params = mcc::BuildFrameParams(frameMeta, dec.getContainerMetadata());
            }
            motioncam::cuda::BayerPipelineConstants C{};
            if (!BuildBakedBayerConstants(params, cs, C))
                throw std::runtime_error("target is OCIO-only; use a baked target");
            C.highlight_recovery = highlight_recovery ? 1 : 0;
            C.highlight_rolloff  =
                (highlight_recovery && mcc::IsDisplayEncoded(cs)) ? 1 : 0;
            if (!bake_vignette) { C.lsm_w = 0; C.lsm_h = 0; C.lsm_host = nullptr; }

            const int W = C.width, H = C.height;
            const int CW = subsample422 ? (W / 2) : W;
            py::array_t<uint16_t> Y({H, W});
            py::array_t<uint16_t> Cb({H, CW});
            py::array_t<uint16_t> Cr({H, CW});
            bool ok;
            {
                py::gil_scoped_release release;
                const uint16_t* raw = reinterpret_cast<const uint16_t*>(rawBuf.data());
                ok = motioncam::cuda::ProcessBayerToYuvPlanarHost(
                    raw, params.asShotNeutral, C, subsample422,
                    Y.mutable_data(), Cb.mutable_data(), Cr.mutable_data());
            }
            if (!ok) throw std::runtime_error("ProcessBayerToYuvPlanarHost failed");
            return py::make_tuple(Y, Cb, Cr);
        },
        py::arg("decoder"), py::arg("timestamp"), py::arg("target_colorspace"),
        py::arg("subsample422"), py::arg("highlight_recovery") = false,
        py::arg("bake_vignette") = true,
        "Phase I pack-kernel correctness probe (baked targets only).");
```

- [ ] **Step 7: Build**

Run: `cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && ninja -C build'` → clean.

- [ ] **Step 8: Run the verify script — expect pass**

Run: `python test\phase_i_pack_verify.py`
Expected: all 12 checks PASS (2 colorspaces × 2 subsamplings × 3 planes), max|diff| ≤ 2 cv with frac(>1) < 1e-4.

- [ ] **Step 9: Regression sweep**

Run: `python test\player_preview_verify.py` → all PASS (render-ctx buffer additions must not disturb the preview ctx).
Run: `$env:MCRAW_GPU_YUV = "1"; python test\smoke.py test\fixtures\smoke2.mcraw; Remove-Item Env:MCRAW_GPU_YUV` → `12/12 passed`.

- [ ] **Step 10: Commit**

```powershell
git add lib/include/motioncam/CudaHwHandoff.hpp lib/cuda/BayerPipeline.cu python/mcraw_py.cpp test/phase_i_pack_verify.py
git commit -m @'
Phase I.A2 - GPU YUV pack kernels (yuv42{2,4}p10le) + verify binding

RgbFloatToYuv444P10 / RgbFloatToYuv422P10 pack the render-context chain
output as BT.709 limited-range 10-bit planar (low-bit u16, pair-average
chroma for 422). ProcessBayerToYuvPlanarHost runs chain -> pack ->
pinned readback. Verified against the numpy reference formula at
<= 2 cv max, frac(>1) < 1e-4, on acescg + srgb, both subsamplings
(test/phase_i_pack_verify.py). Not yet wired into the encoder.
'@
```

---

### Task 5: A2 — MovEncoder pack mode + DoRender producer/consumer + fallback

**Files:**
- Modify: `lib/include/motioncam/MovEncoder.hpp` (PlanarYuvFrame struct + 2 methods)
- Modify: `lib/MovEncoder.cpp` (Impl flags, `EnableGpuBayerPipeline` gate, new methods)
- Modify: `python/mcraw_py.cpp` (`DoRender` pro-codec GPU branch)
- Modify: `test/phase_i_pack_verify.py` (append part 2)

**Interfaces:**
- Consumes: `ProcessBayerToYuvPlanarHost` (Task 4 signature), `EncodeSettings::encoderThreads` (Task 2).
- Produces:

```cpp
// MovEncoder.hpp
struct PlanarYuvFrame {                    // 10-bit values in low bits
    std::vector<uint16_t> y, cb, cr;       // tightly packed rows
    int width = 0, height = 0, chromaWidth = 0;
};
// Producer-thread half (GPU chain + pack + readback). May run concurrently
// with WriteVideoFrameYuv10 on another thread, never with itself.
// Returns false on any CUDA error -> caller must fall back to the CPU
// path (ProcessFrame + WriteVideoFrame) for this and remaining frames.
bool PackFrameFromBayer(const uint16_t* bayer, const float wb[3],
                        const float* lsm, int lsmWidth, int lsmHeight,
                        PlanarYuvFrame& out);
// Consumer-thread half: copy planes into the encoder frame and encode.
void WriteVideoFrameYuv10(const PlanarYuvFrame& f);
```

- [ ] **Step 1: Header additions**

In `lib/include/motioncam/MovEncoder.hpp`: add `#include <vector>` to the includes; add the `PlanarYuvFrame` struct at namespace scope right before `class MovEncoder`; add the two method declarations (exactly as in **Interfaces**, comments included) to the public section after `WriteVideoFrameFromBayer`.

- [ ] **Step 2: Impl flags + eligibility gate**

In `lib/MovEncoder.cpp`, `struct MovEncoder::Impl`, after `bayerConsts`:

```cpp
    // Phase I (A2): GPU bayer->packed-planar pipeline for the CPU
    // intermediate codecs (ProRes / DNxHR / CineForm). Mutually exclusive
    // with useCudaDirectKernel (that's the NVENC hwframe path).
    bool useGpuPackPipeline = false;
    bool packSubsample422   = false;
```

In `EnableGpuBayerPipeline`, replace the opening gate

```cpp
    if (!p->useCudaDirectKernel) return false;
```

with:

```cpp
    // Two GPU front-ends share this setup: the NVENC hwframe path (Phase
    // B/C) and the Phase I pack path for the CPU intermediate codecs.
    bool packMode = false;
    if (!p->useCudaDirectKernel) {
        const AVPixelFormat pf =
            static_cast<AVPixelFormat>(p->yuvFrame ? p->yuvFrame->format
                                                   : p->videoCtx->pix_fmt);
        const bool packFmt = (pf == AV_PIX_FMT_YUV422P10LE ||
                              pf == AV_PIX_FMT_YUV444P10LE);
        const char* env = std::getenv("MCRAW_GPU_YUV");
        const bool envOptIn = env && std::strcmp(env, "1") == 0;
        if (!packFmt || !envOptIn || !motioncam::cuda::IsCudaAvailable())
            return false;
        packMode = true;
        p->packSubsample422 = (pf == AV_PIX_FMT_YUV422P10LE);
    }
```

At the end of the function, after `p->useGpuBayerPipeline = true;` add `p->useGpuPackPipeline = packMode;`, and extend the success log so pack mode is visible:

```cpp
    if (packMode) {
        fprintf(stderr,
            "[MovEncoder] Phase I GPU pack pipeline enabled (target=%d, "
            "%s, curve=%d%s).\n",
            setup.targetColorSpace,
            p->packSubsample422 ? "yuv422p10" : "yuv444p10",
            curve, useLut ? ", 3D LUT" : "");
    }
```

Also confirm `<cstdlib>` is included for `std::getenv` (add if missing).

- [ ] **Step 3: Implement the two methods**

In `lib/MovEncoder.cpp`, after `WriteVideoFrameFromBayer`:

```cpp
bool MovEncoder::PackFrameFromBayer(
    const uint16_t* bayer, const float wb[3],
    const float* lsm, int lsmWidth, int lsmHeight,
    PlanarYuvFrame& out)
{
#if MCRAW_HAVE_CUDA
    if (!p->useGpuPackPipeline) return false;

    const int w = p->settings.width;
    const int h = p->settings.height;
    const int cw = p->packSubsample422 ? (w / 2) : w;
    out.width = w;
    out.height = h;
    out.chromaWidth = cw;
    out.y.resize(size_t(w) * size_t(h));
    out.cb.resize(size_t(cw) * size_t(h));
    out.cr.resize(size_t(cw) * size_t(h));

    // Per-frame LSM, same convention as WriteVideoFrameFromBayer.
    motioncam::cuda::BayerPipelineConstants C = p->bayerConsts;
    if (lsm && lsmWidth >= 2 && lsmHeight >= 2) {
        C.lsm_w = lsmWidth;
        C.lsm_h = lsmHeight;
        C.lsm_host = lsm;
    } else {
        C.lsm_w = 0; C.lsm_h = 0; C.lsm_host = nullptr;
    }

    return motioncam::cuda::ProcessBayerToYuvPlanarHost(
        bayer, wb, C, p->packSubsample422,
        out.y.data(), out.cb.data(), out.cr.data());
#else
    (void)bayer; (void)wb; (void)lsm; (void)lsmWidth; (void)lsmHeight; (void)out;
    return false;
#endif
}

void MovEncoder::WriteVideoFrameYuv10(const PlanarYuvFrame& f) {
    if (f.width != p->settings.width || f.height != p->settings.height)
        Throw("WriteVideoFrameYuv10: frame dimensions mismatch");

    int err = av_frame_make_writable(p->yuvFrame);
    if (err < 0) ThrowAv("av_frame_make_writable(yuv10)", err);

    const int h = f.height;
    for (int row = 0; row < h; ++row) {
        std::memcpy(p->yuvFrame->data[0] + size_t(row) * p->yuvFrame->linesize[0],
                    f.y.data() + size_t(row) * f.width,
                    size_t(f.width) * 2);
        std::memcpy(p->yuvFrame->data[1] + size_t(row) * p->yuvFrame->linesize[1],
                    f.cb.data() + size_t(row) * f.chromaWidth,
                    size_t(f.chromaWidth) * 2);
        std::memcpy(p->yuvFrame->data[2] + size_t(row) * p->yuvFrame->linesize[2],
                    f.cr.data() + size_t(row) * f.chromaWidth,
                    size_t(f.chromaWidth) * 2);
    }

    p->yuvFrame->pts = p->videoPts++;
    err = avcodec_send_frame(p->videoCtx, p->yuvFrame);
    if (err < 0) ThrowAv("avcodec_send_frame(yuv10)", err);
    while (true) {
        int rec = avcodec_receive_packet(p->videoCtx, p->pkt);
        if (rec == AVERROR(EAGAIN) || rec == AVERROR_EOF) break;
        if (rec < 0) ThrowAv("avcodec_receive_packet(yuv10)", rec);
        av_packet_rescale_ts(p->pkt, p->videoCtx->time_base,
                             p->videoStream->time_base);
        p->pkt->stream_index = p->videoStream->index;
        err = av_interleaved_write_frame(p->fmt, p->pkt);
        if (err < 0) ThrowAv("av_interleaved_write_frame(yuv10)", err);
        av_packet_unref(p->pkt);
    }
}
```

(Compile note: `WriteVideoFrame`'s existing packet drain stays as-is — a shared helper would touch the NVENC goto path for no behavioral gain; the drain loop is duplicated deliberately.)

- [ ] **Step 4: DoRender pack branch**

In `python/mcraw_py.cpp`, `DoRender`. Today the branch structure is `if (gpuBayerActive) { sequential NVENC loop } else { producer-consumer float path }`. Change the first branch condition to keep NVENC-only behavior, and add a pack branch. Concretely, replace

```cpp
        if (gpuBayerActive) {
```

with

```cpp
        const bool gpuPackActive = gpuBayerActive && !mcv::IsNvenc(vcodec);
        if (gpuBayerActive && !gpuPackActive) {
```

(the NVENC sequential loop body stays untouched), then insert BEFORE the existing `} else {` of the CPU producer-consumer branch:

```cpp
        } else if (gpuPackActive) {
            // Phase I (A2): producer decodes + runs the GPU chain + packs
            // planar YUV; consumer runs the (multithreaded) CPU encoder.
            // Any pack failure flips permanently to the CPU float path for
            // the remaining frames (logged once).
            struct PackedItem {
                bool packed = false;
                mcv::PlanarYuvFrame yuv;     // when packed
                std::vector<float> rgb;      // when !packed (CPU fallback)
                uint32_t w = 0, h = 0;
            };
            constexpr size_t kQueueLimit = 2;
            std::deque<PackedItem> queue;
            std::mutex mtx;
            std::condition_variable not_full, not_empty;
            std::atomic<bool> cancel{false};
            bool producer_done = false;
            std::exception_ptr producer_err;

            std::thread producer([&]() {
                try {
                    bool gpuOk = true;
                    std::vector<uint8_t> rb = std::move(rawBuf);
                    nlohmann::json fm;
                    mcc::FrameParams cp = params0;
                    int cachedIdx = s;
                    for (int srcIdx : plan.srcIndex) {
                        if (cancel.load()) break;
                        if (cancel_check()) { cancel.store(true); not_empty.notify_all(); break; }
                        if (srcIdx != cachedIdx) {
                            decoder.loadFrame(frames[srcIdx], rb, fm);
                            cp = mcc::BuildFrameParams(fm, containerMeta);
                            cachedIdx = srcIdx;
                        }
                        PackedItem item;
                        const uint16_t* raw = reinterpret_cast<const uint16_t*>(rb.data());
                        if (gpuOk) {
                            const bool useLsm = bake_vignette && !cp.lensShadingMap.empty();
                            item.packed = enc.PackFrameFromBayer(
                                raw, cp.asShotNeutral,
                                useLsm ? cp.lensShadingMap.data() : nullptr,
                                useLsm ? int(cp.lsmWidth)  : 0,
                                useLsm ? int(cp.lsmHeight) : 0,
                                item.yuv);
                            if (!item.packed) {
                                gpuOk = false;
                                fprintf(stderr,
                                    "[render] Phase I GPU pack failed; "
                                    "continuing on the CPU path.\n");
                            }
                        }
                        if (!item.packed) {
                            item.w = cp.width;
                            item.h = cp.height;
                            mcc::ProcessFrame(raw, cp, cs, item.rgb,
                                              highlight_recovery, bake_vignette);
                            xform.Apply(item.rgb.data(), item.w, item.h);
                            if (highlight_recovery && mcc::IsDisplayEncoded(cs)) {
                                mcc::HighlightRolloff(item.rgb.data(), item.w, item.h);
                            }
                        }
                        std::unique_lock<std::mutex> lk(mtx);
                        not_full.wait(lk, [&]{ return queue.size() < kQueueLimit || cancel.load(); });
                        if (cancel.load()) break;
                        queue.push_back(std::move(item));
                        lk.unlock();
                        not_empty.notify_one();
                    }
                } catch (...) {
                    producer_err = std::current_exception();
                }
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    producer_done = true;
                }
                not_empty.notify_all();
            });

            int written = 0;
            try {
                while (true) {
                    std::unique_lock<std::mutex> lk(mtx);
                    not_empty.wait(lk, [&]{ return !queue.empty() || producer_done; });
                    if (queue.empty() && producer_done) break;
                    PackedItem item = std::move(queue.front());
                    queue.pop_front();
                    lk.unlock();
                    not_full.notify_one();

                    if (item.packed) enc.WriteVideoFrameYuv10(item.yuv);
                    else             enc.WriteVideoFrame(item.rgb.data());
                    ++written;
                    if (has_progress) {
                        py::gil_scoped_acquire gil;
                        try { progress_obj(written, total_to_render); } catch (...) {}
                    }
                    if (cancel_check()) {
                        cancel.store(true);
                        not_full.notify_all();
                        break;
                    }
                }
            } catch (...) {
                cancel.store(true);
                not_full.notify_all();
                not_empty.notify_all();
                if (producer.joinable()) producer.join();
                throw;
            }
            producer.join();
            if (producer_err) std::rethrow_exception(producer_err);
```

Note: `rawBuf` is moved by whichever branch runs first — the NVENC loop also moves it; the branches are exclusive so this is safe. The CPU `else` branch still copies (`std::vector<uint8_t> rb = rawBuf;`) — leave it untouched.

- [ ] **Step 5: Build**

Run: `cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && ninja -C build'` → clean.

- [ ] **Step 6: Append part 2 to the verify script**

Add to `test/phase_i_pack_verify.py`, called from `main()` after the kernel checks (insert the call `rc = part2(clip)` before the FAILURES summary and fold its result in):

```python
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
        subprocess.run([sys.executable, "-c", code], env=env, check=True,
                       capture_output=True)
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
```

- [ ] **Step 7: Run the full verify script — expect pass**

Run: `python test\phase_i_pack_verify.py`
Expected: kernel checks PASS as before, plus 8 new part-2 checks PASS (2 codecs × [tags + 3 planes]). The GPU render's stderr must show `Phase I GPU pack pipeline enabled` (visible with `--keep`-style debugging if needed).

- [ ] **Step 8: Full regression + bench**

Run: `python test\smoke.py test\fixtures\smoke2.mcraw` → `12/12 passed` (env unset ⇒ pack path OFF; CPU output unchanged).
Run: `$env:MCRAW_GPU_YUV = "1"; python test\smoke.py test\fixtures\smoke2.mcraw; Remove-Item Env:MCRAW_GPU_YUV` → `12/12 passed` (ProRes cases now exercise the pack path).
Run: `python test\phase_g_concurrency_verify.py` → PASS (render-ctx additions must not break render+preview concurrency).
Run: `$env:MCRAW_GPU_YUV = "1"; python test\bench_pro_codecs.py; Remove-Item Env:MCRAW_GPU_YUV` and also without the env var; append both lines to the RESULTS block. Expected: prores4444 ≥ 10 fps with the env var set. If < 10 fps, record the number anyway and flag it in the commit message — the spec's fallback is revisiting Approach C, not blocking this landing.

- [ ] **Step 9: Commit**

```powershell
git add lib/include/motioncam/MovEncoder.hpp lib/MovEncoder.cpp python/mcraw_py.cpp test/phase_i_pack_verify.py test/bench_pro_codecs.py
git commit -m @'
Phase I.A2 - GPU-packed frames feed the pro-codec encoders

EnableGpuBayerPipeline now has a second front-end: for ProRes/DNxHR/
CineForm (yuv42{2,4}p10le) under MCRAW_GPU_YUV=1, the render producer
thread runs decode -> GPU chain -> planar pack -> pinned readback
(PackFrameFromBayer) while the consumer runs the slice-threaded
encoder (WriteVideoFrameYuv10). Pack failure falls back to the CPU
float path mid-clip, logged once. Verified end-to-end: decoded plane
mean offsets vs the CPU sws path within the Phase E.2 1.5 cv bar,
identical stream tags, smoke 12/12 both env states, concurrency
verify green. Bench numbers appended to test/bench_pro_codecs.py.
'@
```

---

### Task 6: Docs + spec closeout

**Files:**
- Modify: `docs/superpowers/specs/2026-07-05-pro-codec-gpu-speedup-design.md` (Status line)
- Modify: `README.md` (features list)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update the spec status**

Change `**Status:** Approved (Approach A)` to `**Status:** Implemented 2026-07-06 — see test/bench_pro_codecs.py RESULTS for final numbers`.

- [ ] **Step 2: README feature bullet**

In `README.md`, in the "Features at a glance" list after the Malvar-He-Cutler bullet, add (fill X/Y from the final bench run):

```markdown
- **Fast professional intermediates** (v0.7) — ProRes / DNxHR / CineForm
  encoders now use every CPU core, and on NVIDIA GPUs the RAW→YUV pipeline
  runs on the GPU with only packed 10-bit planes crossing the bus. ProRes
  4444 at 4K: X fps (was 1.2); DNxHR HQX: Y fps (was 2.3).
```

- [ ] **Step 3: Commit**

```powershell
git add docs/superpowers/specs/2026-07-05-pro-codec-gpu-speedup-design.md README.md
git commit -m "Phase I: docs closeout - spec status + README perf numbers"
```
