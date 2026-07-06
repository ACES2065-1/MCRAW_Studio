# Phase I — Fast professional-format renders (encoder threading + GPU pixel packing)

**Date:** 2026-07-05
**Status:** Implemented 2026-07-06. Final numbers (MCRAW_GPU_YUV=1,
idle GPU): dnxhr_hqx 15.4 fps (target met), cineform 28.7, prores422hq
4.7, prores4444 3.2 — prores_ks is now purely encoder-bound, so the 10 fps
ProRes target needs Approach C (frame-parallel encode), per the fallback
clause below. Implementation also fixed a latent E.2-class bug: the pro
intermediates' sws path defaulted to BT.601 while their tags/decoders
assume BT.709 — now pinned. See test/bench_pro_codecs.py RESULTS.
**Owner:** MCRAW Studio

## Problem

MP4/NVENC renders are GPU-fast, but the professional MOV codecs are not.
Measured 2026-07-05 on the dev box (RTX 4090, 12-core, 4032x1696 clip
`VIDEO_20250114_130136.0.mcraw`, 48 frames, acescg):

| Path                          | fps  |
|-------------------------------|------|
| GPU bayer chain (no readback) | 60.1 |
| CPU pipeline (process_frame)  | 12.0 |
| render prores4444             | 1.2  |
| render prores422hq            | 1.7  |
| render dnxhr_hqx              | 2.3  |
| render cineform               | 5.9  |

Two root causes, confirmed by inspection of `lib/MovEncoder.cpp`:

1. **Single-threaded encoders.** `thread_count` is never set on the
   `AVCodecContext`; libavcodec's default is one thread. `prores_ks`,
   `dnxhd`, and `cineform` all support slice threading. (libx264/x265
   self-thread — smoke logs show `threads=18` / `pool 12` — and NVENC is
   hardware; both unaffected.)
2. **CPU float→YUV conversion.** The consumer thread runs
   `sws_scale(GBRPF32LE → yuv42{2,4}p10le)` in series with the encode,
   while the GPU chain that already produces the final RGB for these
   targets sits idle.

## Goal / success criteria

- ProRes 4444 at 4K reaches **>= 10 fps** on the dev box (>= 8x today).
- All pro MOV codecs improve; nothing regresses for MP4/NVENC/EXR.
- Non-NVIDIA machines still get the threading win (A1 is CPU-only).
- Output remains verified: bit-identical for A1; within the Phase E.2
  numeric bar for A2 (see Verification).

## Component 1 — Encoder multithreading (A1)

**Where:** `lib/MovEncoder.cpp`, before `avcodec_open2`.

- `prores_ks`, `dnxhd`: `thread_type = FF_THREAD_SLICE`,
  `thread_count = <budget>`.
- `cineform`: `thread_type = FF_THREAD_SLICE | FF_THREAD_FRAME`,
  `thread_count = <budget>` (FFmpeg masks unsupported types).
- libx264 / libx265 / NVENC / ProRes-on-GPU paths: untouched.

**Thread budget plumbing:** new `EncodeSettings::encoderThreads`
(0 = auto = all logical cores). Exposed as `encoder_threads=0` kwarg on
`mcraw.render()` and `--threads` on `mcraw_render.exe`. The GUI passes
`max(1, os.cpu_count() // concurrent_workers)` so concurrent renders
don't oversubscribe.

**Verification:**
- `prores_ks` slicing is deterministic w.r.t. thread count, so a
  threaded and a single-threaded render of the same input must decode
  to identical frames (hash comparison in the verify script). If a
  codec turns out NOT to be thread-count-deterministic, the check for
  that codec downgrades to the numeric bar (mean abs decoded diff
  < 0.5 cv @10-bit) with a comment explaining why.
- `test/bench_pro_codecs.py` (committed) records fps before/after.
- Smoke suite 12/12.

## Component 2 — GPU-packed frames for pro codecs (A2, "Phase I" GPU work)

**Kernels** (in `lib/cuda/BayerPipeline.cu`, render context, following the
established kernel pattern):

- `RgbFloatToYuv444P10Kernel`: float RGB → 3 planar u16 planes, 10-bit
  limited range in the low bits (values 64–940 style, matching what
  sws feeds `yuv444p10le`), BT.709 coefficients identical to the
  pinned `sws_setColorspaceDetails` setup from Phase E.2.
- `RgbFloatToYuv422P10Kernel`: same, with 2:1 horizontal chroma
  averaging of pixel pairs, siting matched to sws output (validated
  numerically, not assumed).

**Entry point:** `ProcessBayerToYuvPlanarHost(bayer_host, wb, consts,
format, dst_planes[3], dst_pitches[3])` on the **render** context: run
the existing chain (normalize → LSM → MHC debayer → highlight recovery →
matrix/curve or Phase D LUT → rolloff) → pack kernel into device planar
buffers → async copy to pinned staging → memcpy into the caller's
`AVFrame` planes. Device planar + pinned staging buffers live in the
render `PipelineCtx` and are freed by `ReleaseBayerPipeline`.

**MovEncoder wiring:** `EnableGpuBayerPipeline` eligibility extends to
MOV codecs whose negotiated pix_fmt is `yuv422p10le` or `yuv444p10le`
(ProRes 422/422HQ/4444/4444XQ, DNxHR HQX/444, CineForm), under the same
`MCRAW_GPU_YUV=1` opt-in as the NVENC path (GPU output differs from sws
by rounding, so it stays behind the existing switch).
`WriteVideoFrameFromBayer` fills `yuvFrame` via the GPU path and calls
`send_frame`; the DoRender producer/consumer keeps its shape — producer
does decode → GPU chain → readback into the queue, consumer encodes.

**Fallback:** any CUDA failure returns false and that frame (and the
rest of the clip) takes the CPU `ProcessFrame` + sws path, mirroring the
NVENC path's per-frame fallback. No user-visible error, logged once.

**Interaction rules:** denoise is MP4-only (never eligible here);
frame-rate conversion, trims, audio, and vignette toggle are orthogonal
and must keep working — covered by smoke.

**Verification:**
- `test/phase_i_pack_verify.py`: render the same range GPU-packed and
  CPU-sws, decode both with ffmpeg, require mean luma AND mean chroma
  offsets < 1.5 cv @10-bit per plane (the Phase E.2 bar) and identical
  nclc tags. Runs for one 422 codec and one 444 codec at minimum.
- Smoke 12/12 in both `MCRAW_GPU_YUV` states.
- Bench script re-run; numbers appended to its header comment.

## Out of scope (deferred)

- Approach C (frame-parallel encoder farm) — revisit only if A misses
  the 10 fps target.
- EXR writer speedup, `prores_aw`, 12-bit 4444 profiles, MP4/NVENC
  changes, preview/player changes.

## Landing order

1. A1 threading + bench + verify scripts (independent commit).
2. Measure; record in bench header.
3. A2 pack kernels + `ProcessBayerToYuvPlanarHost` + numeric verify
   binding/script (kernels verified BEFORE encoder wiring, per house
   pattern).
4. MovEncoder wiring + fallback + smoke both env states.
5. GUI thread-budget policy + CLI `--threads`.

Baseline numbers to beat are in the Problem table above.
