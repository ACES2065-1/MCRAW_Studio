# Phase J: ProRes Encoder Farm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.
> Executed inline by the plan author in the same session (full codebase
> context); code detail lives in the implementation commits, interfaces and
> gates are pinned here.

**Goal:** ProRes 4444 at 4K from 3.2 fps to >= 8 fps via N parallel
`prores_ks` instances with in-order muxing, bit-identical to serial output.

**Spec:** `docs/superpowers/specs/2026-07-06-prores-farm-design.md`

## Global Constraints

- Build: `cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && ninja -C build'`
- No Co-Authored-By trailer. MCRAW_GPU_YUV=1 gates the GPU feed only —
  the farm itself must work on both feeds.
- Determinism bar: decoded raw video hash(instances=auto) ==
  hash(instances=1), prores4444 + prores422hq, both env states.
- Existing gates stay green: smoke 12/12 x2, a1_threading_verify,
  phase_i_pack_verify.

### Task 1: failing verify script

`test/prores_farm_verify.py` — subprocess-renders (env evaluated fresh)
16 frames with `encoder_instances=1` vs `0` (auto) for prores4444 +
prores422hq, in both MCRAW_GPU_YUV states; decoded-hash equality,
ffprobe nb_read_frames/duration/audio-codec equality, >1.8x speedup gate
for 4444. First run must fail with unknown-kwarg TypeError.

### Task 2: implementation

**Interfaces produced:**
- `EncodeSettings::encoderInstances` (int, 0=auto, 1=off, N=explicit) —
  `MovEncoder.hpp`.
- `mcraw.render(..., encoder_instances=0)` — `python/mcraw_py.cpp`.
- `mcraw_render.exe --instances <n>` — `mcraw_render.cpp`.
- No DoRender/GUI changes: `WriteVideoFrame` / `WriteVideoFrameYuv10`
  transparently submit to the farm when active.

**MovEncoder.cpp internals (per spec):** `FarmJob {idx, planar|float
payload}`; bounded input deque (cap instances+2); N workers each owning
an AVCodecContext (worker 0 = `p->videoCtx`; siblings opened from the
same field set incl. profile/mbs_per_slice/vendor/color/GLOBAL_HEADER,
thread_count = budget/instances), own AVFrame, lazy own sws+staging for
float jobs; packets keyed by `pkt->pts` into a reorder map; muxer thread
writes strictly next index; drain barrier (close input -> join workers,
each flushes its ctx -> join muxer; missing expected index == error) at
the top of `WriteAudio`, `Finalize`, and the destructor; first error
rethrown at submit/drain. Constructor decides instances BEFORE
avcodec_open2 (auto = clamp(budget/3, 1, 4)) and sets per-instance
thread_count.

### Task 3: verification + bench

Build; run `prores_farm_verify.py` (both states pass); smoke x2;
`a1_threading_verify.py`; `phase_i_pack_verify.py`;
`phase_g_concurrency_verify.py`; bench both states, append RESULTS.
Gate: prores4444 >= 8 fps GPU-fed (record + flag honestly if short).

### Task 4: docs closeout

Bench header, README bullet update, spec Status line. Commit.
