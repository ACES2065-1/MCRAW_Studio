# Phase J — Frame-parallel ProRes encoding (the encoder farm)

**Date:** 2026-07-06
**Status:** Approved (Approach 1 — farm inside MovEncoder)
**Prereq:** Phase I (encoder threading + GPU pack), landed 2026-07-06.

## Problem

After Phase I, `prores_ks` is purely encoder-bound: 3.2 fps at 4K 4444
with 12 slice threads (slice threading plateaus ~2.5x). ProRes matters
to the user's and other artists' delivery workflows. ProRes is
intra-only with per-frame bit targeting — no cross-frame encoder state —
so frames can be encoded by independent encoder instances and muxed back
in order, with output bit-identical to serial encoding.

## Goal / success criteria

- ProRes 4444 at 4K: **>= 8 fps** with the GPU feed (from 3.2).
- Farm output decodes **bit-identical** to `encoder_instances=1` output
  (hash), for 4444 and 422HQ, on both the GPU pack path and the CPU
  float path.
- Frame count, duration, and audio unchanged (ffprobe).
- Transparent to callers: no GUI/CLI/DoRender pipeline changes required
  beyond the new knob; non-ProRes codecs and NVENC/MP4 untouched.
- CPU-only machines benefit too: each worker owns its own SwsContext,
  so the float->YUV conversion parallelises along with the encode.

## Design

**ProResFarm inside `MovEncoder::Impl`** (`lib/MovEncoder.cpp`):

- N workers, each owning its own opened `prores_ks` AVCodecContext
  (instance 0 reuses `p->videoCtx`, so header/codecpar setup is
  unchanged; siblings are configured from the same fields: dims,
  time_base, pix_fmt, profile, `mbs_per_slice=4`, `vendor=apl0`, color
  props, GLOBAL_HEADER flag, per-instance thread_count).
- Bounded input queue (capacity instances + 2; ~250 MB worst case at
  4K 4444) fed by the existing `WriteVideoFrame` (float path — worker
  does deinterleave + own sws) and `WriteVideoFrameYuv10` (planar path
  — worker copies rows), which become submits when the farm is active.
  Frame index is assigned at submit and becomes `AVFrame::pts`.
- Reorder stage: workers push encoded packets keyed by `pkt->pts` into
  a map; a single muxer thread writes strictly next-expected-index via
  `av_interleaved_write_frame` (it is the only thread touching the
  muxer while video is in flight). Flush packets (send_frame(NULL) per
  worker at end) flow through the same keyed map, so encoder delay,
  if any, is handled uniformly.
- Drain barrier: `WriteAudio` and `Finalize` first close the input
  queue, join workers (each flushes its ctx), and join the muxer; if
  the muxer is missing a packet for an expected index when all workers
  are done, that is an error ("encoder dropped frame"), not a hang.
  `Finalize` skips its serial video flush in farm mode (the farm
  already flushed every context, including `videoCtx`).
- Errors: first worker/muxer exception is captured and rethrown on the
  next submit or at the drain barrier, mirroring DoRender's
  producer_err pattern. Destructor tears down via the same drain with
  errors swallowed (as today).

**Eligibility & knobs:**

- Farm activates only for the four ProRes profiles and only when
  instances > 1. New `EncodeSettings::encoderInstances`
  (0 = auto, 1 = serial/off, N = explicit), exposed as
  `encoder_instances=0` on `mcraw.render()` and `--instances` on the
  CLI. GUI passes nothing — auto derives from the existing
  `encoder_threads` budget.
- Auto rule (decided in the MovEncoder constructor, before
  `avcodec_open2`): `budget = encoderThreads > 0 ? encoderThreads :
  hardware_concurrency`; `instances = clamp(budget / 3, 1, 4)`
  (explicit N is clamped to budget); per-instance
  `thread_count = max(1, budget / instances)`. Slice threading at 3
  threads is near its efficient region; 4 instances x 3 threads beats
  1 x 12.

## Verification

- `test/prores_farm_verify.py`: for prores4444 and prores422hq, render
  the same 16-frame range with `encoder_instances=1` and auto; decoded
  raw video must hash identical; ffprobe frame count/duration/audio
  stream equal; auto must be > 1.8x faster for 4444 (hard gate).
  Runs twice: `MCRAW_GPU_YUV=1` (planar submits) and env unset (float
  submits).
- Existing gates stay green: smoke 12/12 both env states,
  `a1_threading_verify`, `phase_i_pack_verify`.
- `test/bench_pro_codecs.py` re-run; RESULTS line appended. Success:
  prores4444 >= 8 fps GPU-fed.

## Out of scope

DNxHR/CineForm farm eligibility (one-line extension later if ever
needed), NVENC/MP4 paths, EXR, segment-split/concat approaches.
