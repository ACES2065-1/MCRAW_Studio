# MCRAW Studio — what it does

*by **ACES2065-1***

A single Windows app for converting MotionCam Pro `.mcraw` recordings into
the format you actually need. Drag clips in, pick where they should go, hit
render.

Designed for VFX, color, and "I just want a normal video" workflows alike.

---

## Getting started

1. Download `MCRAWStudio.exe`. No installer.
2. Double-click it. (First launch: Windows may say *unrecognized app* —
   click **More info** → **Run anyway**. Once we sign the binary this goes
   away.)
3. Drag your `.mcraw` files onto the file list, or click **Add Files…**.
4. Click any file to see a preview frame. Drag the slider to scrub.
5. Pick your **Format** below (MP4 / MOV / EXR / MCRAW). The other settings
   change to match — you only see what's relevant.
6. **Render All** — outputs land next to your source files.

---

## What you can render to

### MP4 — for normal viewing

The thing you send to clients, post on YouTube, watch on TV.

- **H.264** or **H.265** codec (8-bit standard, 10-bit optional for H.265)
- **GPU acceleration via NVENC** if you have any NVIDIA GPU — way faster
  than CPU encoding (about 5–10× on H.265)
- **AV1 (GPU)** if you have an RTX 40-series — newer codec, smaller files
  at the same quality
- **Audio plays everywhere** — MP4s use AAC so Windows Media Player,
  browsers, and phones decode them without complaint
- **Optional denoise** with two sliders (color noise / luma noise) like
  Camera Raw

### MOV — for grading and editing

What Resolve, Premiere, and Avid expect when you bring footage into a
project.

- **ProRes**: 422, 422 HQ, 4444, 4444 XQ — the Apple intermediates
- **DNxHR**: HQX (10-bit), 444 (12-bit) — the Avid/Adobe intermediates
- **GoPro CineForm** — visually-lossless 10-bit working codec
- **H.264 / H.265 / AV1** also available in MOV if you want
- **Multiple clips render in parallel** for batch jobs (1–4 concurrent;
  ~2.7× speedup at 3 workers on ProRes 4444)

### EXR — for VFX and compositing

Half-float linear sequences for Nuke, Fusion, Resolve's color page in
ACEScg, or any deep VFX work.

- **Per-frame .exr** files in a folder, ready to drop into a compositor
- **Compression options**: PIZ (lossless, recommended), ZIP (classic),
  DWAB (lossy, ~5× smaller), and the rest of OpenEXR's options
- **Parallel rendering** — 8 worker threads, hits ~10 fps at 4K
- **Embedded chromaticities** + OCIO color space tags so downstream tools
  pick the right input transform automatically

### MCRAW (trimmed) — keep your RAW for later

The new "save a piece of this clip without losing anything" feature.

- Set **Mark In** and **Mark Out** on the timeline scrubber
- Render → bit-perfect copy of just that frame range as a new `.mcraw`
- Audio is correctly sliced to match
- The output behaves exactly like a fresh MotionCam recording — drop it
  back into this app and re-render to whatever you want
- **No quality loss whatsoever** — no decode, no re-encode, just a smart
  copy of the bytes you care about

---

## Color spaces and transfers

You pick **Color Space** (gamut) and **Transfer Function** (curve) as two
separate dropdowns, like the MotionCam Pro mobile app:

| Color Space | What it's for |
|---|---|
| BT.709 / sRGB | Standard SDR delivery |
| Rec.2020 | HDR and wide-gamut delivery |
| DaVinci Wide Gamut | DaVinci Resolve users — drop-in to a DaVinci timeline |
| ACEScg (AP1) | VFX intermediates, ACES working space |
| ACES AP0 | ACES archival, exchange between facilities |

| Transfer | What it's for |
|---|---|
| Linear | EXR, VFX, scene-referred work |
| BT.709 / Gamma 2.4 | SDR display delivery |
| sRGB | Web video, computer monitors |
| ST2084 PQ | HDR10 delivery (auto-locks to 10-bit) |
| HLG | HDR HLG delivery (auto-locks to 10-bit) |
| Cineon Log | VFX log archival (with ACES AP0 → ADX10) |
| DaVinci Intermediate | DaVinci grading log |
| ACES CCT | ACES log working space |

Only valid combinations are selectable — invalid ones (e.g. *DaVinci
Intermediate* with *Rec.2020 primaries*) are filtered out automatically.

---

## What's automatically handled for you

- **Lens shading / vignette correction** — every MotionCam frame ships
  with a per-channel correction map. We apply it before debayer so corners
  come out the same brightness as the center, no manual setup.
- **Debayer** (bilinear) — converts the bayer raw to full RGB.
- **White balance + color matrix** — uses the per-frame `asShotNeutral`
  and forward matrix from the source metadata.
- **Audio passes through cleanly** — PCM in MOV, AAC in MP4, sliced to
  match the trim range.
- **Frame timestamps preserved** through trim — re-trim a trimmed file
  and timing stays consistent.

---

## Performance

Real numbers on a 4K (4032 × 1696) clip on an RTX 4090 + 16-core CPU:

| Render target | fps | Notes |
|---|---|---|
| H.265 NVENC (8-bit) | ~10 fps | Producer-bound; GPU encoder is hardly working |
| H.265 NVENC (10-bit) | ~8 fps | Slightly slower due to higher pix_fmt |
| AV1 NVENC | ~8 fps | RTX 40-series; same end-to-end as H.265 NVENC |
| H.265 CPU (libx265) | ~1.7 fps | No NVIDIA GPU? This is the fallback |
| ProRes 4444 (single) | ~0.7 fps | 4K ProRes is heavy, no GPU exists for it |
| ProRes 4444 (3 parallel) | ~1.9 fps aggregate | **2.75× speedup** with concurrent rendering |
| DNxHR HQX | ~2 fps | High-quality CPU intermediate |
| EXR sequences (PIZ, parallel) | ~10 fps | 8 worker threads |

For typical batch work — say 10× 5-second clips — that means:
- **Bulk H.265 NVENC**: ~25 sec total
- **Bulk ProRes 4444 with concurrent=3**: ~1.5 minutes (down from ~4
  minutes single-threaded)
- **Bulk EXR** (PIZ): ~25 sec total

---

## Practical recipes

**"I just want to send this to my client"**
→ Format: MP4 · Codec: H.265 (GPU, NVENC) · Color: BT.709 · Transfer:
BT.709 / Gamma 2.4

**"This clip is too noisy"**
→ Same as above, plus *Denoise chroma 50* (gets rid of the colored
shadow blobs without softening detail)

**"I want to grade this in Resolve"**
→ Format: MOV · Codec: ProRes 4444 · Color: DaVinci Wide Gamut ·
Transfer: DaVinci Intermediate

**"I'm doing VFX in Nuke"**
→ Format: EXR · EXR compression: PIZ · Color: ACEScg (AP1) · Transfer:
Linear

**"I want HDR10"**
→ Format: MP4 · Color: Rec.2020 · Transfer: ST2084 PQ — codec list will
auto-filter to 10-bit-capable encoders and the 10-bit toggle will lock on

**"I just want to keep frames 30 to 120 of this 5-minute clip and toss
the rest"**
→ Format: MCRAW · scrub to frame 30, *Mark In* · scrub to frame 120,
*Mark Out* · *Render All*

---

## Known limitations

- **First-launch SmartScreen warning** — until we get a code-signing
  certificate. Click *Run anyway* once and Windows remembers.
- **libx265 10-bit** silently downgrades to 8-bit on most builds — needs
  a multi-bit-depth FFmpeg build to fix. NVENC 10-bit works regardless.
- **Apple Silicon Macs / Linux** — not built yet. The codebase is
  cross-platform; only the bundled .exe is Windows-specific. Reach out if
  you need a Mac/Linux build.
- **Bilinear debayer** — not the fanciest demosaic algorithm. We're
  planning to upgrade to Malvar-He-Cutler for finer detail at edges.

---

## Reporting bugs / requesting features

Open an issue at [the project repo](https://github.com/) (TBD).

When reporting a render bug, include:
- Source clip's resolution and approximate length
- Format / codec / color space / transfer combination you picked
- Any error message in the file list, and the **About → Help** dialog
  contents (it shows version, NVENC availability, etc.)

A short MCRAW sample (use the *MCRAW (trimmed)* feature to extract a
~30-frame piece of a problematic clip!) is incredibly helpful for
diagnosing format-specific bugs.

---

Happy rendering.
