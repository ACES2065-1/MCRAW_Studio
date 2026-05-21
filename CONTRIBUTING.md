# Contributing to MCRAW Studio

Thanks for thinking about it. This doc is everything you need to go from a fresh `git clone` to running the dev build, finding your way around the code, and shipping a PR.

If you're just here to **report a bug** or **request a feature**, jump straight to [Bug reports](#bug-reports) — no need to read the rest.

---

## Bug reports

Open an issue at [github.com/ACES2065-1/MCRAW_Studio/issues](https://github.com/ACES2065-1/MCRAW_Studio/issues) with:

1. Your **Windows version** (Win10 / Win11, x64 build number) and **MotionCam app version** that recorded the clip.
2. The **render settings** you used (format, codec, colour space, transfer, recover highlights on/off).
3. The **log file** from `%LOCALAPPDATA%\MCRAWStudio\log.txt`. Attach it directly.
4. Where possible, **the `.mcraw` file** that triggered the issue. If it's large or private, a short clip trimmed via the in-app MCRAW trim feature is enough — the bug almost always reproduces on the first few frames.

A bug report without the log file is much harder to act on. Please attach it.

---

## Feature requests

Same place: [issues](https://github.com/ACES2065-1/MCRAW_Studio/issues). Helpful framing:

- **What** you want to do
- **Why** the current flow doesn't work for you
- **How** you'd expect it to work (sketches / mockups welcome but optional)

The roadmap is in the README. Anything not on it is fair game to suggest.

---

## Building from source

### Prerequisites

| | Tested version | Notes |
|---|---|---|
| OS | Windows 10 / 11 x64 | Linux / macOS not currently supported |
| CPU | x86-64 with **AVX2** | Intel Haswell (2013) / AMD Excavator (2015) or newer. The build uses `/arch:AVX2`. |
| Compiler | **MSVC** (VS 2022 Community / Pro / Enterprise, or VS 18 Insiders) | "Desktop development with C++" workload |
| Build tools | CMake 3.15+ and Ninja | Either install standalone or rely on vcpkg's bundled copies |
| Package manager | [vcpkg](https://github.com/microsoft/vcpkg) | Default location: `C:\dev\vcpkg`. Override via `VCPKG_ROOT` env var |
| Python | 3.12 x64 | Needed for the Python bindings + GUI |

### Install dependencies (vcpkg)

```powershell
vcpkg install openexr opencolorio ffmpeg[gpl,x264,x265,nvcodec] pybind11 --triplet x64-windows
```

This pulls in the heavy stuff (OpenEXR, OCIO, FFmpeg with GPL codecs, pybind11). First-time install takes 20-40 min; cached afterwards.

### Install Python deps

```powershell
py -3.12 -m pip install pyside6 pyinstaller
```

### Configure + build

```powershell
.\configure.bat
ninja -C build
```

`configure.bat` auto-detects vcpkg / MSVC / cmake / ninja. If your install isn't at the default paths, set any of these env vars:

- `VCPKG_ROOT` — your vcpkg checkout
- `MSVC_VCVARSALL` — full path to `vcvarsall.bat`
- `CMAKE_EXE` / `NINJA_EXE` — explicit paths to those tools

### Run the GUI from source

```powershell
py -3.12 gui\motioncam_tools.py
```

The GUI auto-finds `mcraw.cp312-win_amd64.pyd` in the `build/` directory next to the script.

### Bundle the .exe

```powershell
py -3.12 -m PyInstaller motioncam_tools.spec --noconfirm
```

Output lands in `dist\MCRAWStudio.exe`. Single-file, ~62 MB. The spec accepts `VCPKG_INSTALL_DIR` and `MSVC_REDIST_BIN_DIR` env vars if your install paths differ from the defaults.

---

## Project layout

```
.
├── CMakeLists.txt           Top-level build script
├── configure.bat            One-shot configure (vcvarsall + cmake)
├── motioncam_tools.spec     PyInstaller spec for the bundled .exe
│
├── lib/                     The transcoder library
│   ├── Decoder.cpp          MCRAW container parser (TLV chunks, footer index)
│   ├── RawData.cpp          Bayer decompression (current scheme)
│   ├── RawData_Legacy.cpp   Bayer decompression (legacy MCRAW v6 scheme)
│   ├── Trimmer.cpp          Bit-perfect MCRAW trim (no decode)
│   ├── ColorPipeline.cpp    WB + matrix + debayer orchestration
│   ├── Debayer.cpp          NormalizeBayer, bilinear debayer, highlight recovery
│   ├── BakedTransform.cpp   Pre-computed cam→output transforms (fast path)
│   ├── OcioTransform.cpp    OpenColorIO wrapper (slow path)
│   ├── Denoise.cpp          Edge-preserving chroma + luma denoise (MP4)
│   ├── MovEncoder.cpp       FFmpeg muxer + video encode wrapper
│   ├── ExrWriter.cpp        OpenEXR sequence writer
│   ├── Parallel.hpp         Parallel-for over CPU threads
│   └── include/motioncam/   Public headers
│
├── python/
│   └── mcraw_py.cpp         pybind11 bindings (Decoder, OcioTransform, render, trim)
│
├── gui/
│   ├── motioncam_tools.py   PySide6 GUI app (single file, ~1800 lines)
│   ├── style.qss            Dark-mode Qt stylesheet
│   ├── icon.png / .ico      App icon
│
├── mcraw_render.cpp         Standalone CLI transcoder
├── example.cpp              Minimal decoder usage example
│
├── thirdparty/              Vendored single-header / small libs (no submodules)
│   ├── nlohmann/json.hpp    JSON parser
│   ├── audiofile/           WAV writer
│   ├── simde/               SIMD portability headers
│   └── tinydng/             DNG writer (used by the example)
│
└── .github/workflows/build.yml   CI: Windows core-build smoke test
```

The C++ side compiles into three artifacts:

- `motioncam_decoder.lib` — container + bayer decode, no external deps.
- `mcraw_color.lib` — full transcoder library (depends on OpenEXR + OCIO + FFmpeg).
- `mcraw.cp312-win_amd64.pyd` — Python bindings.
- `mcraw_render.exe` — standalone CLI.
- `example.exe` — minimal decoder example.

---

## Pipeline at a glance

```
 .mcraw file
     │
     ▼
 Decoder  ──► (compressed bayer, frame metadata JSON, container metadata JSON)
     │
     ▼
 NormalizeBayer   ── black-level subtract + per-channel WB
     │
     ▼
 ApplyLensShading (optional, from LSM in frame metadata)
     │
     ▼
 DebayerBilinear  ── camera RGB
     │
     ▼
 NeutraliseClippedHighlights (optional, if "Recover highlights" is on)
     │
     ▼
 ApplyMatrixInPlace ── camera RGB → ACEScg (or LinearRec709 directly)
     │
     ▼
 OcioTransform / BakedTransform ── ACEScg → user-selected output space
     │
     ▼
 HighlightRolloff (optional, display-encoded outputs only)
     │
     ▼
 ExrWriter   OR   MovEncoder (sws_scale → YUV → libavcodec)
```

All stages are parallelised over CPU threads via `motioncam::internal::ParallelForRange` ([lib/Parallel.hpp](lib/Parallel.hpp)).

---

## Coding style

Nothing rigid. A few things that keep the codebase consistent:

- **C++17.** No 20 features. `std::filesystem` is fine.
- **MSVC `/W4` clean.** New code shouldn't add warnings. (There are some pre-existing ones in `RawData.cpp` from the original MotionCam upstream — don't worry about those.)
- **`/utf-8` source encoding.** ASCII-safe is best in `.bat` / `.cmake` files (em-dash etc. trips cmd.exe in batch).
- **RAII everywhere.** No `new` / `delete` pairs. `std::unique_ptr` for ownership, raw pointers only for non-owning references.
- **Exceptions for errors.** Throw `std::runtime_error` with a clear message. The Python binding layer catches them and re-raises as Python `RuntimeError`.
- **Comments only where the *why* isn't obvious.** Don't narrate what the code does — name things well instead. Do explain a workaround, a non-obvious invariant, or a constraint coming from an external library.
- **Python code:** PEP 8-ish. Type hints encouraged but not enforced.

---

## Submitting a PR

1. Fork on GitHub, branch off `main`.
2. Make the change. Keep PRs focused — one logical fix or feature per PR.
3. Test locally before pushing:
   - `ninja -C build` succeeds clean
   - Run the smoke tests against any `.mcraw` you have on hand:

     ```powershell
     py -3.12 test\smoke.py path\to\sample.mcraw
     ```

     Renders one short slice through every supported format / codec
     combination and ffprobes the outputs. Should take 20-30 s and end with
     `6/6 passed`. Catches most encoder regressions.
4. Push, open a PR against `main`. The Windows core-build CI runs automatically.
5. In the PR description: what changed, why, and what you tested. Screenshots welcome for any UI / output-quality changes.

I read PRs on weekends mostly. Don't be surprised by 3-7 day review latency. Drive-by typo / docs fixes get merged fast; feature work I want to look at carefully.

### What I'll push back on

- **Breaking API changes** without a clear reason — the Python bindings are user-facing.
- **New external dependencies** unless they replace something or unlock a much-requested feature. The vcpkg dep list is already heavy.
- **Removing comments that explain *why*.** Comments that explain *what* are fair game to delete.
- **Refactor-only PRs without an accompanying behaviour change** unless they fix a specific maintainability pain point I've also felt.

---

## Architecture decisions worth knowing

- **Container format (`.mcraw`)** is MotionCam's, not ours. We only read it. The container is a simple TLV-chunked layout with a footer index — see [lib/include/motioncam/Container.hpp](lib/include/motioncam/Container.hpp) and `Decoder::init()` for the parse order. Bayer compression is custom (Rice-coded predictor in `RawData.cpp`).

- **Two output transform fast paths.** When the user picks an output space that has a baked transform ([lib/BakedTransform.cpp](lib/BakedTransform.cpp)), we skip OCIO entirely — pure matrix math, ~3× faster. When the user picks something we haven't baked, we fall through to OCIO. The baked list is curated by what users actually pick.

- **Producer-consumer encode pipeline.** [mcraw_render.cpp::RunMov](mcraw_render.cpp) and [python/mcraw_py.cpp::DoRender](python/mcraw_py.cpp) both run a decode-and-colour thread that feeds a queue, with the encoder draining on the main thread. This overlaps decode of frame N+1 with encode of frame N — ~30-50% throughput win on multi-core.

- **Highlight recovery** lives in camera-RGB space *before* the cam→output matrix, in [Debayer.cpp::NeutraliseClippedHighlights](lib/Debayer.cpp). It uses a smoothstep blend on `max_norm` (saturation amount) gated by `min_norm` (colour-gate, so pure red lights don't get bleached white). The display-encoded rolloff in [Debayer.cpp::HighlightRolloff](lib/Debayer.cpp) is a separate stage that runs *after* the OCIO transform, only when the target is display-encoded.

- **GUI lives in one file** ([gui/motioncam_tools.py](gui/motioncam_tools.py), ~1800 lines). It's a known smell but works fine; we'd rather have new contributors edit one file than navigate a four-module package. Split is on the roadmap.

---

## License

Source: **Apache 2.0** ([LICENSE](LICENSE)).

The bundled `MCRAWStudio.exe` is GPL-2.0+ because it statically links FFmpeg with libx264 and libx265 — see [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt). The source you contribute is Apache 2.0; the *bundled* binary inherits the GPL constraint. If you're contributing code, you're contributing under Apache 2.0 to the source repo — no CLA, no copyright assignment.

---

## Questions?

Open an issue or ping in the discussion threads. Discord coming soon.

— ACES2065-1
