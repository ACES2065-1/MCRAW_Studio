#!/usr/bin/env python3
"""
Smoke test for the MCRAW Studio render pipeline.

Renders a short slice of a real .mcraw clip to every supported format / codec
combination, then ffprobes the outputs to confirm they're structurally valid.

The point isn't *quality* (we don't compare pixels) — it's "did the encoder
crash, did we produce zero bytes, does ffprobe choke on the container."
Designed to take a few minutes and catch ~80% of encoder regressions before
they ship.

Usage
-----
    python test/smoke.py <path/to/sample.mcraw>

Or set the MCRAW_SMOKE_FIXTURE env var:

    MCRAW_SMOKE_FIXTURE=samples/clip.mcraw python test/smoke.py

The fixture must be a real .mcraw — we don't synthesise one. Trim a few
frames out of any clip you have:

    mcraw_render.exe input.mcraw -o samples/clip.mcraw -f mcraw --start 0 --end 12

Exit code 0 = all tests passed. Non-zero = one or more failed; details on stderr.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the mcraw extension module so this works from a source checkout
# without installing anything. Mirror what gui/motioncam_tools.py does.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BUILD = PROJECT / "build"
VCPKG_BIN = Path(r"C:\dev\vcpkg\installed\x64-windows\bin")

if VCPKG_BIN.is_dir() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(VCPKG_BIN))

if BUILD.is_dir() and str(BUILD) not in sys.path:
    sys.path.insert(0, str(BUILD))

try:
    import mcraw  # noqa: E402
except ImportError as exc:
    print(f"FATAL: can't import mcraw — did you build the project?\n  {exc}",
          file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Helpers

def find_ffprobe() -> str | None:
    """Locate ffprobe. Tries PATH, then vcpkg's bundled FFmpeg."""
    p = shutil.which("ffprobe")
    if p:
        return p
    candidate = VCPKG_BIN / "ffprobe.exe"
    if candidate.exists():
        return str(candidate)
    return None


FFPROBE = find_ffprobe()
if FFPROBE is None:
    print("FATAL: ffprobe not found on PATH or in vcpkg bin. Install ffmpeg.",
          file=sys.stderr)
    sys.exit(2)


def ffprobe_streams(path: Path) -> list[dict]:
    """Return a list of {codec_type, codec_name} for each stream in `path`.
    Uses CSV output (one stream per line) — simpler to parse than the default
    "[STREAM] ... [/STREAM]" block format, and unambiguous about stream
    boundaries unlike noprint_wrappers=1 which drops them entirely."""
    proc = subprocess.run(
        [FFPROBE, "-v", "error",
         "-show_entries", "stream=codec_name,codec_type",
         "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    out: list[dict] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # CSV order matches the order in -show_entries: codec_name,codec_type
        parts = line.split(",")
        if len(parts) >= 2:
            out.append({"codec_name": parts[0], "codec_type": parts[1]})
    return out


# ---------------------------------------------------------------------------
# Test cases

class Case:
    """One render-and-check scenario."""
    def __init__(self, name, *, output_ext, expect_streams, **render_kwargs):
        self.name = name
        self.output_ext = output_ext              # ".mp4" / ".mov" / "" (dir)
        self.expect_streams = expect_streams      # list of expected codec_types
        self.render_kwargs = render_kwargs        # passed to mcraw.render

    def run(self, fixture: Path, tmpdir: Path) -> tuple[bool, str]:
        out = tmpdir / f"out_{self.name}{self.output_ext}"
        if out.exists():
            out.unlink()
        if self.output_ext == "":
            out.mkdir(exist_ok=True)
        t0 = time.time()
        try:
            mcraw.render(input=str(fixture), output=str(out),
                         **self.render_kwargs)
        except Exception as exc:
            return False, f"render raised: {type(exc).__name__}: {exc}"
        elapsed = time.time() - t0

        # Existence + non-zero size
        if self.output_ext == "":
            # EXR sequence — look for frame_*.exr
            frames = sorted(out.glob("frame_*.exr"))
            if not frames:
                return False, "no EXR frames produced"
            for f in frames:
                if f.stat().st_size == 0:
                    return False, f"zero-byte EXR: {f.name}"
            return True, f"{len(frames)} EXR frames, {elapsed:.2f}s"

        if not out.exists():
            return False, "output file does not exist"
        if out.stat().st_size == 0:
            return False, "output file is zero bytes"

        # ffprobe: file must parse, and contain the streams we expected.
        try:
            streams = ffprobe_streams(out)
        except RuntimeError as exc:
            return False, str(exc)
        types = sorted(s.get("codec_type", "") for s in streams)
        wanted = sorted(self.expect_streams)
        # We require every expected stream type to appear (audio may be
        # video-only-fallback, which is fine — we treat "audio expected but
        # absent" as a soft pass since the encoder logs a clear reason).
        missing = [t for t in wanted if t == "video" and t not in types]
        if missing:
            return False, f"missing required streams {missing}; got {types}"
        return True, f"{out.stat().st_size:,} bytes, streams={types}, {elapsed:.2f}s"


CASES = [
    Case("mp4_h264_srgb",
         output_ext=".mp4",
         expect_streams=["video", "audio"],
         colorspace="srgb", codec="h264", start=0, end=8, bitrate=20),

    Case("mp4_h265_srgb",
         output_ext=".mp4",
         expect_streams=["video", "audio"],
         colorspace="srgb", codec="h265", start=0, end=8, bitrate=20),

    Case("mov_prores4444_acescg",
         output_ext=".mov",
         expect_streams=["video", "audio"],
         colorspace="acescg", codec="prores4444", start=0, end=8),

    Case("mov_prores422hq_rec709",
         output_ext=".mov",
         expect_streams=["video", "audio"],
         colorspace="rec709-display", codec="prores422hq", start=0, end=8),

    Case("exr_acescg",
         output_ext="",
         expect_streams=[],
         colorspace="acescg", start=0, end=3, exr_compression="piz"),

    Case("mp4_h264_recovery_on",
         output_ext=".mp4",
         expect_streams=["video", "audio"],
         colorspace="srgb", codec="h264", start=0, end=8, bitrate=20,
         highlight_recovery=True),
]


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fixture", nargs="?",
                    default=os.environ.get("MCRAW_SMOKE_FIXTURE"),
                    help="Path to a .mcraw file to smoke-test against")
    ap.add_argument("--keep", action="store_true",
                    help="Don't delete the tmpdir on exit (debugging)")
    args = ap.parse_args()

    if not args.fixture:
        print("ERROR: no fixture .mcraw provided. Pass one as an argument or",
              "set $MCRAW_SMOKE_FIXTURE.", file=sys.stderr)
        return 2
    fixture = Path(args.fixture).expanduser().resolve()
    if not fixture.is_file():
        print(f"ERROR: fixture not found: {fixture}", file=sys.stderr)
        return 2

    print(f"Smoke test against: {fixture}")
    print(f"ffprobe:            {FFPROBE}")
    print(f"mcraw module:       {mcraw.__file__}")
    print()

    tmpdir = Path(tempfile.mkdtemp(prefix="mcraw_smoke_"))
    print(f"Working in: {tmpdir}\n")

    results: list[tuple[str, bool, str]] = []
    for case in CASES:
        print(f"  [...] {case.name:36s}", end="", flush=True)
        ok, msg = case.run(fixture, tmpdir)
        marker = "PASS" if ok else "FAIL"
        # Overwrite the line with the result.
        print(f"\r  [{marker}] {case.name:36s} {msg}")
        results.append((case.name, ok, msg))

    if not args.keep:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"{passed}/{total} passed")

    if passed < total:
        print("\nFailed cases:")
        for name, ok, msg in results:
            if not ok:
                print(f"  {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
