"""MCRAW Studio — drag-and-drop bulk MCRAW transcoder.

Made by ACES2065-1. Apache 2.0.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def _setup_paths() -> None:
    """Ensure mcraw.pyd and its native deps resolve.

    Frozen (PyInstaller) bundles: vcpkg DLLs are extracted to
    `_MEIPASS\\ffmpeg_runtime\\` (see motioncam_tools.spec). Register that
    subdirectory with os.add_dll_directory() so the Windows loader resolves
    mcraw.pyd's implicit imports (avcodec / swscale / x265 / OpenColorIO /
    etc.) from our bundled copy. Because Python's import machinery calls
    LoadLibraryExW with LOAD_LIBRARY_SEARCH_USER_DIRS, our subdirectory is
    consulted BEFORE the user's PATH — so a separately installed ffmpeg in
    PATH can no longer shadow ours and trigger ABI-mismatch failures
    (ENOMEM at avcodec_open2, "external library error" on send_frame,
    etc.). Achieves the same protection as SetDefaultDllDirectories but
    without importing ctypes (which collides with vcpkg's ffi-8.dll in
    the bundle).

    Dev-from-source: load mcraw.pyd from `build/` and pull vcpkg DLLs
    from the local install. Unchanged behaviour.
    """
    if getattr(sys, "frozen", False):
        if hasattr(os, "add_dll_directory"):
            runtime = Path(sys._MEIPASS) / "ffmpeg_runtime"
            if runtime.is_dir():
                os.add_dll_directory(str(runtime))
        return
    here = Path(__file__).resolve().parent
    project_root = here.parent
    build_dir = project_root / "build"
    vcpkg_bin = Path(r"C:\dev\vcpkg\installed\x64-windows\bin")
    if vcpkg_bin.is_dir():
        os.add_dll_directory(str(vcpkg_bin))
    if build_dir.is_dir() and str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))


_setup_paths()

try:
    import mcraw  # noqa: E402
except Exception as exc:  # pragma: no cover
    print(f"Failed to import mcraw: {exc}", file=sys.stderr)
    raise

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

# The real-time player lives in its own module. Import AFTER mcraw — player.py
# imports mcraw directly and relies on _setup_paths() having already run.
from player import (  # noqa: E402
    HAVE_GL,
    AudioPlayer,
    FullscreenPreview,
    GLPreview,
    PlaybackWorker,
    ScopePanel,
    transport_icon,
)


def _safe_error(exc: BaseException) -> str:
    """Format an exception for the log without leaking source structure.
    We drop the Python traceback (file names, line numbers, function names)
    because the .exe ships closed-source and those frames would expose the
    module layout. The exception class + message alone is enough to diagnose
    every error path the app raises — they're either C++ runtime_error
    messages we already write clearly, or Python ones we wrote ourselves."""
    return f"{type(exc).__name__}: {exc}"


def _safe_unhandled(exc_type, exc_value, _exc_tb) -> str:
    """Sanitised formatter for the (type, value, tb) triple that
    sys.excepthook / threading.excepthook hand us. Discards `_exc_tb`."""
    name = getattr(exc_type, "__name__", "Exception")
    return f"{name}: {exc_value}"


# ----- Diagnostic host probes -------------------------------------------------
# All probes deliberately avoid `import ctypes`. Python's ctypes pulls
# `_ctypes.pyd` and `libffi-8.dll` into the PyInstaller bundle, which
# collides at runtime with vcpkg's `ffi-8.dll` (also bundled by the spec's
# vcpkg/bin glob) and segfaults the process before any logging starts.
# PowerShell + subprocess gets us the same diagnostic data without that
# bundling side-effect.

def _run_subprocess_quiet(args, timeout):
    try:
        import subprocess
        flags = 0
        if sys.platform == "win32":
            flags = 0x08000000  # CREATE_NO_WINDOW
        return subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            creationflags=flags,
        )
    except Exception:
        return None


def _ram_status_gb():
    """(total_gb, available_gb, percent_used) via PowerShell. Never raises."""
    if sys.platform != "win32":
        return None
    ps = (
        "$os = Get-CimInstance Win32_OperatingSystem;"
        " '{0}|{1}' -f $os.TotalVisibleMemorySize, $os.FreePhysicalMemory"
    )
    cp = _run_subprocess_quiet(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        timeout=8.0,
    )
    if cp is None or cp.returncode != 0:
        return None
    try:
        parts = cp.stdout.strip().split("|")
        if len(parts) != 2:
            return None
        total_kb = int(parts[0])
        free_kb = int(parts[1])
        total_gb = total_kb / (1024 * 1024)
        avail_gb = free_kb / (1024 * 1024)
        pct = int(round((1.0 - free_kb / total_kb) * 100)) if total_kb else 0
        return (total_gb, avail_gb, pct)
    except Exception:
        return None


def _log_gpu_info():
    """Enumerate GPUs. nvidia-smi first (accurate VRAM); WMI fallback."""
    if sys.platform != "win32":
        log.info("GPU enumeration: skipped (non-Windows host)")
        return
    import shutil
    logged_any = False
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        cp = _run_subprocess_quiet(
            [nvsmi,
             "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            timeout=5.0,
        )
        if cp is not None and cp.returncode == 0:
            for line in cp.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    name, vram_mb, free_mb, drv = parts[:4]
                    log.info(
                        "GPU [NVIDIA]: %s - VRAM %s MB (%s MB free), driver %s",
                        name, vram_mb, free_mb, drv,
                    )
                    logged_any = True
    ps_cmd = (
        "Get-CimInstance Win32_VideoController | "
        "ForEach-Object { '{0}|{1}|{2}' -f $_.Name, $_.AdapterRAM, $_.DriverVersion }"
    )
    cp = _run_subprocess_quiet(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
        timeout=10.0,
    )
    if cp is not None and cp.returncode == 0:
        for line in cp.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 3 or not parts[0].strip():
                continue
            name = parts[0].strip()
            ram_raw = parts[1].strip()
            drv = parts[2].strip() or "unknown"
            try:
                ram_bytes = int(ram_raw) if ram_raw else 0
            except ValueError:
                ram_bytes = 0
            ram_mb = ram_bytes // (1024 * 1024)
            if ram_mb <= 0:
                vram_str = "unknown"
            elif ram_mb >= 4090:
                vram_str = f"~{ram_mb} MB (WMI uint32 cap)"
            else:
                vram_str = f"{ram_mb} MB"
            log.info("GPU [WMI]: %s - VRAM %s, driver %s", name, vram_str, drv)
            logged_any = True
    if not logged_any:
        log.info("GPU: no adapters enumerated")


def _log_host_info():
    """Log CPU + RAM + GPU at startup. Cheap (~1-2s for PowerShell calls)."""
    try:
        cpu_name = (platform.processor() or "").strip() or "unknown"
        log.info("CPU: %s (%d logical cores)", cpu_name, os.cpu_count() or 0)
    except Exception as exc:
        log.info("CPU probe failed: %s", _safe_error(exc))
    ram = _ram_status_gb()
    if ram is not None:
        log.info("RAM: %.1f GB total, %.1f GB available (%d%% used)",
                 ram[0], ram[1], ram[2])
    else:
        log.info("RAM: probe unavailable")
    _log_gpu_info()


def _probe_input_resolution(path):
    """Return 'WxH' for an .mcraw input, or a short reason on failure.
    Tries container_metadata first (free); falls back to decoding one
    frame (~200ms on a 4K MCRAW). Never raises."""
    try:
        d = mcraw.Decoder(path)
        meta = d.container_metadata or {}
        w = meta.get("width") if isinstance(meta, dict) else None
        h = meta.get("height") if isinstance(meta, dict) else None
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            return f"{w}x{h}"
        ts = d.frames
        if ts:
            _, fh, fw = d.process_frame_rgb24(ts[0], "srgb")
            return f"{fw}x{fh}"
        return "no-frames"
    except Exception as exc:
        return f"probe-failed:{type(exc).__name__}"


# ----- Color space lookup ----------------------------------------------------
# Maps a (primaries label, transfer label) pair to the mcraw color-space
# short_name. The list is curated — only combinations that map to a real
# OCIO color space in studio-config-v4.0.0_aces-v2.0_ocio-v2.5 are listed.
# Anything not in the table is shown grayed-out with "(not available)".
PRIMARIES_LIST = ("BT.709", "sRGB", "Rec.2020", "DaVinci Wide Gamut",
                  "ACEScg (AP1)", "ACES AP0")

TRANSFER_LIST = ("Linear", "BT.709 / Gamma 2.4", "sRGB",
                 "ST2084 PQ", "HLG", "Cineon Log",
                 "DaVinci Intermediate", "ACES CCT")

# (primaries, transfer) -> mcraw color-space short_name.
CS_LOOKUP: dict[tuple[str, str], str] = {
    ("BT.709", "Linear"):                              "rec709",
    ("BT.709", "BT.709 / Gamma 2.4"):                  "rec709-display",
    ("BT.709", "sRGB"):                                "srgb",
    # sRGB primaries are bit-identical to BT.709; aliases for UI clarity.
    ("sRGB", "Linear"):                                "rec709",
    ("sRGB", "BT.709 / Gamma 2.4"):                    "rec709-display",
    ("sRGB", "sRGB"):                                  "srgb",
    ("Rec.2020", "Linear"):                            "rec2020-linear",
    ("Rec.2020", "ST2084 PQ"):                         "rec2020-pq",
    ("Rec.2020", "HLG"):                               "rec2020-hlg",
    ("DaVinci Wide Gamut", "Linear"):                  "davinci-wg-linear",
    ("DaVinci Wide Gamut", "DaVinci Intermediate"):    "davinci-intermediate",
    ("ACEScg (AP1)", "Linear"):                        "acescg",
    ("ACEScg (AP1)", "ACES CCT"):                      "acescct",
    ("ACES AP0", "Linear"):                            "aces2065-1",
    ("ACES AP0", "Cineon Log"):                        "adx10",
}

# Per-primaries valid transfers, in display order.
VALID_TRANSFERS: dict[str, list[str]] = {
    "BT.709":             ["Linear", "BT.709 / Gamma 2.4", "sRGB"],
    "sRGB":               ["Linear", "BT.709 / Gamma 2.4", "sRGB"],
    "Rec.2020":           ["Linear", "ST2084 PQ", "HLG"],
    "DaVinci Wide Gamut": ["Linear", "DaVinci Intermediate"],
    "ACEScg (AP1)":       ["Linear", "ACES CCT"],
    "ACES AP0":           ["Linear", "Cineon Log"],
}

# HDR transfers — when selected, the codec dropdown filters to 10-bit-capable codecs.
HDR_TRANSFERS = {"ST2084 PQ", "HLG"}


# ----- Background render worker -----------------------------------------------

class RenderWorker(QtCore.QObject):
    file_started = QtCore.Signal(int, str)
    file_progress = QtCore.Signal(int, int, int)  # row, current_frame, total_frames
    file_done = QtCore.Signal(int, bool, str)
    all_done = QtCore.Signal()

    def __init__(self, jobs: list[tuple[int, str, str, int, int]],
                 settings: dict, n_workers: int = 1):
        # Each job: (row, input_path, output_path, start_frame, end_frame).
        # n_workers > 1 spawns concurrent encoders; useful for CPU-bound codecs
        # (ProRes / DNxHR / libx265) where a single encode doesn't saturate the
        # machine. NVENC users should leave this at 1 (the GPU encoder is
        # already fast and parallel sessions can fight each other).
        super().__init__()
        self.jobs = list(jobs)
        self.settings = settings
        self.n_workers = max(1, min(8, int(n_workers)))
        self._lock = threading.Lock()
        self._next = 0
        # Pause / cancel events. _pause_event is "set" when NOT paused (so
        # workers don't block); clearing it pauses them. _cancel_event is
        # set to permanently abort.
        self._pause_event = threading.Event(); self._pause_event.set()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Permanently cancel the batch. C++ render loops will see this on
        the next frame boundary and return cleanly. Wakes any paused workers
        so they can finish promptly."""
        self._cancel_event.set()
        self._pause_event.set()

    def pause(self) -> None:
        """Pause all in-flight renders at the next frame boundary."""
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume from a paused state."""
        self._pause_event.set()

    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def _check_pause_cancel(self) -> bool:
        """Called from C++ render loops every frame.
        Blocks while paused; returns True to abort."""
        # Block here while paused. Wakes when the user hits Resume or Cancel.
        self._pause_event.wait()
        return self._cancel_event.is_set()

    @staticmethod
    def _cleanup_partial_output(out_path: str, fmt: str) -> None:
        """Delete a partial render output after a cancel.

        - For container files (.mov, .mp4, .mcraw) the partial bytes are
          incomplete and unplayable, so we delete the file.
        - For EXR sequences (a directory) we leave whatever frames already
          got written — each .exr file is independently valid, the user can
          re-render and skip what's done.
        """
        try:
            p = Path(out_path)
            if fmt == "exr":
                # Leave the directory alone — partial EXR sequences are useful.
                return
            if p.is_file():
                p.unlink(missing_ok=True)
                log.info("Removed partial output: %s", p)
        except Exception as exc:
            log.warning("Could not remove partial output %s: %s", out_path, exc)

    @property
    def _cancel(self) -> bool:
        # Backwards-compat shim for existing callers that read `._cancel`.
        return self._cancel_event.is_set()

    def _take_next_job(self):
        with self._lock:
            if self._next >= len(self.jobs):
                return None
            job = self.jobs[self._next]
            self._next += 1
            return job

    def _run_one(self, job: tuple[int, str, str, int, int]) -> None:
        row, inp, outp, start, end = job
        if self._cancel:
            self.file_done.emit(row, False, "cancelled")
            return
        self.file_started.emit(row, inp)

        def progress(current: int, total: int, _row: int = row) -> None:
            self.file_progress.emit(_row, current, total)

        # Log a sanitised view of the settings (no callable progress arg).
        safe_settings = {k: v for k, v in self.settings.items() if k != "progress"}
        res_str = _probe_input_resolution(inp)
        ram = _ram_status_gb()
        ram_str = (f"{ram[1]:.1f}/{ram[0]:.1f} GB free"
                   if ram is not None else "n/a")
        log.info("Render START  in=%s  out=%s  range=[%d..%d]  res=%s  ram=%s  settings=%s",
                 inp, outp, start, end, res_str, ram_str, safe_settings)
        t_start = time.time()

        # MCRAW trim: bypass the color/encode pipeline entirely.
        if self.settings.get("format") == "mcraw":
            try:
                mcraw.trim_mcraw(input=inp, output=outp,
                                 start=start, end=end, progress=progress,
                                 cancel=self._check_pause_cancel)
                if self._cancel_event.is_set():
                    self._cleanup_partial_output(outp, "mcraw")
                    log.info("Render CANCELLED  trim of %s", Path(inp).name)
                    self.file_done.emit(row, False, "cancelled")
                else:
                    log.info("Render OK     trim of %s -> %s  took=%.2fs",
                             Path(inp).name, Path(outp).name, time.time() - t_start)
                    self.file_done.emit(row, True, "done")
            except Exception as exc:
                log.error("Render FAILED  trim of %s  %s",
                          inp, _safe_error(exc))
                self.file_done.emit(row, False, f"error: {exc}")
            return

        try:
            kwargs = dict(
                input=inp,
                output=outp,
                colorspace=self.settings["colorspace"],
                bitrate=self.settings.get("bitrate", 80),
                progress=progress,
                start=start,
                end=end,
            )
            codec = self.settings.get("codec")
            if codec:
                kwargs["codec"] = codec
            exr_comp = self.settings.get("exr_compression")
            if exr_comp:
                kwargs["exr_compression"] = exr_comp
            if "denoise_chroma" in self.settings:
                kwargs["denoise_chroma"] = self.settings["denoise_chroma"]
            if "denoise_luma" in self.settings:
                kwargs["denoise_luma"] = self.settings["denoise_luma"]
            if self.settings.get("ten_bit"):
                kwargs["ten_bit"] = True
            if self.settings.get("highlight_recovery"):
                kwargs["highlight_recovery"] = True
            if self.settings.get("frame_rate_conversion", 0.0) > 0.0:
                kwargs["frame_rate_conversion"] = float(self.settings["frame_rate_conversion"])
            if not self.settings.get("bake_vignette", True):
                kwargs["bake_vignette"] = False
            kwargs["cancel"] = self._check_pause_cancel
            mcraw.render(**kwargs)
            if self._cancel_event.is_set():
                self._cleanup_partial_output(outp, self.settings.get("format", ""))
                log.info("Render CANCELLED  %s", Path(inp).name)
                self.file_done.emit(row, False, "cancelled")
            else:
                log.info("Render OK     %s -> %s  took=%.2fs",
                         Path(inp).name, Path(outp).name, time.time() - t_start)
                self.file_done.emit(row, True, "done")
        except Exception as exc:
            log.error("Render FAILED  %s  %s", inp, _safe_error(exc))
            self.file_done.emit(row, False, f"error: {exc}")

    def _worker_loop(self) -> None:
        while not self._cancel:
            job = self._take_next_job()
            if job is None:
                return
            self._run_one(job)

    @QtCore.Slot()
    def run(self) -> None:
        if self.n_workers <= 1:
            # Sequential — same as the original behaviour.
            for job in self.jobs:
                if self._cancel:
                    row = job[0]
                    self.file_done.emit(row, False, "cancelled")
                    continue
                self._run_one(job)
        else:
            threads = [
                threading.Thread(target=self._worker_loop, daemon=True,
                                 name=f"RenderWorker-{i}")
                for i in range(self.n_workers)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # If any jobs got skipped due to cancel, mark them so the UI is consistent.
            with self._lock:
                while self._next < len(self.jobs):
                    row = self.jobs[self._next][0]
                    self._next += 1
                    self.file_done.emit(row, False, "cancelled")
        self.all_done.emit()


# ----- Thumbnail decoder (background) ----------------------------------------

class ThumbWorker(QtCore.QObject):
    """Decodes a single frame to an sRGB QImage on a Python thread.

    Signals don't care which thread emits them — Qt queues across the main
    thread automatically, so the decode runs detached from the GUI loop.
    Always decodes through the scopes binding (the histogram pass is cheap)
    so the scopes panel tracks the scrub position, not just playback.
    """
    # path, frame_idx, image, hist_bytes (3*256 LE uint32), clip_lo, clip_hi
    done = QtCore.Signal(str, int, QtGui.QImage, bytes, float, float)

    def submit(self, path: str, frame_idx: int, bake_vignette: bool = True) -> None:
        threading.Thread(
            target=self._work, args=(path, frame_idx, bake_vignette), daemon=True
        ).start()

    _PREVIEW_MAX_W = 1920
    _PREVIEW_MAX_H = 1080

    def _work(self, path: str, frame_idx: int, bake_vignette: bool = True) -> None:
        try:
            d = mcraw.Decoder(path)
            timestamps = d.frames
            if frame_idx < 0 or frame_idx >= len(timestamps):
                return
            # numpy-free path: C++ side returns RGB888 packed bytes directly.
            buf, h, w, hist, clip_lo, clip_hi = d.process_frame_rgb24_scopes(
                timestamps[frame_idx], "srgb", False, bake_vignette)
            full = QtGui.QImage(buf, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            if w > self._PREVIEW_MAX_W or h > self._PREVIEW_MAX_H:
                img = full.scaled(
                    self._PREVIEW_MAX_W, self._PREVIEW_MAX_H,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            else:
                img = full.copy()
            self.done.emit(path, frame_idx, img, hist, clip_lo, clip_hi)
        except Exception as exc:
            print(f"[thumb] decode failed for {Path(path).name} frame {frame_idx}: {exc}")


# ----- File list with drag-and-drop ------------------------------------------

class DropList(QtWidgets.QListWidget):
    files_added = QtCore.Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self._placeholder = (
            "Drop .mcraw files here\n\n"
            "(or click 'Add Files...' below)"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if self.count() == 0:
            painter = QtGui.QPainter(self.viewport())
            painter.setPen(QtGui.QColor(140, 140, 140))
            font = painter.font()
            font.setPointSize(13)
            painter.setFont(font)
            painter.drawText(self.viewport().rect(),
                             QtCore.Qt.AlignCenter, self._placeholder)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths: list[str] = []
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_file() and p.suffix.lower() == ".mcraw":
                paths.append(str(p))
        if paths:
            self.files_added.emit(paths)
            event.acceptProposedAction()


# ----- Preferences ------------------------------------------------------------

def _system_ram_bytes() -> int:
    """Total physical RAM in bytes (Windows GlobalMemoryStatusEx), or 0 if it
    can't be determined. Used to cap the preview-cache budget at a fraction of
    RAM so the user can't starve the OS."""
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys)
    except Exception:
        pass
    return 0


class _PreferencesDialog(QtWidgets.QDialog):
    """App-wide preferences — the set-once-and-forget options kept out of the
    main window so it stays uncluttered. Organised into group boxes so new
    rarely-touched settings slot in without crowding anything. Opened from the
    Settings menu; persisted to config.json by the owner."""

    def __init__(self, owner: QtWidgets.QWidget, prefs: dict) -> None:
        super().__init__(owner)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.setMinimumWidth(440)
        self._ram = _system_ram_bytes()

        root = QtWidgets.QVBoxLayout(self)

        # ---- Memory & Cache -------------------------------------------------
        grp = QtWidgets.QGroupBox("Memory && Cache")
        form = QtWidgets.QFormLayout(grp)

        self.cacheEnable = QtWidgets.QCheckBox("Enable live preview cache")
        self.cacheEnable.setChecked(bool(prefs.get("preview_cache_enabled", True)))
        self.cacheEnable.setToolTip(
            "Keep decoded preview frames in RAM so replays and loops play back at\n"
            "full rate. The first play fills the cache; later passes read from it.")
        form.addRow(self.cacheEnable)

        # Cap the ceiling at 75% of physical RAM (leave headroom for the OS and
        # the decode/encode working set). Fall back to 8 GB if RAM is unknown.
        self._max_gb = 8.0
        if self._ram > 0:
            self._max_gb = max(1.0, round(self._ram * 0.75 / (1024 ** 3), 2))
        self.cacheSize = QtWidgets.QDoubleSpinBox()
        self.cacheSize.setDecimals(2)
        self.cacheSize.setSingleStep(0.25)
        self.cacheSize.setSuffix(" GB")
        self.cacheSize.setRange(0.25, self._max_gb)
        self.cacheSize.setValue(min(self._max_gb,
                                    float(prefs.get("preview_cache_gb", 1.2))))
        self.cacheSize.setToolTip(
            "Maximum RAM the preview cache may use. When it fills, the oldest\n"
            "frames are dropped first (least-recently-used).")
        form.addRow("Maximum cache size:", self.cacheSize)

        if self._ram > 0:
            hint = QtWidgets.QLabel(
                f"System RAM: {self._ram / (1024 ** 3):.1f} GB   ·   "
                f"capped at {self._max_gb:.1f} GB to leave headroom for the OS "
                f"and decoding.")
        else:
            hint = QtWidgets.QLabel(
                "System RAM unknown — choose a value your machine can spare.")
        hint.setStyleSheet("color:#9a9a9a;")
        hint.setWordWrap(True)
        form.addRow(hint)

        self.cacheEnable.toggled.connect(self.cacheSize.setEnabled)
        self.cacheSize.setEnabled(self.cacheEnable.isChecked())

        root.addWidget(grp)
        root.addStretch(1)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def values(self) -> dict:
        """The chosen preference values, merged over the existing dict by caller."""
        return {
            "preview_cache_enabled": bool(self.cacheEnable.isChecked()),
            "preview_cache_gb": float(self.cacheSize.value()),
        }


# ----- Main window -----------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MCRAW Studio")
        self.resize(1200, 760)
        self.setAcceptDrops(True)

        # App-wide preferences (auto-loaded; separate from the user-saved render
        # presets). Drives the preview-cache budget below and grows over time.
        self._prefs: dict = self._load_config()

        self._thread: QtCore.QThread | None = None
        self._worker: RenderWorker | None = None
        self._file_progress: dict[int, float] = {}
        self._total_jobs: int = 0

        # Per-file state keyed by absolute path.
        # state[path] = { 'total': int, 'start': int, 'end': int, 'scrub': int,
        #                 'thumb_img': QImage | None, 'thumb_frame': int | None }
        self._file_state: dict[str, dict] = {}

        # Debounce timer for scrubber → thumbnail decode.
        self._scrubTimer = QtCore.QTimer(self)
        self._scrubTimer.setSingleShot(True)
        self._scrubTimer.setInterval(180)
        self._scrubTimer.timeout.connect(self._kick_thumb_decode)

        # Background thumbnail decoder.
        self._thumb = ThumbWorker()
        self._thumb.done.connect(self._on_thumb_done)

        # Real-time playback engine + transport state.
        self._player = PlaybackWorker()
        self._player.frame.connect(self._on_play_frame)
        self._player.gl_frame.connect(self._on_gl_frame)
        self._player.scopes.connect(self._on_play_scopes)
        self._player.finished.connect(self._on_play_finished)
        self._scopes_on = bool(self._prefs.get("scopes_visible", False))
        # Zero-readback GL display: available when CUDA + Qt OpenGL are both
        # present. Fullscreen still uses the readback/QLabel path (v1).
        self._gl_ok = bool(HAVE_GL and GLPreview is not None
                           and mcraw.cuda_built() and mcraw.cuda_available())
        self._fs_window = None
        self._muted = False
        self._volume = 0.8
        self._playback_realtime = True
        # Push the persisted cache budget / enable into the worker's cache.
        self._apply_cache_prefs()

        # Preview audio (best-effort, auto-muted when playback isn't real-time).
        self._audio = AudioPlayer()
        self._audio.set_volume(self._volume)
        self._audio_cache: dict[str, tuple] = {}   # path -> (pcm, rate, channels, ts)
        self._audio_slice = b""                    # PCM for the current in/out range
        self._audio_rate = 0
        self._audio_channels = 0
        self._audio_ts: list = []
        self._play_start = 0
        # Hysteresis for the audio auto-mute so frame-time jitter doesn't stutter
        # the sink: start after a short real-time run, stop after a longer slow run.
        self._rt_run = 0
        self._slow_run = 0
        self._audio_should_play = False

        # Player keyboard shortcuts (window-scoped). Buttons use NoFocus so
        # Space reaches these instead of toggling a focused button.
        for keys, slot in (
            (QtGui.QKeySequence(QtCore.Qt.Key_Space), self._toggle_play),
            (QtGui.QKeySequence(QtCore.Qt.Key_F),     self._toggle_fullscreen),
            (QtGui.QKeySequence(QtCore.Qt.Key_I),     self._mark_in),
            (QtGui.QKeySequence(QtCore.Qt.Key_O),     self._mark_out),
        ):
            sc = QtGui.QShortcut(keys, self)
            sc.setContext(QtCore.Qt.WindowShortcut)
            sc.activated.connect(slot)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # Top split: file list (left)  ↔  preview pane (right)
        topSplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.dropList = DropList()
        self.dropList.files_added.connect(self._add_files)
        self.dropList.itemSelectionChanged.connect(self._on_selection_changed)
        topSplit.addWidget(self.dropList)

        # Preview pane
        previewBox = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(previewBox)
        pv.setContentsMargins(8, 4, 4, 4)

        # Preview surface stack: page 0 is the QLabel (thumbs, CPU playback,
        # readback GPU playback); page 1 is the zero-readback GL surface,
        # shown only while a GL-mode playback is running.
        self.previewLabel = QtWidgets.QLabel("(select a file to preview)")
        self.previewLabel.setObjectName("previewLabel")
        self.previewLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.previewLabel.setMinimumHeight(220)
        self.previewStack = QtWidgets.QStackedWidget()
        self.previewStack.addWidget(self.previewLabel)
        self.glPreview = None
        if self._gl_ok:
            try:
                self.glPreview = GLPreview()
                self.previewStack.addWidget(self.glPreview)
            except Exception as exc:
                log.warning("GL preview unavailable: %s", _safe_error(exc))
                self.glPreview = None
                self._gl_ok = False
        pv.addWidget(self.previewStack, 1)

        self.scrubSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrubSlider.setRange(0, 0)
        self.scrubSlider.setEnabled(False)
        self.scrubSlider.valueChanged.connect(self._on_scrub_changed)
        pv.addWidget(self.scrubSlider)

        # ----- Transport bar (player controls) -------------------------------
        # Cached icons (play/pause + volume swap on state changes).
        self._icon_play = transport_icon("play")
        self._icon_pause = transport_icon("pause")
        self._icon_vol_on = transport_icon("vol_on")
        self._icon_vol_off = transport_icon("vol_off")

        def _tbtn(kind: str, tip: str, slot) -> QtWidgets.QPushButton:
            b = QtWidgets.QPushButton()
            b.setIcon(transport_icon(kind))
            b.setIconSize(QtCore.QSize(18, 18))
            b.setToolTip(tip)
            b.setFixedSize(34, 28)
            b.setFlat(True)
            b.setEnabled(False)
            b.setFocusPolicy(QtCore.Qt.NoFocus)  # so Space/keys hit shortcuts, not buttons
            b.clicked.connect(slot)
            return b

        transport = QtWidgets.QHBoxLayout()
        transport.setSpacing(4)
        self.markInBtn = _tbtn("mark_in", "Set In point at the playhead (I)", self._mark_in)
        self.goStartBtn = _tbtn("skip_start", "Jump to In point", self._go_to_in)
        self.playBtn = _tbtn("play", "Play / Pause (Space)", self._toggle_play)
        self.goEndBtn = _tbtn("skip_end", "Jump to Out point", self._go_to_out)
        self.markOutBtn = _tbtn("mark_out", "Set Out point at the playhead (O)", self._mark_out)
        self.resetRangeBtn = _tbtn("reset", "Reset In/Out to the whole clip", self._reset_range)
        for b in (self.markInBtn, self.goStartBtn, self.playBtn,
                  self.goEndBtn, self.markOutBtn):
            transport.addWidget(b)
        transport.addSpacing(10)
        transport.addWidget(self.resetRangeBtn)

        self.frameLabel = QtWidgets.QLabel("Frame —")
        self.frameLabel.setObjectName("frameLabel")
        transport.addSpacing(12)
        transport.addWidget(self.frameLabel)
        transport.addStretch(1)

        # Live preview rate / proxy indicator (set during playback).
        self.playRateLabel = QtWidgets.QLabel("")
        self.playRateLabel.setObjectName("playRateLabel")
        transport.addWidget(self.playRateLabel)
        transport.addSpacing(10)

        # Preview quality / proxy: decode at 1/N resolution so weak machines can
        # keep playback near real-time (handy for spotting dropped frames in the
        # source). Pure preview — never affects renders.
        transport.addWidget(QtWidgets.QLabel("Quality:"))
        self.previewQualityCombo = QtWidgets.QComboBox()
        self.previewQualityCombo.setToolTip(
            "Preview decode resolution. Lower = faster playback on slow machines\n"
            "(decodes at 1/N, then scales to fit the preview). Does not affect renders.")
        for _qlabel, _qbin in (("Full", 1), ("1/2", 2), ("1/3", 3), ("1/4", 4)):
            self.previewQualityCombo.addItem(_qlabel, _qbin)
        _qi = self.previewQualityCombo.findData(int(self._prefs.get("preview_quality_bin", 1)))
        self.previewQualityCombo.setCurrentIndex(_qi if _qi >= 0 else 0)
        self.previewQualityCombo.currentIndexChanged.connect(self._on_preview_quality_changed)
        transport.addWidget(self.previewQualityCombo)
        transport.addSpacing(10)

        self.muteBtn = _tbtn("vol_on", "Mute / unmute preview audio", self._toggle_mute)
        self.muteBtn.setEnabled(True)
        transport.addWidget(self.muteBtn)
        self.volumeSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.volumeSlider.setRange(0, 100)
        self.volumeSlider.setValue(80)
        self.volumeSlider.setFixedWidth(90)
        self.volumeSlider.setToolTip("Preview volume")
        self.volumeSlider.valueChanged.connect(self._on_volume_changed)
        transport.addWidget(self.volumeSlider)
        transport.addSpacing(8)
        self.scopeBtn = _tbtn("scope", "Show / hide RGB scopes (histogram + clipping)",
                              self._toggle_scopes)
        self.scopeBtn.setEnabled(True)
        self.scopeBtn.setCheckable(True)
        self.scopeBtn.setChecked(self._scopes_on)
        transport.addWidget(self.scopeBtn)
        self.fullscreenBtn = _tbtn("fullscreen", "Fullscreen preview (F / Esc to exit)",
                                   self._toggle_fullscreen)
        transport.addWidget(self.fullscreenBtn)
        pv.addLayout(transport)

        # RGB scopes: histogram of the displayed image + pre-quantize clip
        # fractions. Hidden unless toggled; fed by playback and scrub decodes.
        self.scopePanel = ScopePanel()
        self.scopePanel.setVisible(self._scopes_on)
        self._player.set_scopes_enabled(self._scopes_on)
        pv.addWidget(self.scopePanel)

        topSplit.addWidget(previewBox)
        topSplit.setStretchFactor(0, 1)
        topSplit.setStretchFactor(1, 2)
        topSplit.setSizes([380, 760])
        outer.addWidget(topSplit, 1)

        fileBtns = QtWidgets.QHBoxLayout()
        addBtn = QtWidgets.QPushButton("Add Files...")
        addBtn.clicked.connect(self._browse_files)
        rmBtn = QtWidgets.QPushButton("Remove Selected")
        rmBtn.clicked.connect(self._remove_selected)
        clearBtn = QtWidgets.QPushButton("Clear All")
        clearBtn.clicked.connect(self._clear_all)
        fileBtns.addWidget(addBtn)
        fileBtns.addWidget(rmBtn)
        fileBtns.addWidget(clearBtn)
        fileBtns.addStretch(1)
        outer.addLayout(fileBtns)

        # ----- Output Settings (2-column grid layout) ------------------------
        # Each row pairs two related controls so the panel stays compact on
        # small laptop screens. Long help text moves into tooltips.
        settingsBox = QtWidgets.QGroupBox("Output Settings")
        grid = QtWidgets.QGridLayout(settingsBox)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        # Label columns stay tight; field columns expand evenly.
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 1)

        def _lbl(text: str) -> QtWidgets.QLabel:
            w = QtWidgets.QLabel(text)
            w.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            return w

        # ---- create the widgets ----
        self.formatCombo = QtWidgets.QComboBox()
        self.formatCombo.addItem("MOV    (for grading and editing)", "mov")
        self.formatCombo.addItem("MP4    (for delivery and normal viewing)", "mp4")
        self.formatCombo.addItem("EXR    (image sequence for VFX / compositing)", "exr")
        self.formatCombo.addItem("MCRAW    (trim a section, keep the RAW for later)", "mcraw")
        self.formatCombo.currentIndexChanged.connect(self._format_changed)

        self.codecCombo = QtWidgets.QComboBox()
        # Probe NVENC encoders once; hide the GPU options if the machine can't run them.
        # The probe is `encoder_available()` which now does an actual avcodec_open2,
        # so it correctly returns False on machines without an NVIDIA GPU/driver.
        probe = (lambda n: mcraw.encoder_available(n)) if hasattr(mcraw, "encoder_available") else (lambda n: True)
        all_codecs = []
        for c in mcraw.codecs():
            if c == "h264_nvenc" and not probe("h264_nvenc"): continue
            if c == "h265_nvenc" and not probe("hevc_nvenc"): continue
            if c == "av1_nvenc"  and not probe("av1_nvenc"):  continue
            all_codecs.append(c)
        self._all_codecs = all_codecs
        self._mp4_codecs = [c for c in all_codecs if c in (
            "h264", "h265", "h264_nvenc", "h265_nvenc", "av1_nvenc"
        )]

        _codec_labels = {
            "prores422":    "ProRes 422",
            "prores422hq":  "ProRes 422 HQ",
            "prores4444":   "ProRes 4444",
            "prores4444xq": "ProRes 4444 XQ",
            "h264":         "H.264 (CPU)",
            "h265":         "H.265 (CPU)",
            "h264_nvenc":   "H.264 (GPU)",
            "h265_nvenc":   "H.265 (GPU)",
            "av1_nvenc":    "AV1 (GPU)",
            "dnxhr_hqx":    "DNxHR HQX (10-bit)",
            "dnxhr_444":    "DNxHR 444 (12-bit)",
            "cineform":     "GoPro CineForm (10-bit, visually lossless)",
        }
        for c in self._all_codecs:
            self.codecCombo.addItem(_codec_labels.get(c, c), c)
        idx = self.codecCombo.findData("prores4444")
        if idx >= 0:
            self.codecCombo.setCurrentIndex(idx)
        self.codecCombo.currentIndexChanged.connect(self._update_ten_bit_state)

        self.primariesCombo = QtWidgets.QComboBox()
        for label in ("BT.709", "sRGB", "Rec.2020", "DaVinci Wide Gamut",
                      "ACEScg (AP1)", "ACES AP0"):
            self.primariesCombo.addItem(label, label)
        self.primariesCombo.currentIndexChanged.connect(self._primaries_changed)

        self.transferCombo = QtWidgets.QComboBox()
        self.transferCombo.currentIndexChanged.connect(self._transfer_changed)
        self._populate_transfer_combo()

        self.bitrateSpin = QtWidgets.QSpinBox()
        self.bitrateSpin.setRange(1, 500)
        self.bitrateSpin.setValue(80)
        self.bitrateSpin.setSuffix(" Mbps")
        self.bitrateSpin.setToolTip(
            "Target bitrate for H.264 / H.265 / AV1.\n"
            "Has no effect on ProRes / DNxHR / CineForm (their bitrate is\n"
            "set by the chosen profile)."
        )

        self.tenBitCheck = QtWidgets.QCheckBox("Encode 10-bit")
        self.tenBitCheck.setToolTip(
            "Higher precision — smoother gradients, more room for grading.\n\n"
            "Available for H.265 and AV1. ProRes, DNxHR, and CineForm are\n"
            "already high-precision so this option doesn't apply to them."
        )

        self.highlightRecoveryCheck = QtWidgets.QCheckBox("Recover highlights")
        self.highlightRecoveryCheck.setToolTip(
            "Neutralises clipped highlights so suns and bright window areas render\n"
            "as bright neutral white instead of magenta blobs.\n\n"
            "Display-encoded outputs (sRGB / Rec.709 / PQ / HLG) additionally get\n"
            "a soft shoulder rolloff above 1.0 for a more cinematic falloff.\n"
            "Scene-referred outputs (EXR ACEScg / Linear) keep their full HDR\n"
            "brightness at the neutralised hue, so grading downstream is unaffected\n"
            "outside the clipped region."
        )

        # Frame-rate conversion: checkbox + target dropdown. Off = keep the
        # source rate; on = resample (duplicate/drop) to the chosen constant
        # rate, staying in A/V sync.
        self.frcCheck = QtWidgets.QCheckBox("Convert to")
        self.frcCheck.setToolTip(
            "Resample to a constant target frame rate, duplicating or dropping\n"
            "frames to keep audio in sync. Leave off to keep the source rate.\n"
            "Set this before importing into DaVinci / Premiere."
        )
        self.frcCombo = QtWidgets.QComboBox()
        for _lab, _val in (("23.976", 23.976), ("24", 24.0), ("25", 25.0),
                           ("29.97", 29.97), ("30", 30.0), ("48", 48.0),
                           ("50", 50.0), ("59.94", 59.94), ("60", 60.0)):
            self.frcCombo.addItem(f"{_lab} fps", _val)
        self.frcCombo.setCurrentIndex(self.frcCombo.findData(24.0))
        self.frcCombo.setEnabled(False)
        self.frcCheck.toggled.connect(self.frcCombo.setEnabled)
        self.frcCheck.toggled.connect(lambda _=None: self._refresh_clip_header())
        self.frcCombo.currentIndexChanged.connect(lambda _=None: self._refresh_clip_header())
        frcComposite = QtWidgets.QWidget()
        frcRow = QtWidgets.QHBoxLayout(frcComposite)
        frcRow.setContentsMargins(0, 0, 0, 0)
        frcRow.setSpacing(4)
        frcRow.addWidget(self.frcCheck)
        frcRow.addWidget(self.frcCombo, 1)

        # Vignette / lens-shading correction. On by default (matches the raw
        # pipeline); off keeps the natural optical falloff in the corners.
        self.vignetteCheck = QtWidgets.QCheckBox("Bake vignette correction")
        self.vignetteCheck.setChecked(True)
        self.vignetteCheck.setToolTip(
            "Apply the lens-shading gainmap to flatten the natural lens vignette\n"
            "and corner colour cast. Turn off to keep the natural optical falloff\n"
            "and avoid lifting noise in dark corners."
        )
        self.vignetteCheck.toggled.connect(self._on_vignette_toggled)

        self.denoiseChromaSpin = QtWidgets.QSpinBox()
        self.denoiseChromaSpin.setRange(0, 100)
        self.denoiseChromaSpin.setValue(0)
        self.denoiseChromaSpin.setToolTip(
            "Color noise reduction (0–100). Safe to push high — kills\n"
            "magenta/green chroma blobs without softening detail.\n"
            "Applies to MP4 output only."
        )

        self.denoiseLumaSpin = QtWidgets.QSpinBox()
        self.denoiseLumaSpin.setRange(0, 100)
        self.denoiseLumaSpin.setValue(0)
        self.denoiseLumaSpin.setToolTip(
            "Brightness-noise reduction (0–100). Higher values smooth grain\n"
            "but soften fine detail. Applies to MP4 output only."
        )

        self.exrCompCombo = QtWidgets.QComboBox()
        _exr_labels = {
            "piz":   "PIZ      (lossless, recommended for camera footage)",
            "zip":   "ZIP      (lossless, classic default)",
            "zips":  "ZIPS     (lossless)",
            "rle":   "RLE      (lossless, best for graphics / CG)",
            "dwab":  "DWAB    (lossy, ~5× smaller, very high quality)",
            "dwaa":  "DWAA    (lossy, smaller files than ZIP)",
            "b44a":  "B44A    (lossy, fixed quality)",
            "b44":   "B44      (lossy, fixed quality)",
            "pxr24": "PXR24  (lossy, slight precision drop)",
            "none":  "None     (uncompressed, biggest files)",
        }
        for c in mcraw.exr_compressions():
            self.exrCompCombo.addItem(_exr_labels.get(c, c), c)
        idx = self.exrCompCombo.findData("piz")
        if idx >= 0:
            self.exrCompCombo.setCurrentIndex(idx)

        self.concurrentSpin = QtWidgets.QSpinBox()
        self.concurrentSpin.setRange(1, 4)
        self.concurrentSpin.setValue(1)
        self.concurrentSpin.setToolTip(
            "How many clips to render at the same time.\n\n"
            "1 — safest, one clip at a time.\n"
            "2-3 — faster for big batches when using ProRes, DNxHR, CineForm,\n"
            "        or CPU H.264 / H.265 (the slow codecs).\n\n"
            "Leave at 1 if you're using GPU encoding (NVENC). Higher values\n"
            "use more memory."
        )

        # Output dir + browse — composite goes in one cell
        self.outputDirEdit = QtWidgets.QLineEdit()
        self.outputDirEdit.setPlaceholderText("(same directory as each input)")
        outBrowse = QtWidgets.QPushButton("Browse…")
        outBrowse.clicked.connect(self._browse_output)
        outComposite = QtWidgets.QWidget()
        outRow = QtWidgets.QHBoxLayout(outComposite)
        outRow.setContentsMargins(0, 0, 0, 0)
        outRow.setSpacing(4)
        outRow.addWidget(self.outputDirEdit, 1)
        outRow.addWidget(outBrowse)

        # Save / Load preset buttons — composite goes in one cell
        self.savePresetBtn = QtWidgets.QPushButton("Save…")
        self.savePresetBtn.setToolTip("Save the current Output Settings as a preset file.")
        self.savePresetBtn.clicked.connect(self._save_preset)
        self.loadPresetBtn = QtWidgets.QPushButton("Load…")
        self.loadPresetBtn.setToolTip("Load Output Settings from a preset file.")
        self.loadPresetBtn.clicked.connect(self._load_preset)
        presetComposite = QtWidgets.QWidget()
        presetRow = QtWidgets.QHBoxLayout(presetComposite)
        presetRow.setContentsMargins(0, 0, 0, 0)
        presetRow.setSpacing(4)
        presetRow.addWidget(self.savePresetBtn)
        presetRow.addWidget(self.loadPresetBtn)
        presetRow.addStretch(1)

        # ---- place into the 4-column grid ----
        r = 0
        grid.addWidget(_lbl("Format:"),     r, 0); grid.addWidget(self.formatCombo,    r, 1)
        grid.addWidget(_lbl("Codec:"),      r, 2); grid.addWidget(self.codecCombo,     r, 3); r += 1
        grid.addWidget(_lbl("Color space:"), r, 0); grid.addWidget(self.primariesCombo, r, 1)
        grid.addWidget(_lbl("Transfer:"),   r, 2); grid.addWidget(self.transferCombo,  r, 3); r += 1
        grid.addWidget(_lbl("Bitrate:"),    r, 0); grid.addWidget(self.bitrateSpin,    r, 1)
        grid.addWidget(_lbl("Bit depth:"),  r, 2); grid.addWidget(self.tenBitCheck,    r, 3); r += 1
        grid.addWidget(_lbl("Highlights:"), r, 0); grid.addWidget(self.highlightRecoveryCheck, r, 1, 1, 3); r += 1
        grid.addWidget(_lbl("Frame rate:"), r, 0); grid.addWidget(frcComposite, r, 1)
        grid.addWidget(_lbl("Vignette:"),   r, 2); grid.addWidget(self.vignetteCheck, r, 3); r += 1
        grid.addWidget(_lbl("Denoise color:"), r, 0); grid.addWidget(self.denoiseChromaSpin, r, 1)
        grid.addWidget(_lbl("Denoise luma:"),  r, 2); grid.addWidget(self.denoiseLumaSpin,   r, 3); r += 1
        grid.addWidget(_lbl("EXR compression:"), r, 0); grid.addWidget(self.exrCompCombo, r, 1)
        grid.addWidget(_lbl("Concurrent renders:"), r, 2); grid.addWidget(self.concurrentSpin, r, 3); r += 1
        grid.addWidget(_lbl("Output:"), r, 0); grid.addWidget(outComposite,  r, 1)
        grid.addWidget(_lbl("Preset:"), r, 2); grid.addWidget(presetComposite, r, 3); r += 1

        outer.addWidget(settingsBox)

        # Progress + actions
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        outer.addWidget(self.progress)

        actions = QtWidgets.QHBoxLayout()
        self.pauseBtn = QtWidgets.QPushButton("Pause")
        self.pauseBtn.setEnabled(False)
        self.pauseBtn.setToolTip(
            "Pause the running render at the next frame boundary.\n"
            "Click Resume to continue."
        )
        self.pauseBtn.clicked.connect(self._toggle_pause)
        self.cancelBtn = QtWidgets.QPushButton("Cancel")
        self.cancelBtn.setEnabled(False)
        self.cancelBtn.setToolTip(
            "Stop the current render. Partial output files are deleted."
        )
        self.cancelBtn.clicked.connect(self._cancel)
        self.renderBtn = QtWidgets.QPushButton("Render All")
        self.renderBtn.setObjectName("renderBtn")
        self.renderBtn.setMinimumWidth(140)
        self.renderBtn.clicked.connect(self._render_all)
        actions.addStretch(1)
        actions.addWidget(self.pauseBtn)
        actions.addWidget(self.cancelBtn)
        actions.addWidget(self.renderBtn)
        outer.addLayout(actions)

        self.statusBar().showMessage("Ready")
        self._build_gpu_indicator()
        self._format_changed()
        self._build_menus()

    def _build_gpu_indicator(self) -> None:
        """Permanent right-side status-bar widget showing whether NVENC
        renders will take the CUDA RGB->NV12 kernel path (Phase B).

        States:
          "GPU YUV: ON  (NVENC)" - MCRAW_GPU_YUV=1 + this binary has CUDA
                                   + a CUDA-capable GPU is visible.
          "GPU YUV: off"         - default; CPU sws_scale path for all
                                   codecs including NVENC.
          "GPU YUV: off (no GPU)" - env var set but no NVIDIA card found
                                   at runtime; CPU path used silently.
          "GPU YUV: off (no CUDA)" - this build wasn't compiled with CUDA
                                   support; CPU path always.

        Only affects NVENC encoders. ProRes / DNxHR / CineForm / x264 /
        x265 / EXR are unaffected by this flag — they're always CPU.
        """
        label = QtWidgets.QLabel()
        label.setObjectName("gpuIndicator")
        label.setStyleSheet(
            "QLabel#gpuIndicator { padding: 0 8px; color: #888; }"
        )

        try:
            built = bool(mcraw.cuda_built())
        except Exception:
            built = False
        try:
            available = bool(mcraw.cuda_available()) if built else False
        except Exception:
            available = False
        env_value = os.environ.get("MCRAW_GPU_YUV", "")
        env_set = env_value == "1"

        if not built:
            text = "GPU YUV: off (no CUDA in build)"
            tip = ("This build of MCRAW Studio wasn't compiled with CUDA "
                   "support. NVENC renders still work but always go through "
                   "the CPU YUV pipeline.")
        elif env_set and available:
            text = "GPU YUV: ON  (NVENC only)"
            label.setStyleSheet(
                "QLabel#gpuIndicator { padding: 0 8px; color: #6abe6a; }"
            )
            tip = ("MCRAW_GPU_YUV=1 detected and a CUDA-capable GPU is "
                   "visible. NVENC renders (h264_nvenc / h265_nvenc / "
                   "av1_nvenc) at 8-bit BT.709 / sRGB / Rec.709 will use "
                   "the GPU RGB->NV12 kernel. Rec.2020 PQ/HLG and 10-bit "
                   "NVENC still take the CPU YUV path.")
        elif env_set and not available:
            text = "GPU YUV: off (no NVIDIA GPU)"
            tip = ("MCRAW_GPU_YUV=1 is set, but no CUDA-capable GPU was "
                   "detected at startup. Falling back to the CPU YUV "
                   "pipeline. Install or update your NVIDIA driver if "
                   "you expected a GPU here.")
        else:
            text = "GPU YUV: off"
            tip = ("CPU YUV pipeline (sws_scale). To enable the GPU path "
                   "for NVENC renders, launch MCRAW Studio via the "
                   "'MCRAWStudio (GPU).bat' launcher, or set the "
                   "MCRAW_GPU_YUV=1 environment variable.")

        label.setText(text)
        label.setToolTip(tip)
        self.statusBar().addPermanentWidget(label)

    # ----- Menus -----
    def _build_menus(self) -> None:
        bar = self.menuBar()

        # Help submenu (left side, conventional position).
        helpMenu = bar.addMenu("&Help")
        aboutAction = helpMenu.addAction("&About MCRAW Studio…")
        aboutAction.triggered.connect(self._show_about)
        helpMenu.addSeparator()
        viewLogAction = helpMenu.addAction("&View log file…")
        viewLogAction.setToolTip(
            "Open the diagnostic log. Send this file along with bug reports."
        )
        viewLogAction.triggered.connect(self._open_log_file)
        openLogDirAction = helpMenu.addAction("Open log &folder…")
        openLogDirAction.triggered.connect(self._open_log_folder)

        # Settings — home for the set-once-and-forget options (memory/cache and,
        # over time, other rarely-touched preferences) so the main window stays
        # uncluttered. Mirrors After Effects' Edit > Preferences pattern.
        settingsMenu = bar.addMenu("&Settings")
        prefsAction = settingsMenu.addAction("&Preferences…")
        prefsAction.setToolTip("Memory & cache and other app options.")
        prefsAction.triggered.connect(self._open_preferences)

        # Top-level entries — visible directly in the menu bar without a submenu.
        # These are intentionally surfaced because Donate / Report-a-problem are
        # the actions users actually need quick access to.
        reportAction = bar.addAction("&Report a problem…")
        reportAction.setToolTip(
            "Send a bug report or contact the developer."
        )
        reportAction.triggered.connect(lambda: self._open_url(REPORT_URL))

        donateAction = bar.addAction("♥ &Donate")
        donateAction.setToolTip(
            "Support development. Opens the donation page in your browser."
        )
        donateAction.triggered.connect(lambda: self._open_url(DONATE_URL))

    def _open_url(self, url: str) -> None:
        """Open a URL in the user's default browser. If the URL still has the
        REPLACE placeholder, surface a friendly dialog so the dev knows."""
        if "REPLACE-WITH-YOUR" in url:
            QtWidgets.QMessageBox.information(
                self,
                "Link not configured",
                "This link hasn't been set up yet. Edit DONATE_URL / REPORT_URL / "
                "PROJECT_URL near the top of motioncam_tools.py to point at your "
                "real pages, then rebundle.\n\nPlaceholder URL was:\n\n" + url,
            )
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

    @QtCore.Slot()
    def _open_log_file(self) -> None:
        """Open log.txt in the user's default text editor."""
        if not LOG_PATH.exists():
            QtWidgets.QMessageBox.information(
                self, "No log yet",
                f"The log file hasn't been written yet.\n\nIt will appear at:\n{LOG_PATH}"
            )
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(LOG_PATH)))

    @QtCore.Slot()
    def _open_log_folder(self) -> None:
        """Open the folder containing log.txt in Explorer."""
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(LOG_PATH.parent)))

    @QtCore.Slot()
    def _show_about(self) -> None:
        text = f"""\
<h3>MCRAW Studio</h3>
<p><b>Version:</b> {APP_VERSION} &nbsp;·&nbsp; <b>made by ACES2065-1</b></p>
<p><a href="{PROJECT_URL}">{PROJECT_URL}</a></p>
"""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("About MCRAW Studio")
        box.setTextFormat(QtCore.Qt.RichText)
        box.setText(text)
        # Make donate / GitHub URLs in the body clickable so users don't
        # have to manually copy-paste them out of the dialog.
        box.setTextInteractionFlags(
            QtCore.Qt.TextBrowserInteraction | QtCore.Qt.LinksAccessibleByMouse
        )
        # Use the main window's icon for the dialog too.
        if not self.windowIcon().isNull():
            box.setIconPixmap(self.windowIcon().pixmap(64, 64))
        box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        box.exec()

    # ----- Drag-drop on the whole window -----
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths = [str(Path(u.toLocalFile())) for u in event.mimeData().urls()
                 if Path(u.toLocalFile()).suffix.lower() == ".mcraw"]
        if paths:
            self._add_files(paths)
            event.acceptProposedAction()

    # ----- File list management -----
    def _add_files(self, paths: list[str]) -> None:
        existing = {self.dropList.item(i).data(QtCore.Qt.UserRole)
                    for i in range(self.dropList.count())}
        for p in paths:
            if p in existing:
                continue
            try:
                d = mcraw.Decoder(p)
                total = int(d.frame_count)
                ts = list(d.frames)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Failed to open", f"{Path(p).name}\n{exc}")
                continue
            # MotionCam timestamps are in nanoseconds (Android camera clock).
            # Use the MEDIAN inter-frame interval (robust to dropped frames),
            # matching the C++ DetectFrameRate — the mean is skewed by drops.
            fps = self._median_fps(ts)
            duration = (ts[-1] - ts[0]) / 1.0e9 if total >= 2 else 0.0
            self._file_state[p] = {
                "total": total,
                "ts": ts,
                "fps": fps,
                "duration": duration,
                "start": 0,
                "end": total,
                "scrub": 0,
                "thumb_img": None,
                "thumb_frame": None,
                "status": "queued",       # queued | rendering | done | failed | cancelled
                "render_started": None,    # time.time() float, set when worker picks it up
                "render_finished": None,
                "render_progress": 0,      # current frame within [start..end)
                "render_total_frames": 0,  # total frames the worker reports (for ETA)
                "out_path": None,
                "error_msg": None,
            }
            it = QtWidgets.QListWidgetItem()
            it.setData(QtCore.Qt.UserRole, p)
            self._apply_item_text(it, p)
            self.dropList.addItem(it)
        self.dropList.viewport().update()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60.0:
            return f"{seconds:.2f}s"
        m, s = divmod(seconds, 60.0)
        if m < 60.0:
            return f"{int(m)}:{s:05.2f}"
        h, m = divmod(m, 60.0)
        return f"{int(h)}:{int(m):02d}:{s:05.2f}"

    @staticmethod
    def _median_fps(ts: list) -> float:
        """Source fps from the median inter-frame interval (robust to drops)."""
        if len(ts) < 2:
            return 0.0
        gaps = sorted(ts[i + 1] - ts[i] for i in range(len(ts) - 1) if ts[i + 1] > ts[i])
        if not gaps:
            return 0.0
        med = gaps[len(gaps) // 2]
        return 1.0e9 / med if med > 0 else 0.0

    @staticmethod
    def _plan_counts(ts: list, start: int, end: int, target_fps: float):
        """Mirror of C++ BuildFramePlan: (out_frames, dropped, duplicated) for
        a nearest-timestamp resample to a constant target_fps. target<=0 = off."""
        n = len(ts)
        start = max(0, min(start, n))
        end = max(start, min(end, n))
        if target_fps <= 0.0 or end - start <= 0:
            return (end - start, 0, 0)
        t0 = ts[start]
        span = ts[end - 1] - t0
        if end - start == 1 or span <= 0:
            return (1, 0, 0)
        out_count = max(1, round(span / 1.0e9 * target_fps) + 1)
        cursor, last, dup = start, -1, 0
        used = set()
        for j in range(out_count):
            tj = t0 + round(j / target_fps * 1.0e9)
            while cursor + 1 < end and abs(ts[cursor + 1] - tj) <= abs(ts[cursor] - tj):
                cursor += 1
            if cursor == last:
                dup += 1
            used.add(cursor)
            last = cursor
        dropped = sum(1 for i in range(start, end) if i not in used)
        return (out_count, dropped, dup)

    def _frc_target(self) -> float:
        """Selected conversion target fps, or 0.0 when conversion is off."""
        if hasattr(self, "frcCheck") and self.frcCheck.isChecked():
            return float(self.frcCombo.currentData() or 0.0)
        return 0.0

    def _refresh_clip_header(self) -> None:
        """Re-apply every list item's text so the fps line reflects the current
        frame-rate-conversion setting."""
        if not hasattr(self, "dropList"):
            return
        for i in range(self.dropList.count()):
            it = self.dropList.item(i)
            p = it.data(QtCore.Qt.UserRole)
            if p:
                self._apply_item_text(it, p)

    @staticmethod
    def _format_fps(fps: float) -> str:
        if fps <= 0:
            return "—"
        # Show common fractional rates with their snapped form
        for target, label in ((23.976, "23.976"), (29.97, "29.97"), (59.94, "59.94")):
            if abs(fps - target) < 0.05:
                return label
        if abs(fps - round(fps)) < 0.01:
            return f"{int(round(fps))}"
        return f"{fps:.2f}"

    def _format_item_text(self, path: str, status: str | None = None) -> str:
        """Two-line item text: first line = name + state, second = clip stats."""
        st = self._file_state.get(path)
        name = Path(path).name
        if not st:
            return f"{name}  —  {status or 'queued'}"
        s = status if status is not None else st.get("status", "queued")
        # Status tag shows render progress when rendering, took-time when done.
        if s == "rendering" and st.get("render_total_frames"):
            tag = f"rendering {st['render_progress']}/{st['render_total_frames']}"
        elif s == "done" and st.get("render_started") and st.get("render_finished"):
            took = st["render_finished"] - st["render_started"]
            tag = f"done in {self._format_duration(took)}"
        elif s == "failed" and st.get("error_msg"):
            tag = f"failed: {st['error_msg'][:40]}"
        else:
            tag = s

        line1 = f"{name}  —  {tag}"
        # Second line: frame count, fps, duration, render range if non-default.
        target = self._frc_target()
        if target > 0.0 and st.get("ts"):
            out_n, dropped, dup = self._plan_counts(
                st["ts"], st["start"], st["end"], target)
            fps_bit = (f"{self._format_fps(st['fps'])} → {self._format_fps(target)} fps "
                       f"(−{dropped} / +{dup})")
            count_bit = f"{out_n} frames"
        else:
            fps_bit = f"{self._format_fps(st['fps'])} fps"
            count_bit = f"{st['total']} frames"
        bits = [count_bit, fps_bit, self._format_duration(st['duration'])]
        if st['start'] != 0 or st['end'] != st['total']:
            bits.append(f"range [{st['start']}..{st['end']}]")
        line2 = "  ·  ".join(bits)
        return f"{line1}\n{line2}"

    def _format_item_tooltip(self, path: str) -> str:
        st = self._file_state.get(path)
        if not st:
            return path
        lines = [
            Path(path).name,
            f"Path:        {path}",
            f"Frames:      {st['total']}",
            f"Frame rate:  {self._format_fps(st['fps'])} fps  ({st['fps']:.4f} measured)" if st['fps'] > 0 else "Frame rate:  unknown",
            f"Duration:    {self._format_duration(st['duration'])}",
            f"Range:       [{st['start']} .. {st['end']})  ({st['end'] - st['start']} frames to render)",
            f"State:       {st['status']}",
        ]
        if st.get("render_started"):
            started = datetime.fromtimestamp(st['render_started']).strftime("%H:%M:%S")
            if st.get("render_finished"):
                finished = datetime.fromtimestamp(st['render_finished']).strftime("%H:%M:%S")
                took = st['render_finished'] - st['render_started']
                lines.append(f"Started:     {started}")
                lines.append(f"Finished:    {finished}  (took {self._format_duration(took)})")
            else:
                elapsed = time.time() - st['render_started']
                lines.append(f"Started:     {started}  (running {self._format_duration(elapsed)})")
                if st.get("render_total_frames", 0) > 0 and st['render_progress'] > 0:
                    pct = 100.0 * st['render_progress'] / st['render_total_frames']
                    lines.append(f"Progress:    {st['render_progress']}/{st['render_total_frames']}  ({pct:.1f}%)")
        if st.get("out_path"):
            lines.append(f"Output:      {st['out_path']}")
        if st.get("error_msg"):
            lines.append(f"Error:       {st['error_msg']}")
        return "\n".join(lines)

    def _apply_item_text(self, it: QtWidgets.QListWidgetItem, path: str,
                         status: str | None = None) -> None:
        it.setText(self._format_item_text(path, status))
        it.setToolTip(self._format_item_tooltip(path))
        # Two-line items need a taller size hint or QListWidget clips them.
        fm = self.dropList.fontMetrics()
        it.setSizeHint(QtCore.QSize(0, fm.height() * 2 + 10))

    def _browse_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select MCRAW files", "", "MCRAW recordings (*.mcraw)")
        if files:
            self._add_files(files)

    def _remove_selected(self) -> None:
        for item in self.dropList.selectedItems():
            path = item.data(QtCore.Qt.UserRole)
            self._file_state.pop(path, None)
            self.dropList.takeItem(self.dropList.row(item))
        self._refresh_preview_for_current()
        self.dropList.viewport().update()

    def _clear_all(self) -> None:
        self.dropList.clear()
        self._file_state.clear()
        self._refresh_preview_for_current()

    def _browse_output(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory")
        if d:
            self.outputDirEdit.setText(d)

    # ----- Render-settings preset (save / load) -----
    def _collect_settings_dict(self) -> dict:
        """Gather every Output-Settings control into a JSON-able dict."""
        return {
            "version": 1,
            "format":             self.formatCombo.currentData(),
            "codec":              self.codecCombo.currentData(),
            "primaries":          self.primariesCombo.currentData(),
            "transfer":           self.transferCombo.currentData(),
            "bitrate":            int(self.bitrateSpin.value()),
            "ten_bit":            bool(self.tenBitCheck.isChecked()),
            "highlight_recovery": bool(self.highlightRecoveryCheck.isChecked()),
            "frc_enabled":        bool(self.frcCheck.isChecked()),
            "frc_target":         float(self.frcCombo.currentData() or 0.0),
            "bake_vignette":      bool(self.vignetteCheck.isChecked()),
            "denoise_chroma":     int(self.denoiseChromaSpin.value()),
            "denoise_luma":       int(self.denoiseLumaSpin.value()),
            "exr_compression":    self.exrCompCombo.currentData(),
            "concurrent_renders": int(self.concurrentSpin.value()),
            "output_dir":         self.outputDirEdit.text(),
        }

    def _apply_settings_dict(self, data: dict) -> None:
        """Restore every Output-Settings control from a saved preset dict.
        Missing or invalid entries are silently ignored so older / partial
        presets don't error out."""
        def set_combo(combo: QtWidgets.QComboBox, value):
            if value is None:
                return
            i = combo.findData(value)
            if i >= 0:
                combo.setCurrentIndex(i)

        # Apply in an order that respects inter-dependencies:
        # format change resets the codec list, primaries change resets transfers.
        set_combo(self.formatCombo, data.get("format"))
        set_combo(self.codecCombo,  data.get("codec"))
        set_combo(self.primariesCombo, data.get("primaries"))
        set_combo(self.transferCombo,  data.get("transfer"))
        if "bitrate" in data:
            try: self.bitrateSpin.setValue(int(data["bitrate"]))
            except Exception: pass
        if "ten_bit" in data:
            self.tenBitCheck.setChecked(bool(data["ten_bit"]))
        if "highlight_recovery" in data:
            self.highlightRecoveryCheck.setChecked(bool(data["highlight_recovery"]))
        if "frc_target" in data:
            i = self.frcCombo.findData(float(data["frc_target"]))
            if i >= 0:
                self.frcCombo.setCurrentIndex(i)
        if "frc_enabled" in data:
            self.frcCheck.setChecked(bool(data["frc_enabled"]))
        if "bake_vignette" in data:
            self.vignetteCheck.setChecked(bool(data["bake_vignette"]))
        if "denoise_chroma" in data:
            try: self.denoiseChromaSpin.setValue(int(data["denoise_chroma"]))
            except Exception: pass
        if "denoise_luma" in data:
            try: self.denoiseLumaSpin.setValue(int(data["denoise_luma"]))
            except Exception: pass
        set_combo(self.exrCompCombo, data.get("exr_compression"))
        if "concurrent_renders" in data:
            try: self.concurrentSpin.setValue(int(data["concurrent_renders"]))
            except Exception: pass
        if "output_dir" in data:
            self.outputDirEdit.setText(str(data.get("output_dir") or ""))

    def _preset_dir(self) -> str:
        """Default location for preset files. Created on first use."""
        d = Path(os.path.expanduser("~/Documents/MCRAWStudio/presets"))
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            d = Path(os.path.expanduser("~"))
        return str(d)

    @QtCore.Slot()
    def _save_preset(self) -> None:
        import json
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save render settings preset",
            self._preset_dir() + "/preset.json",
            "MCRAW Studio preset (*.json)"
        )
        if not fname:
            return
        if not fname.lower().endswith(".json"):
            fname += ".json"
        try:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(self._collect_settings_dict(), f, indent=2)
            self.statusBar().showMessage(f"Saved preset: {Path(fname).name}", 5000)
            log.info("Preset saved: %s", fname)
        except Exception as exc:
            log.error("Preset save failed for %s: %s", fname, exc)
            QtWidgets.QMessageBox.warning(self, "Save failed",
                                          f"Could not write preset:\n{exc}")

    @QtCore.Slot()
    def _load_preset(self) -> None:
        import json
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load render settings preset",
            self._preset_dir(),
            "MCRAW Studio preset (*.json)"
        )
        if not fname:
            return
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            log.error("Preset load failed for %s: %s", fname, exc)
            QtWidgets.QMessageBox.warning(self, "Load failed",
                                          f"Could not read preset:\n{exc}")
            return
        self._apply_settings_dict(data)
        self.statusBar().showMessage(f"Loaded preset: {Path(fname).name}", 5000)
        log.info("Preset loaded: %s", fname)

    # ----- App preferences (auto-persisted config.json) -----
    def _config_path(self) -> Path:
        """Location of the auto-loaded app config. Sits beside the presets
        folder under ~/Documents/MCRAWStudio; falls back to the home dir."""
        d = Path(os.path.expanduser("~/Documents/MCRAWStudio"))
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            d = Path(os.path.expanduser("~"))
        return d / "config.json"

    def _load_config(self) -> dict:
        """Read config.json at startup. Missing / unreadable -> empty dict (the
        callers all default each key), so a first run or a corrupt file is fine."""
        import json
        try:
            with open(self._config_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _save_config(self) -> None:
        import json
        try:
            with open(self._config_path(), "w", encoding="utf-8") as f:
                json.dump(self._prefs, f, indent=2)
        except Exception as exc:
            log.warning("Could not save config: %s", exc)

    def _apply_cache_prefs(self) -> None:
        """Push the persisted preview-cache settings into the playback worker."""
        enabled = bool(self._prefs.get("preview_cache_enabled", True))
        gb = float(self._prefs.get("preview_cache_gb", 1.2))
        self._player.set_cache_enabled(enabled)
        self._player.set_cache_budget(int(gb * (1024 ** 3)))

    @QtCore.Slot()
    def _open_preferences(self) -> None:
        dlg = _PreferencesDialog(self, self._prefs)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self._prefs.update(dlg.values())
            self._apply_cache_prefs()
            self._save_config()
            self.statusBar().showMessage("Preferences saved.", 4000)

    # ----- Preview pane (selection / scrub / mark in/out) -----
    def _current_path(self) -> str | None:
        items = self.dropList.selectedItems()
        if not items:
            return None
        return items[0].data(QtCore.Qt.UserRole)

    def _on_vignette_toggled(self, _checked: bool = False) -> None:
        # The cached preview was decoded with the previous vignette setting —
        # invalidate every clip's thumb so the current one re-decodes, and drop
        # the live playback cache (next play re-fills it with the new setting).
        for st in self._file_state.values():
            st["thumb_img"] = None
            st["thumb_frame"] = None
        self._player.clear_cache()
        self._refresh_preview_for_current()

    def _refresh_preview_for_current(self) -> None:
        path = self._current_path()
        st = self._file_state.get(path) if path else None
        if not st:
            self.previewLabel.setText("(select a file to preview)")
            self.previewLabel.setPixmap(QtGui.QPixmap())
            self.scrubSlider.setEnabled(False)
            self.scrubSlider.setRange(0, 0)
            self._set_transport_enabled(False)
            self.frameLabel.setText("Frame —")
            return

        total = st["total"]
        # Update slider
        self.scrubSlider.blockSignals(True)
        self.scrubSlider.setRange(0, max(0, total - 1))
        self.scrubSlider.setValue(min(st["scrub"], total - 1))
        self.scrubSlider.blockSignals(False)
        self.scrubSlider.setEnabled(total > 1)
        self._set_transport_enabled(total > 0)

        # Frame label: playhead position + In/Out range.
        self.frameLabel.setText(
            f"Frame {st['scrub'] + 1} / {total}    "
            f"(In {st['start']}  ·  Out {st['end']})"
        )

        # Show cached thumb if it matches; otherwise kick a decode
        if st["thumb_img"] is not None and st["thumb_frame"] == st["scrub"]:
            self._set_preview_pixmap(QtGui.QPixmap.fromImage(st["thumb_img"]))
        else:
            self.previewLabel.setText("Decoding preview…")
            self._scrubTimer.start()

    def _set_preview_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        if pixmap.isNull():
            self.previewLabel.setText("(no preview)")
            return
        self.previewLabel.setPixmap(pixmap.scaled(
            self.previewLabel.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        ))
        if getattr(self, "_fs_window", None) is not None:
            self._fs_window.set_image(pixmap)

    @QtCore.Slot()
    def _on_selection_changed(self) -> None:
        # Stop playback when moving to a different clip (worker is per-path).
        if self._player.is_playing():
            self._stop_playback()
        self._refresh_preview_for_current()

    @QtCore.Slot(int)
    def _on_scrub_changed(self, value: int) -> None:
        path = self._current_path()
        st = self._file_state.get(path) if path else None
        if not st:
            return
        st["scrub"] = value
        self.frameLabel.setText(
            f"Frame {value + 1} / {st['total']}    "
            f"(In {st['start']}  ·  Out {st['end']})"
        )
        # While playing, the worker drives frames — don't also kick a thumb decode.
        if not self._player.is_playing():
            self._scrubTimer.start()

    @QtCore.Slot()
    def _kick_thumb_decode(self) -> None:
        path = self._current_path()
        st = self._file_state.get(path) if path else None
        if not st:
            return
        bake = self.vignetteCheck.isChecked() if hasattr(self, "vignetteCheck") else True
        self._thumb.submit(path, int(st["scrub"]), bake)

    @QtCore.Slot(str, int, QtGui.QImage, bytes, float, float)
    def _on_thumb_done(self, path: str, frame_idx: int, img: QtGui.QImage,
                       hist: bytes, clip_lo: float, clip_hi: float) -> None:
        st = self._file_state.get(path)
        if not st:
            return
        st["thumb_img"] = img
        st["thumb_frame"] = frame_idx
        # Only repaint if this file is still the selected one and the user
        # hasn't moved the scrubber to a different frame in the meantime.
        if self._current_path() == path and st["scrub"] == frame_idx:
            self._set_preview_pixmap(QtGui.QPixmap.fromImage(img))
            if self._scopes_on and not self._player.is_playing():
                self.scopePanel.set_data(hist, clip_lo, clip_hi)

    @QtCore.Slot()
    def _mark_in(self) -> None:
        path = self._current_path()
        st = self._file_state.get(path) if path else None
        if not st:
            return
        st["start"] = int(st["scrub"])
        if st["end"] <= st["start"]:
            st["end"] = min(st["total"], st["start"] + 1)
        self._sync_item_text(path)
        self._refresh_preview_for_current()

    @QtCore.Slot()
    def _mark_out(self) -> None:
        path = self._current_path()
        st = self._file_state.get(path) if path else None
        if not st:
            return
        st["end"] = min(st["total"], int(st["scrub"]) + 1)
        if st["start"] >= st["end"]:
            st["start"] = max(0, st["end"] - 1)
        self._sync_item_text(path)
        self._refresh_preview_for_current()

    # ----- Transport: in/out jumps, play/pause, volume, fullscreen --------
    def _set_transport_enabled(self, on: bool) -> None:
        for b in (self.markInBtn, self.goStartBtn, self.playBtn,
                  self.goEndBtn, self.markOutBtn, self.resetRangeBtn,
                  self.fullscreenBtn):
            b.setEnabled(on)

    @QtCore.Slot()
    def _go_to_in(self) -> None:
        st = self._file_state.get(self._current_path() or "")
        if st:
            self.scrubSlider.setValue(int(st["start"]))

    @QtCore.Slot()
    def _go_to_out(self) -> None:
        st = self._file_state.get(self._current_path() or "")
        if st:
            self.scrubSlider.setValue(max(0, int(st["end"]) - 1))

    @QtCore.Slot()
    def _toggle_play(self) -> None:
        if self._player.is_playing():
            self._stop_playback()
            return
        path = self._current_path()
        st = self._file_state.get(path) if path else None
        if not st or st["total"] < 2:
            return
        # GPU when available (real-time at full preview size); CPU otherwise,
        # at a smaller proxy size so it has a chance of keeping up.
        gpu = bool(mcraw.cuda_available())
        max_dim = 1280 if gpu else 768
        in_idx = int(st["start"])           # In point — the loop-back target
        end = int(st["end"])                # Out point
        start = int(st["scrub"])            # begin from the playhead…
        if not (in_idx <= start < end - 1):
            start = in_idx                  # …unless it's outside the range / at the end
        preview_bin = int(self.previewQualityCombo.currentData() or 1)
        # Zero-readback GL surface: GPU decode straight to a GL texture, no
        # PCIe round trip, full-res preview. Fullscreen keeps the QLabel/
        # readback path (its window has no GL surface in v1).
        use_gl = bool(gpu and self._gl_ok and self.glPreview is not None
                      and getattr(self, "_fs_window", None) is None)
        # Prepare the preview audio over the whole loop range [In, Out). It's
        # started later in _on_play_frame, but only while playback is real-time.
        self._prepare_audio(path, in_idx, end)
        self.playBtn.setIcon(self._icon_pause)
        if use_gl:
            self.previewStack.setCurrentWidget(self.glPreview)
        self._player.play(path, start, end, float(st["fps"]) or 30.0,
                          max_dim, self.vignetteCheck.isChecked(), gpu,
                          preview_bin=preview_bin, loop_start=in_idx, loop=True,
                          use_gl=use_gl)

    def _stop_playback(self) -> None:
        self._player.stop()
        self._audio.stop()
        self.playBtn.setIcon(self._icon_play)
        self.playRateLabel.setText("")
        self._leave_gl_surface()

    def _leave_gl_surface(self) -> None:
        """Return the preview stack to the QLabel page after GL playback and
        refresh it with a decoded thumb of the current frame so the image
        doesn't visibly jump back to a stale one."""
        if self.glPreview is not None and \
                self.previewStack.currentWidget() is self.glPreview:
            self.previewStack.setCurrentWidget(self.previewLabel)
            self._kick_thumb_decode()

    @QtCore.Slot()
    def _on_preview_quality_changed(self) -> None:
        # Remember the choice (low-end users want it to stick) and drop the cache
        # since its frames are at the previous resolution. If we're mid-playback,
        # restart so the new quality takes effect right away.
        self._prefs["preview_quality_bin"] = int(self.previewQualityCombo.currentData() or 1)
        self._save_config()
        self._player.clear_cache()
        if self._player.is_playing():
            self._stop_playback()
            self._toggle_play()

    @QtCore.Slot(str, int, QtGui.QImage, float, float, bool)
    def _on_play_frame(self, path: str, idx: int, img: QtGui.QImage,
                       achieved: float, target: float, from_cache: bool) -> None:
        if self._current_path() != path:
            self._audio.stop()   # worker still on the old clip — silence it
            return
        # The worker may have silently fallen back from GL mode (e.g. interop
        # failure) — make sure the QLabel surface is the one showing.
        if self.glPreview is not None and \
                self.previewStack.currentWidget() is self.glPreview:
            self.previewStack.setCurrentWidget(self.previewLabel)
        st = self._file_state.get(path)
        if st is not None:
            st["scrub"] = idx
        self._set_preview_pixmap(QtGui.QPixmap.fromImage(img))
        self.scrubSlider.blockSignals(True)
        self.scrubSlider.setValue(idx)
        self.scrubSlider.blockSignals(False)
        total = st["total"] if st else idx + 1
        self.frameLabel.setText(f"Frame {idx + 1} / {total}")
        self._playback_tick(idx, achieved, target, from_cache, gl=False)

    @QtCore.Slot(str, int, int, int, float, float)
    def _on_gl_frame(self, path: str, idx: int, w: int, h: int,
                     achieved: float, target: float) -> None:
        """Zero-readback frame: the pixels are already sitting in the device
        display buffer — just size the GL surface and schedule its repaint."""
        if self._current_path() != path:
            self._audio.stop()
            return
        st = self._file_state.get(path)
        if st is not None:
            st["scrub"] = idx
        if self.glPreview is not None:
            if self.previewStack.currentWidget() is not self.glPreview:
                self.previewStack.setCurrentWidget(self.glPreview)
            self.glPreview.set_frame_size(w, h)
            self.glPreview.update()
        self.scrubSlider.blockSignals(True)
        self.scrubSlider.setValue(idx)
        self.scrubSlider.blockSignals(False)
        total = st["total"] if st else idx + 1
        self.frameLabel.setText(f"Frame {idx + 1} / {total}")
        self._playback_tick(idx, achieved, target, False, gl=True)

    def _playback_tick(self, idx: int, achieved: float, target: float,
                       from_cache: bool, gl: bool) -> None:
        """Shared per-frame tail for both display paths: real-time indicator
        and the audio auto-mute hysteresis."""
        # A cache hit is delivered instantly, so it always counts as real-time;
        # a miss below target means we're still filling the cache this pass.
        realtime = from_cache or achieved >= target * 0.9
        self._playback_realtime = realtime
        if realtime:
            label = f"{min(achieved, target):.0f} fps"
            if gl:
                label += " · GL"
            elif from_cache:
                label += " · cached"
        elif not from_cache and not gl:
            label = f"{achieved:.0f} fps · caching"
        else:
            label = f"{achieved:.0f} fps · slow"
        self.playRateLabel.setText(label)

        # Audio: audible only when playback is *sustained* real-time. Hysteresis
        # (start after a short real-time run, stop after a longer slow run) keeps
        # frame-time jitter from stuttering the sink. Auto-muted on slow / caching
        # passes; starts aligned to the current frame so it's in sync wherever it
        # kicks in.
        if realtime:
            self._rt_run += 1
            self._slow_run = 0
        else:
            self._slow_run += 1
            self._rt_run = 0
        if self._rt_run >= 5:
            self._audio_should_play = True
        elif self._slow_run >= 12:
            self._audio_should_play = False
        if self._audio.available() and self._audio_slice:
            audible = self._audio_should_play and not self._muted and self._volume > 0.0
            if audible and not self._audio.is_playing():
                self._audio.start(self._audio_slice, self._audio_offset_for(idx))
            elif not audible and self._audio.is_playing():
                self._audio.stop()

    @QtCore.Slot(str)
    def _on_play_finished(self, path: str) -> None:
        self._audio.stop()
        self.playBtn.setIcon(self._icon_play)
        self._leave_gl_surface()

    @QtCore.Slot(str, int, bytes, float, float)
    def _on_play_scopes(self, path: str, _idx: int, hist: bytes,
                        clip_lo: float, clip_hi: float) -> None:
        if self._scopes_on and self._current_path() == path:
            self.scopePanel.set_data(hist, clip_lo, clip_hi)

    @QtCore.Slot()
    def _toggle_scopes(self) -> None:
        """Show/hide the RGB scopes panel. Scope data rides along with each
        decoded frame, so toggling drops the preview cache (its entries were
        built with the other setting) and restarts playback if running —
        same pattern as the preview-quality combo."""
        self._scopes_on = not self._scopes_on
        self.scopeBtn.setChecked(self._scopes_on)
        self.scopePanel.setVisible(self._scopes_on)
        self._prefs["scopes_visible"] = self._scopes_on
        self._save_config()
        self._player.set_scopes_enabled(self._scopes_on)
        self._player.clear_cache()
        if self._player.is_playing():
            self._stop_playback()
            self._toggle_play()
        elif self._scopes_on:
            self.scopePanel.clear()
            self._kick_thumb_decode()   # populate from the current scrub frame

    @QtCore.Slot()
    def _toggle_mute(self) -> None:
        self._muted = not getattr(self, "_muted", False)
        self.muteBtn.setIcon(self._icon_vol_off if self._muted else self._icon_vol_on)
        self._audio.set_muted(self._muted)

    @QtCore.Slot(int)
    def _on_volume_changed(self, value: int) -> None:
        self._volume = value / 100.0
        self._audio.set_volume(self._volume)

    # ----- Preview audio helpers -----
    def _ensure_audio_loaded(self, path: str) -> tuple:
        """Load + cache a clip's PCM (numpy-free), sample rate, channels and frame
        timestamps. Returns (pcm, rate, channels, ts); empties on failure or a
        silent clip. Cached per clip so we only decode the audio once."""
        cached = self._audio_cache.get(path)
        if cached is not None:
            return cached
        result = (b"", 0, 0, [])
        if self._audio.available():
            try:
                d = mcraw.Decoder(path)
                rate = int(d.audio_sample_rate)
                ch = int(d.audio_channels)
                pcm = d.load_audio_bytes() if (rate > 0 and ch > 0) else b""
                result = (pcm, rate, ch, list(d.frames))
            except Exception as exc:
                print(f"[audio] load failed {Path(path).name}: {exc}")
        self._audio_cache[path] = result
        return result

    def _prepare_audio(self, path: str, start: int, end: int) -> None:
        """Build the PCM slice for the [start, end) frame range and configure the
        sink. Audio starts at frame 0, so slice by timestamp (matches the render
        path). Stores slice + context for _on_play_frame / _audio_offset_for."""
        self._audio_slice = b""
        self._audio_rate = 0
        self._audio_channels = 0
        self._audio_ts = []
        self._play_start = start
        self._rt_run = 0
        self._slow_run = 0
        self._audio_should_play = False
        if not self._audio.available():
            return
        pcm, rate, ch, ts = self._ensure_audio_loaded(path)
        if not pcm or rate <= 0 or ch <= 0 or not ts:
            return
        if not self._audio.configure(rate, ch):
            return
        n = len(ts)
        s = max(0, min(start, n - 1))
        e = max(s + 1, min(end, n))
        bps = ch * 2                       # bytes per interleaved sample-frame
        skip = int((ts[s] - ts[0]) / 1e9 * rate) * bps
        keep = int((ts[e - 1] - ts[s]) / 1e9 * rate) * bps
        if skip >= len(pcm):
            return
        self._audio_slice = pcm[skip:skip + keep] if keep > 0 else pcm[skip:]
        self._audio_rate = rate
        self._audio_channels = ch
        self._audio_ts = ts
        self._play_start = s

    def _audio_offset_for(self, idx: int) -> int:
        """Byte offset into the current slice for video frame `idx`, so audio can
        (re)start in sync with the playhead wherever it kicks in."""
        sl = self._audio_slice
        ts = self._audio_ts
        if not sl or self._audio_rate <= 0 or self._audio_channels <= 0 or not ts:
            return 0
        i = max(0, min(idx, len(ts) - 1))
        bps = self._audio_channels * 2
        off = int((ts[i] - ts[self._play_start]) / 1e9 * self._audio_rate) * bps
        return (off % len(sl)) if off > 0 else 0

    @QtCore.Slot()
    def _toggle_fullscreen(self) -> None:
        # Pop the preview label out to a frameless fullscreen window; Esc/F/click
        # returns it. Playback keeps running (same worker, same signal) — but
        # the fullscreen surface is pixmap-based, so a GL-mode playback is
        # restarted in readback mode for the duration (and back on exit).
        was_playing = self._player.is_playing()
        if was_playing:
            self._stop_playback()
        if getattr(self, "_fs_window", None) is not None:
            self._exit_fullscreen()
            if was_playing:
                self._toggle_play()
            return
        fs = FullscreenPreview(self)
        fs.showFullScreen()
        self._fs_window = fs
        # Re-show the current frame in the fullscreen surface.
        st = self._file_state.get(self._current_path() or "")
        if st and st.get("thumb_img") is not None:
            fs.set_image(QtGui.QPixmap.fromImage(st["thumb_img"]))
        if was_playing:
            self._toggle_play()

    def _exit_fullscreen(self) -> None:
        fs = getattr(self, "_fs_window", None)
        if fs is not None:
            fs.close()
            self._fs_window = None

    @QtCore.Slot()
    def _reset_range(self) -> None:
        path = self._current_path()
        st = self._file_state.get(path) if path else None
        if not st:
            return
        st["start"] = 0
        st["end"] = st["total"]
        self._sync_item_text(path)
        self._refresh_preview_for_current()

    def _sync_item_text(self, path: str, status: str | None = None) -> None:
        for i in range(self.dropList.count()):
            it = self.dropList.item(i)
            if it.data(QtCore.Qt.UserRole) == path:
                self._apply_item_text(it, path, status)
                break

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Re-fit current preview to new label size
        path = self._current_path()
        st = self._file_state.get(path) if path else None
        if st and st.get("thumb_img") is not None:
            self._set_preview_pixmap(QtGui.QPixmap.fromImage(st["thumb_img"]))

    # ----- Color space / transfer dropdowns -----
    def _populate_transfer_combo(self) -> None:
        """Refill the transfer dropdown based on the current primaries selection."""
        primaries = self.primariesCombo.currentData() or PRIMARIES_LIST[0]
        valid = VALID_TRANSFERS.get(primaries, ["Linear"])
        prev = self.transferCombo.currentData() if self.transferCombo.count() else None
        self.transferCombo.blockSignals(True)
        self.transferCombo.clear()
        for t in valid:
            self.transferCombo.addItem(t, t)
        # Try to keep the previous transfer if still valid; else first.
        idx = self.transferCombo.findData(prev) if prev else -1
        self.transferCombo.setCurrentIndex(max(0, idx))
        self.transferCombo.blockSignals(False)

    @QtCore.Slot()
    def _primaries_changed(self) -> None:
        self._populate_transfer_combo()
        self._refresh_codec_filter()

    @QtCore.Slot()
    def _transfer_changed(self) -> None:
        self._refresh_codec_filter()

    def _current_colorspace(self) -> str:
        """Resolve the (primaries, transfer) pair to an mcraw short_name.
        Falls back to ACEScg if somehow an invalid combo slipped through."""
        p = self.primariesCombo.currentData()
        t = self.transferCombo.currentData()
        return CS_LOOKUP.get((p, t), "acescg")

    def _refresh_codec_filter(self) -> None:
        """Phase A.4: when transfer is HDR (PQ/HLG), filter the codec dropdown
        to codecs that can write 10-bit. SDR transfers leave it as-is."""
        # _format_changed already manages the codec list per format. We just
        # need to nudge it so the HDR filter is re-applied on top.
        self._format_changed()

    @QtCore.Slot()
    def _update_ten_bit_state(self) -> None:
        """Enable the 10-bit checkbox only for codecs where it's meaningful.
        When an HDR transfer is selected, also force-check it (PQ/HLG in 8-bit
        produces severe banding and shouldn't be allowed)."""
        codec = self.codecCombo.currentData()
        ten_bit_relevant = codec in ("h265", "h265_nvenc", "av1_nvenc")
        self.tenBitCheck.setEnabled(ten_bit_relevant)
        if not ten_bit_relevant:
            self.tenBitCheck.setChecked(False)
            return

        transfer = self.transferCombo.currentData() if hasattr(self, "transferCombo") else None
        if transfer in HDR_TRANSFERS:
            # HDR delivery requires 10-bit. Lock it on.
            self.tenBitCheck.setChecked(True)
            self.tenBitCheck.setEnabled(False)

    def _format_changed(self) -> None:
        fmt = self.formatCombo.currentData()
        prev_codec = self.codecCombo.currentData()

        if fmt == "exr":
            self.codecCombo.setEnabled(False)
            self.bitrateSpin.setEnabled(False)
            self.exrCompCombo.setEnabled(True)
            self.denoiseChromaSpin.setEnabled(False)
            self.denoiseLumaSpin.setEnabled(False)
            self.tenBitCheck.setEnabled(False)
            self.primariesCombo.setEnabled(True)
            self.transferCombo.setEnabled(True)
            return

        if fmt == "mcraw":
            # Trim mode — only the In/Out scrubber matters. Everything else is
            # irrelevant because we're doing a bit-perfect copy of the source.
            self.codecCombo.setEnabled(False)
            self.bitrateSpin.setEnabled(False)
            self.exrCompCombo.setEnabled(False)
            self.denoiseChromaSpin.setEnabled(False)
            self.denoiseLumaSpin.setEnabled(False)
            self.tenBitCheck.setEnabled(False)
            self.tenBitCheck.setChecked(False)
            self.primariesCombo.setEnabled(False)
            self.transferCombo.setEnabled(False)
            return

        # Re-enable color space dropdowns for video formats (they may have been
        # disabled by the MCRAW branch above).
        self.primariesCombo.setEnabled(True)
        self.transferCombo.setEnabled(True)
        self.exrCompCombo.setEnabled(False)
        # Denoise is MP4-only for now (MOV stays clean for NLE workflows).
        denoise_enabled = (fmt == "mp4")
        self.denoiseChromaSpin.setEnabled(denoise_enabled)
        self.denoiseLumaSpin.setEnabled(denoise_enabled)

        # Restock codec list based on container.
        _codec_labels = {
            "prores422":    "ProRes 422",
            "prores422hq":  "ProRes 422 HQ",
            "prores4444":   "ProRes 4444",
            "prores4444xq": "ProRes 4444 XQ",
            "h264":         "H.264 (CPU)",
            "h265":         "H.265 (CPU)",
            "h264_nvenc":   "H.264 (GPU)",
            "h265_nvenc":   "H.265 (GPU)",
            "av1_nvenc":    "AV1 (GPU)",
            "dnxhr_hqx":    "DNxHR HQX (10-bit)",
            "dnxhr_444":    "DNxHR 444 (12-bit)",
            "cineform":     "GoPro CineForm (10-bit, visually lossless)",
        }
        allowed = list(self._mp4_codecs if fmt == "mp4" else self._all_codecs)

        # Phase A.4: HDR transfers (PQ / HLG) require 10-bit-capable codecs.
        # Drop the 8-bit-only ones and auto-enable the 10-bit toggle.
        transfer = self.transferCombo.currentData() if hasattr(self, "transferCombo") else None
        hdr_active = transfer in HDR_TRANSFERS
        if hdr_active:
            EIGHT_BIT_ONLY = {"h264", "h264_nvenc"}
            allowed = [c for c in allowed if c not in EIGHT_BIT_ONLY]

        self.codecCombo.blockSignals(True)
        self.codecCombo.clear()
        for c in allowed:
            label = _codec_labels.get(c, c)
            if hdr_active and c in ("h265", "h265_nvenc", "av1_nvenc"):
                label += "  · 10-bit (HDR)"
            self.codecCombo.addItem(label, c)
        # Restore previous selection if still allowed; else pick a sensible default.
        idx = self.codecCombo.findData(prev_codec)
        if idx < 0:
            if fmt == "mp4":
                default = "h265_nvenc" if "h265_nvenc" in allowed else \
                          ("av1_nvenc" if "av1_nvenc" in allowed else "h265")
            else:
                default = "prores4444"
            idx = self.codecCombo.findData(default)
        self.codecCombo.setCurrentIndex(max(0, idx))
        self.codecCombo.blockSignals(False)
        self.codecCombo.setEnabled(True)
        self._update_ten_bit_state()
        self.bitrateSpin.setEnabled(True)

    def _output_path_for(self, input_path: str) -> str:
        inp = Path(input_path)
        out_dir = Path(self.outputDirEdit.text().strip()) if self.outputDirEdit.text().strip() else inp.parent
        fmt = self.formatCombo.currentData()
        if fmt == "mcraw":
            # Embed the trim range so successive trims get unique names.
            st = self._file_state.get(input_path) or {}
            s, e = int(st.get("start", 0)), int(st.get("end", 0))
            return str(out_dir / f"{inp.stem}_trim_{s}_{e}.mcraw")
        cs_short = self._current_colorspace()
        if fmt == "mov":
            codec = self.codecCombo.currentData()
            return str(out_dir / f"{inp.stem}_{codec}_{cs_short}.mov")
        if fmt == "mp4":
            codec = self.codecCombo.currentData()
            return str(out_dir / f"{inp.stem}_{codec}_{cs_short}.mp4")
        return str(out_dir / f"{inp.stem}_{cs_short}_exr")

    # ----- Render -----
    def _render_all(self) -> None:
        if self.dropList.count() == 0:
            QtWidgets.QMessageBox.information(
                self, "No files", "Add some .mcraw files first.")
            return

        jobs: list[tuple[int, str, str, int, int]] = []
        for i in range(self.dropList.count()):
            it = self.dropList.item(i)
            inp = it.data(QtCore.Qt.UserRole)
            st = self._file_state.get(inp)
            if st is None:
                continue
            # Reset render-time state so re-renders show fresh stats / tooltip.
            st["status"] = "queued"
            st["render_started"] = None
            st["render_finished"] = None
            st["render_progress"] = 0
            st["render_total_frames"] = 0
            st["error_msg"] = None
            start = int(st.get("start", 0))
            end = int(st.get("end", st.get("total", 0)))
            outp = self._output_path_for(inp)
            st["out_path"] = outp
            jobs.append((i, inp, outp, start, end))
            self._apply_item_text(it, inp)

        fmt = self.formatCombo.currentData()
        is_video = fmt in ("mov", "mp4")
        settings = {
            "format": fmt,
            "colorspace": self._current_colorspace(),
            "bitrate": self.bitrateSpin.value(),
        }
        if is_video:
            settings["codec"] = self.codecCombo.currentData()
            # tenBitCheck may be disabled because HDR forced it on — read the
            # checked state regardless of enabled state for HDR.
            if self.tenBitCheck.isChecked():
                settings["ten_bit"] = True
        if self.highlightRecoveryCheck.isChecked():
            settings["highlight_recovery"] = True
        # Vignette + frame-rate conversion apply to video targets (the GPU/CPU
        # bayer pipeline); EXR is a raw scene-referred dump where neither makes
        # sense to bake in.
        if is_video:
            settings["bake_vignette"] = bool(self.vignetteCheck.isChecked())
            tgt = self._frc_target()
            if tgt > 0.0:
                settings["frame_rate_conversion"] = tgt
        if fmt == "mp4":
            settings["denoise_chroma"] = int(self.denoiseChromaSpin.value())
            settings["denoise_luma"] = int(self.denoiseLumaSpin.value())
        if fmt == "exr":
            settings["exr_compression"] = self.exrCompCombo.currentData()

        self._worker = RenderWorker(jobs, settings,
                                    n_workers=int(self.concurrentSpin.value()))
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_progress.connect(self._on_file_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.all_done.connect(self._on_all_done)
        self._thread.started.connect(self._worker.run)

        self._file_progress = {row: 0.0 for row, _, _, _, _ in jobs}
        self._total_jobs = len(jobs)
        self.progress.setVisible(True)
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.renderBtn.setEnabled(False)
        self.cancelBtn.setEnabled(True)
        self.pauseBtn.setEnabled(True)
        self.pauseBtn.setText("Pause")
        self._thread.start()

    def _refresh_overall(self) -> None:
        if self._total_jobs <= 0:
            return
        overall = sum(self._file_progress.values()) / self._total_jobs
        self.progress.setValue(int(overall * 1000))

    @QtCore.Slot(int, str)
    def _on_file_started(self, row: int, path: str) -> None:
        st = self._file_state.get(path)
        if st is not None:
            st["status"] = "rendering"
            st["render_started"] = time.time()
            st["render_finished"] = None
            st["render_progress"] = 0
            st["render_total_frames"] = 0
            st["error_msg"] = None
            # Stash the output path we computed so the tooltip can show it.
            st["out_path"] = self._output_path_for(path)
        it = self.dropList.item(row)
        if it:
            self._apply_item_text(it, path)
        self.statusBar().showMessage(f"Rendering {Path(path).name}")

    @QtCore.Slot(int, int, int)
    def _on_file_progress(self, row: int, current: int, total: int) -> None:
        if total > 0:
            self._file_progress[row] = current / total
            self._refresh_overall()
        it = self.dropList.item(row)
        if not it:
            return
        path = it.data(QtCore.Qt.UserRole)
        st = self._file_state.get(path)
        if st is not None:
            st["render_progress"] = current
            st["render_total_frames"] = total
        self._apply_item_text(it, path)

    @QtCore.Slot(int, bool, str)
    def _on_file_done(self, row: int, ok: bool, msg: str) -> None:
        it = self.dropList.item(row)
        if it:
            path = it.data(QtCore.Qt.UserRole)
            st = self._file_state.get(path)
            if st is not None:
                st["render_finished"] = time.time()
                if ok:
                    st["status"] = "done"
                elif msg == "cancelled":
                    st["status"] = "cancelled"
                else:
                    st["status"] = "failed"
                    st["error_msg"] = msg
            self._apply_item_text(it, path)
        self._file_progress[row] = 1.0 if ok else self._file_progress.get(row, 0.0)
        self._refresh_overall()

    @QtCore.Slot()
    def _on_all_done(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.renderBtn.setEnabled(True)
        self.cancelBtn.setEnabled(False)
        self.pauseBtn.setEnabled(False)
        self.pauseBtn.setText("Pause")
        self.statusBar().showMessage("All done")

    @QtCore.Slot()
    def _toggle_pause(self) -> None:
        """Pause / resume the running batch. Pause halts at the next frame
        boundary (sub-second on most clips); resume picks up exactly where it
        left off — the encoder context is preserved."""
        if not self._worker:
            return
        if self._worker.is_paused():
            self._worker.resume()
            self.pauseBtn.setText("Pause")
            self.statusBar().showMessage("Resuming…", 2000)
            log.info("Render resumed by user")
        else:
            self._worker.pause()
            self.pauseBtn.setText("Resume")
            self.statusBar().showMessage("Paused — click Resume to continue")
            log.info("Render paused by user")

    def _cancel(self) -> None:
        """Confirm with the user, then cancel running renders mid-frame.
        Partial output files (.mov / .mp4 / .mcraw) are deleted. Partial
        EXR sequences keep whatever frames already wrote."""
        if not self._worker:
            return
        # If the user paused first, the dialog won't be confused — but make
        # sure the workers are awake to receive the cancel.
        was_paused = self._worker.is_paused()

        reply = QtWidgets.QMessageBox.question(
            self,
            "Cancel render?",
            "Stop all running renders now?\n\n"
            "Partial output files will be deleted. Files already finished "
            "in this batch will be kept.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            # User backed out. If we were paused, they probably want to resume.
            if was_paused:
                self._worker.resume()
                self.pauseBtn.setText("Pause")
                self.statusBar().showMessage("Resumed", 2000)
            return

        self._worker.cancel()
        self.cancelBtn.setEnabled(False)
        self.pauseBtn.setEnabled(False)
        self.statusBar().showMessage("Cancelling…")
        log.info("Render cancelled by user")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._thread and self._thread.isRunning():
            reply = QtWidgets.QMessageBox.question(
                self, "Render in progress",
                "A render is running. Cancel and exit?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            if self._worker:
                self._worker.cancel()
            self._thread.quit()
            self._thread.wait(5000)
        # Stop the preview player thread + audio + close any fullscreen surface.
        if getattr(self, "_player", None) is not None:
            self._player.stop()
            self._player.clear_cache()
        if getattr(self, "_audio", None) is not None:
            self._audio.stop()
        self._exit_fullscreen()
        # Release CUDA-GL interop state before the GL context goes away.
        if getattr(self, "glPreview", None) is not None:
            try:
                self.glPreview.cleanup()
                mcraw.gl_release()
            except Exception:
                pass
        event.accept()


def _resource_path(name: str) -> Path | None:
    """Locate a bundled resource (qss / icon) at dev-time or in a PyInstaller
    onefile/onedir bundle. Returns None if not found anywhere."""
    candidates = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / name)  # type: ignore[attr-defined]
    candidates.append(Path(__file__).resolve().parent / name)
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def _load_stylesheet() -> str:
    p = _resource_path("style.qss")
    if p is None:
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


APP_VERSION = "0.6.0"

# ----- External links surfaced in the menu bar -------------------------------
DONATE_URL  = "https://afnisse-shop.fourthwall.com/products/mcraw-studio"
REPORT_URL  = "https://github.com/ACES2065-1/MCRAW_Studio"
PROJECT_URL = "https://github.com/ACES2065-1/MCRAW_Studio"

# ----- Crash + render logging ------------------------------------------------
# All logging goes to a single file at:
#   %LOCALAPPDATA%\MCRAWStudio\log.txt   (Windows)
#   ~/.local/share/MCRAWStudio/log.txt   (Linux/Mac)
# The file rotates at ~512 KB and we keep two backups, so it never grows
# without bound. Workflow: user hits a bug → they email this file → you see
# the crash traceback + the exact render settings that triggered it.

def _log_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.path.expanduser("~/.local/share")
    d = Path(base) / "MCRAWStudio"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fall back to the user's home dir if LOCALAPPDATA is unwritable.
        d = Path(os.path.expanduser("~"))
    return d

LOG_PATH = _log_dir() / "log.txt"
log = logging.getLogger("motioncam_tools")


def _setup_logging() -> None:
    """Configure the root logger to write to log.txt (rotating) + stderr in dev."""
    log.setLevel(logging.INFO)
    # Avoid duplicate handlers if _setup_logging is somehow called twice.
    if log.handlers:
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        fh = logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=512_000, backupCount=2, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except Exception as exc:
        # If we can't open the log file (read-only filesystem etc.), keep
        # going without file logging. Console handler below still works.
        print(f"[motioncam_tools] could not open log file {LOG_PATH}: {exc}",
              file=sys.stderr)

    # Console handler — useful when running from source (dev mode). The
    # bundled .exe has console=False so nothing reads stderr; harmless there.
    if not getattr(sys, "frozen", False):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        log.addHandler(ch)

    # Header row at every startup so log.txt has clear session boundaries.
    log.info("=" * 64)
    log.info("MCRAW Studio v%s starting (frozen=%s)",
             APP_VERSION, getattr(sys, "frozen", False))
    log.info("Python %s on %s", sys.version.split()[0], platform.platform())
    log.info("Executable: %s", sys.executable)
    try:
        nv264 = mcraw.encoder_available("h264_nvenc")
        nv265 = mcraw.encoder_available("hevc_nvenc")
        nvav1 = mcraw.encoder_available("av1_nvenc")
        log.info("NVENC available: H.264=%s H.265=%s AV1=%s", nv264, nv265, nvav1)
    except Exception as exc:
        log.info("NVENC probe failed: %s", exc)
    # Diagnostic host probes are best-effort — never block app startup if
    # they fail (e.g. PowerShell unreachable, WMI restricted, etc.).
    try:
        _log_host_info()
    except Exception as exc:
        log.info("host probes failed: %s", _safe_error(exc))


def _install_excepthooks() -> None:
    """Catch unhandled exceptions on the main thread + worker threads,
    write them to log.txt, and surface a friendly dialog so the user sees
    *something* instead of a window vanishing."""

    def main_hook(exc_type, exc_value, exc_tb):
        safe_msg = _safe_unhandled(exc_type, exc_value, exc_tb)
        log.error("UNCAUGHT MAIN-THREAD EXCEPTION  %s", safe_msg)
        # Try to show a dialog if Qt is up.
        app_inst = QtWidgets.QApplication.instance()
        if app_inst is not None:
            try:
                box = QtWidgets.QMessageBox()
                box.setIcon(QtWidgets.QMessageBox.Critical)
                box.setWindowTitle("MCRAW Studio — error")
                box.setText(
                    "Something went wrong and the app couldn't recover.\n\n"
                    f"A diagnostic log was saved to:\n{LOG_PATH}\n\n"
                    "If this keeps happening, please send the log file via "
                    "Help → Report a problem so it can be fixed."
                )
                box.setDetailedText(safe_msg)
                box.setStandardButtons(QtWidgets.QMessageBox.Ok)
                box.exec()
            except Exception:
                pass
        # In dev (running from source) it's still useful to see the full
        # traceback on the console — but only when not frozen.
        if not getattr(sys, "frozen", False):
            sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = main_hook

    def thread_hook(args):
        safe_msg = _safe_unhandled(args.exc_type, args.exc_value, args.exc_traceback)
        thread_name = getattr(args.thread, "name", "<unknown>")
        log.error("UNCAUGHT THREAD EXCEPTION in %s  %s", thread_name, safe_msg)

    threading.excepthook = thread_hook


def main() -> int:
    # Logging + crash capture must come up first, BEFORE QApplication, so any
    # exception during early Qt setup also gets logged.
    _setup_logging()
    _install_excepthooks()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("MCRAW Studio")
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("ACES2065-1")

    icon_path = _resource_path("icon.png")
    if icon_path is not None:
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))

    qss = _load_stylesheet()
    if qss:
        app.setStyleSheet(qss)

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
