"""Real-time preview player for MCRAW Studio.

Everything the transport bar drives lives here: the playback engine
(PlaybackWorker + PreviewCache), best-effort preview audio (AudioPlayer +
LoopBuffer), the fullscreen surface, and the vector transport icons.

The host (gui/motioncam_tools.py) must run its DLL-path setup and
`import mcraw` BEFORE importing this module — we import mcraw directly and
rely on it already being resolvable.
"""
from __future__ import annotations

import collections
import threading
import time
from pathlib import Path

import mcraw

from PySide6 import QtCore, QtGui, QtWidgets

# QtMultimedia drives preview audio (QAudioSink). It's optional: if the module
# or an audio backend isn't present (e.g. a stripped bundle or a headless box),
# the player runs video-only and the volume/mute controls become no-ops rather
# than crashing. Audio is explicitly the secondary, best-effort feature here.
try:
    from PySide6 import QtMultimedia
    _HAVE_QTMM = True
except Exception as _exc:  # pragma: no cover
    import sys as _sys
    print(f"QtMultimedia unavailable — preview audio disabled: {_exc}",
          file=_sys.stderr)
    _HAVE_QTMM = False


def _qimage_bytes(img: QtGui.QImage) -> int:
    """Best-effort byte size of a QImage (sizeInBytes on PySide6, byteCount on
    older bindings; falls back to a w*h*3 estimate)."""
    fn = getattr(img, "sizeInBytes", None) or getattr(img, "byteCount", None)
    try:
        return int(fn()) if fn else img.width() * img.height() * 3
    except Exception:
        return img.width() * img.height() * 3


def _entry_bytes(entry) -> int:
    """Byte cost of a cache entry: (QImage, hist_bytes|None, clip_lo, clip_hi)."""
    img, hist, _lo, _hi = entry
    return _qimage_bytes(img) + (len(hist) if hist else 0)


class PreviewCache:
    """LRU cache of decoded + scaled preview frames, keyed by frame index for a
    single (clip, params) signature.

    The first play populates it and is decode-limited, so it can run below the
    target rate (e.g. ~15 fps on a heavy 4K CPU path). Replays — and every loop
    pass after the first — read straight from the cache and hit the target rate.
    Bounded by a byte budget with least-recently-used eviction so a long clip
    can't grow it without limit; the most-recent window stays resident.

    Thread-safe: the playback worker fills it from its decode thread while the
    GUI thread may clear it (close / vignette toggle). QImage is implicitly
    shared and never mutated after insertion, so handing the same object to both
    the cache and the UI is safe.
    """
    def __init__(self, budget_bytes: int = 1_200_000_000,
                 enabled: bool = True) -> None:
        self._budget = max(0, int(budget_bytes))
        self._enabled = bool(enabled)
        self._bytes = 0
        self._sig: tuple | None = None
        # idx -> (QImage, hist_bytes | None, clip_lo, clip_hi)
        self._store: "collections.OrderedDict[int, tuple]" = \
            collections.OrderedDict()
        self._lock = threading.Lock()

    def _evict_locked(self) -> None:
        """Drop least-recently-used frames until at/under budget. Caller holds
        the lock. Always keeps at least one frame so a single oversized frame
        can't starve the cache."""
        while self._bytes > self._budget and len(self._store) > 1:
            _, old = self._store.popitem(last=False)
            self._bytes -= _entry_bytes(old)

    def set_budget(self, budget_bytes: int) -> None:
        """Change the RAM ceiling (from Preferences). Shrinks immediately if the
        new budget is smaller than what's currently resident."""
        with self._lock:
            self._budget = max(0, int(budget_bytes))
            self._evict_locked()

    def set_enabled(self, enabled: bool) -> None:
        """Turn caching on/off. Disabling frees the resident frames right away;
        re-enabling lets the next play re-fill it."""
        with self._lock:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._store.clear()
                self._bytes = 0

    def set_signature(self, sig: tuple) -> None:
        """Point the cache at a (clip, params) signature, dropping everything if
        the signature changed (different clip, vignette state or preview size)."""
        with self._lock:
            if sig != self._sig:
                self._store.clear()
                self._bytes = 0
                self._sig = sig

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._bytes = 0
            self._sig = None

    def get(self, idx: int) -> tuple | None:
        with self._lock:
            entry = self._store.get(idx)
            if entry is not None:
                self._store.move_to_end(idx)
            return entry

    def put(self, idx: int, entry: tuple) -> None:
        with self._lock:
            if not self._enabled or self._budget <= 0:
                return
            if idx in self._store:
                self._store.move_to_end(idx)
                return
            self._store[idx] = entry
            self._bytes += _entry_bytes(entry)
            self._evict_locked()


class LoopBuffer(QtCore.QIODevice):
    """Read-only QIODevice that yields a fixed PCM byte slice on repeat, so a
    QAudioSink (pull mode) loops it gaplessly. Starts at `start` bytes in so the
    caller can align the loop to the video playhead. Touches only its own state,
    so the audio backend thread can pull from it while the GUI drives start/stop."""
    def __init__(self, data: bytes, start: int = 0) -> None:
        super().__init__()
        self._data = data or b""
        self._pos = (start % len(self._data)) if self._data else 0

    def readData(self, maxlen: int) -> bytes:
        data = self._data
        if not data or maxlen <= 0:
            return b""
        out = bytearray()
        need = int(maxlen)
        ld = len(data)
        while need > 0:
            end = min(self._pos + need, ld)
            out += data[self._pos:end]
            need -= (end - self._pos)
            self._pos = end
            if self._pos >= ld:
                self._pos = 0
        return bytes(out)

    def writeData(self, _data) -> int:
        return 0

    def bytesAvailable(self) -> int:
        # Effectively endless (we loop), so always advertise data is ready.
        return 0x7FFFFFFF if self._data else 0

    def isSequential(self) -> bool:
        return True


class AudioPlayer(QtCore.QObject):
    """Best-effort preview audio: loops a PCM slice through a QAudioSink. It only
    sounds when the video is playing in real time (the caller auto-mutes it on
    slow / caching passes) and degrades to a silent no-op when QtMultimedia or an
    output device is missing. Lives on the GUI thread; driven from the playback
    slots."""
    def __init__(self) -> None:
        super().__init__()
        self._ok = _HAVE_QTMM
        self._sink = None
        self._dev: LoopBuffer | None = None
        self._rate = 0
        self._channels = 0
        self._volume = 0.8
        self._muted = False
        self._playing = False

    def available(self) -> bool:
        return bool(self._ok)

    def configure(self, rate: int, channels: int) -> bool:
        """Build / rebuild the sink for a clip's PCM format. No-op if unchanged.
        Returns True if audio is usable."""
        if not self._ok or rate <= 0 or channels <= 0:
            return False
        if self._sink is not None and rate == self._rate and channels == self._channels:
            return True
        try:
            fmt = QtMultimedia.QAudioFormat()
            fmt.setSampleRate(int(rate))
            fmt.setChannelCount(int(channels))
            fmt.setSampleFormat(QtMultimedia.QAudioFormat.SampleFormat.Int16)
            dev = QtMultimedia.QMediaDevices.defaultAudioOutput()
            if dev is None or dev.isNull():
                self._ok = False
                return False
            self.stop()
            self._sink = QtMultimedia.QAudioSink(dev, fmt)
            self._rate = int(rate)
            self._channels = int(channels)
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[audio] sink init failed: {exc}")
            self._ok = False
            return False

    def is_playing(self) -> bool:
        return self._playing

    # Target sink latency: small enough that any residual A/V offset is barely
    # perceptible, large enough that the loop device never underruns (it's a pure
    # memcpy on the audio thread, so this can be tight).
    _LATENCY_SEC = 0.04

    def start(self, pcm: bytes, start_byte: int = 0) -> None:
        """Begin looping `pcm`, positioned `start_byte` in (aligns to playhead).
        Pre-rolls by the sink latency so the sound that actually emerges (~one
        buffer later) lines up with where the video will be by then, instead of
        lagging behind it."""
        if not self._ok or self._sink is None or not pcm:
            return
        try:
            self.stop()
            comp = 0
            if self._rate > 0 and self._channels > 0:
                comp = int(self._rate * self._LATENCY_SEC) * self._channels * 2
                try:
                    self._sink.setBufferSize(comp)   # shrink the default buffer
                except Exception:
                    pass
            pos = ((start_byte + comp) % len(pcm)) if pcm else 0
            self._dev = LoopBuffer(pcm, pos)
            self._dev.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
            self._sink.setVolume(0.0 if self._muted else self._volume)
            self._sink.start(self._dev)
            self._playing = True
        except Exception as exc:  # pragma: no cover
            print(f"[audio] start failed: {exc}")
            self._playing = False

    def stop(self) -> None:
        self._playing = False
        if self._sink is not None:
            try:
                self._sink.stop()
            except Exception:
                pass
        if self._dev is not None:
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None

    def set_volume(self, v: float) -> None:
        self._volume = max(0.0, min(1.0, float(v)))
        if self._sink is not None and not self._muted:
            try:
                self._sink.setVolume(self._volume)
            except Exception:
                pass

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        if self._sink is not None:
            try:
                self._sink.setVolume(0.0 if self._muted else self._volume)
            except Exception:
                pass


class PlaybackWorker(QtCore.QObject):
    """Plays a clip frame-by-frame on a background thread, pacing to the clip
    fps. Decodes each frame to a downscaled sRGB QImage (GPU pipeline when
    available, else CPU) and emits it; measures the achieved rate so the UI can
    show a proxy/real-time indicator and auto-mute audio when it can't keep up.
    """
    # path, frame_idx, image, achieved_fps, target_fps, from_cache
    frame = QtCore.Signal(str, int, QtGui.QImage, float, float, bool)
    # path, frame_idx, hist_bytes (3*256 LE uint32), clip_lo, clip_hi
    scopes = QtCore.Signal(str, int, bytes, float, float)
    finished = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Scopes are opt-in (the panel is hidden by default); the flag is
        # sampled at play() time and baked into the cache signature so cached
        # frames always carry matching scope data.
        self._scopes_enabled = False
        # Live preview cache: first play fills it, replays/loops read it back at
        # the target rate. Persists across play/stop (the worker outlives both).
        self._cache = PreviewCache()

    def set_scopes_enabled(self, enabled: bool) -> None:
        """Toggle scope computation. Caller should clear the cache and restart
        playback for it to take effect mid-play (mirrors the quality combo)."""
        self._scopes_enabled = bool(enabled)

    def clear_cache(self) -> None:
        """Drop the cached preview frames (free memory on close / param change)."""
        self._cache.clear()

    def set_cache_budget(self, budget_bytes: int) -> None:
        """Apply a new preview-cache RAM ceiling (from Preferences)."""
        self._cache.set_budget(budget_bytes)

    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable/disable the preview cache (from Preferences)."""
        self._cache.set_enabled(enabled)

    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._thread = None

    def play(self, path: str, start_idx: int, end_idx: int, fps: float,
             max_dim: int, bake_vignette: bool, prefer_gpu: bool,
             preview_bin: int = 1, loop_start: int | None = None,
             loop: bool = True) -> None:
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(path, start_idx, end_idx, fps, max_dim, bake_vignette,
                  prefer_gpu, preview_bin, loop_start, loop),
            daemon=True)
        self._thread.start()

    def _run(self, path, start_idx, end_idx, fps, max_dim, bake, prefer_gpu,
             preview_bin, loop_start, loop):
        try:
            d = mcraw.Decoder(path)
            ts = d.frames
        except Exception as exc:
            print(f"[play] open failed {Path(path).name}: {exc}")
            self.finished.emit(path)
            return
        n = len(ts)
        end_idx = min(end_idx, n)
        idx = max(0, min(start_idx, end_idx - 1)) if end_idx > 0 else 0
        # Loop back to the In point, not to where playback happened to start.
        loop_back = idx if loop_start is None else max(0, min(int(loop_start), end_idx - 1))
        period = (1.0 / fps) if fps > 0 else (1.0 / 30.0)
        # GPU preview frame source when requested + available (process_frame_rgb24
        # runs the CUDA bayer pipeline for baked targets); else CPU.
        use_gpu = bool(prefer_gpu and mcraw.cuda_available())
        preview_bin = max(1, int(preview_bin))
        want_scopes = bool(self._scopes_enabled)
        # Point the live cache at this clip+params. The first pass decodes &
        # fills it (may run below real-time); replays and later loop passes read
        # it back, so they hit the target rate even on a slow CPU path. The bin
        # factor is part of the key so each preview quality caches separately;
        # so is the scopes flag, since cached entries carry their scope data.
        self._cache.set_signature((path, bool(bake), int(max_dim), use_gpu,
                                   preview_bin, want_scopes))
        while not self._stop.is_set():
            t0 = time.perf_counter()
            entry = self._cache.get(idx)
            from_cache = entry is not None
            if from_cache:
                img, hist, clip_lo, clip_hi = entry
            else:
                try:
                    if want_scopes:
                        buf, h, w, hist, clip_lo, clip_hi = \
                            d.process_frame_rgb24_scopes(ts[idx], "srgb", False,
                                                         bake, use_gpu, preview_bin)
                    else:
                        buf, h, w = d.process_frame_rgb24(ts[idx], "srgb", False,
                                                          bake, use_gpu, preview_bin)
                        hist, clip_lo, clip_hi = None, 0.0, 0.0
                except Exception as exc:
                    print(f"[play] decode failed frame {idx}: {exc}")
                    break
                img = QtGui.QImage(buf, w, h, 3 * w, QtGui.QImage.Format_RGB888)
                if max_dim > 0 and (w > max_dim or h > max_dim):
                    img = img.scaled(max_dim, max_dim, QtCore.Qt.KeepAspectRatio,
                                     QtCore.Qt.SmoothTransformation)
                else:
                    img = img.copy()  # detach from `buf` before it's freed
                self._cache.put(idx, (img, hist, clip_lo, clip_hi))
            dt = time.perf_counter() - t0
            achieved = (1.0 / dt) if dt > 0 else fps
            self.frame.emit(path, idx, img, achieved, fps, from_cache)
            if want_scopes and hist is not None:
                self.scopes.emit(path, idx, hist, clip_lo, clip_hi)
            idx += 1
            if idx >= end_idx:
                if not loop:
                    break
                idx = loop_back
            sleep = period - (time.perf_counter() - t0)
            if sleep > 0:
                self._stop.wait(sleep)
        self.finished.emit(path)


class FullscreenPreview(QtWidgets.QWidget):
    """Frameless black fullscreen surface for the preview. Esc / F / double-click
    returns to the window; Space toggles playback."""
    def __init__(self, owner) -> None:
        super().__init__()
        self._owner = owner
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet("background:#000;")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._label = QtWidgets.QLabel()
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        self._label.setStyleSheet("background:#000;")
        lay.addWidget(self._label)
        self._pixmap: QtGui.QPixmap | None = None

    def set_image(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = pixmap
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap and not self._pixmap.isNull():
            self._label.setPixmap(self._pixmap.scaled(
                self.size(), QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        self._rescale()

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_F):
            self._owner._exit_fullscreen()
        elif e.key() == QtCore.Qt.Key_Space:
            self._owner._toggle_play()
        else:
            super().keyPressEvent(e)

    def mouseDoubleClickEvent(self, e: QtGui.QMouseEvent) -> None:
        self._owner._exit_fullscreen()


class ScopePanel(QtWidgets.QWidget):
    """RGB histogram + clipping readout for the preview player.

    Shows the distribution of the displayed image (3 x 256 bins, log-scaled
    counts) with the R/G/B channels overlaid, plus the fraction of pixels
    clipping above 1.0 / below 0.0 measured on the PRE-quantize float output
    (so highlight clipping is real, not a byproduct of the 8-bit preview).
    Fed by PlaybackWorker.scopes during playback and by the thumb decoder
    while scrubbing.
    """

    _COLORS = (QtGui.QColor(224, 85, 85),     # R
               QtGui.QColor(85, 201, 100),    # G
               QtGui.QColor(90, 160, 232))    # B

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(96)
        self.setMaximumHeight(140)
        self._hist: list[list[float]] | None = None   # 3 x 256, log-scaled 0..1
        self._clip_lo = 0.0
        self._clip_hi = 0.0

    def clear(self) -> None:
        self._hist = None
        self._clip_lo = 0.0
        self._clip_hi = 0.0
        self.update()

    def set_data(self, hist_bytes: bytes, clip_lo: float, clip_hi: float) -> None:
        """hist_bytes: 3*256 little-endian uint32 (numpy-free — the bundle
        excludes numpy, so parse with array + math)."""
        import math
        from array import array
        raw = array("I", hist_bytes)
        if len(raw) != 768:
            return
        out: list[list[float]] = []
        for c in range(3):
            ch = raw[c * 256:(c + 1) * 256]
            peak = max(ch) or 1
            log_peak = math.log1p(peak)
            out.append([math.log1p(v) / log_peak for v in ch])
        self._hist = out
        self._clip_lo = float(clip_lo)
        self._clip_hi = float(clip_hi)
        self.update()

    def paintEvent(self, _e: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(16, 16, 16))
        w, h = self.width(), self.height()
        pad = 6
        plot_w, plot_h = w - 2 * pad, h - 2 * pad
        if plot_w <= 10 or plot_h <= 10:
            p.end()
            return

        # Quarter gridlines.
        p.setPen(QtGui.QColor(40, 40, 40))
        for q in (0.25, 0.5, 0.75):
            x = pad + int(plot_w * q)
            p.drawLine(x, pad, x, pad + plot_h)

        if self._hist is None:
            p.setPen(QtGui.QColor(120, 120, 120))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "(no scope data)")
            p.end()
            return

        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        for c in range(3):
            col = QtGui.QColor(self._COLORS[c])
            col.setAlpha(150)
            poly = QtGui.QPolygonF()
            poly.append(QtCore.QPointF(pad, pad + plot_h))
            vals = self._hist[c]
            for i in range(256):
                x = pad + plot_w * (i / 255.0)
                y = pad + plot_h * (1.0 - vals[i])
                poly.append(QtCore.QPointF(x, y))
            poly.append(QtCore.QPointF(pad + plot_w, pad + plot_h))
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(col)
            p.drawPolygon(poly)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)

        # Clip readout, top-right. Highlight clipping turns hot when > 0.1%.
        p.setFont(QtGui.QFont(self.font().family(), 8))
        hi_hot = self._clip_hi > 0.001
        p.setPen(QtGui.QColor(240, 120, 90) if hi_hot else QtGui.QColor(150, 150, 150))
        txt_hi = f"▲ {self._clip_hi * 100.0:.2f}%"
        txt_lo = f"▼ {self._clip_lo * 100.0:.2f}%"
        p.drawText(QtCore.QRect(pad, pad, plot_w - 4, 14),
                   QtCore.Qt.AlignRight | QtCore.Qt.AlignTop, txt_hi)
        p.setPen(QtGui.QColor(150, 150, 150))
        p.drawText(QtCore.QRect(pad, pad + 14, plot_w - 4, 14),
                   QtCore.Qt.AlignRight | QtCore.Qt.AlignTop, txt_lo)
        p.end()


# ----- Transport icons (drawn with QPainter — crisp, themeable, no assets) ---

def transport_icon(kind: str, color: str = "#dcdcdc", size: int = 64) -> QtGui.QIcon:
    """Vector transport-bar icon rendered to a QIcon. Drawn on a 64-unit
    canvas so Qt scales it crisply to any button/DPI size."""
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    s = size / 64.0
    col = QtGui.QColor(color)

    def fill():
        p.setPen(QtCore.Qt.NoPen); p.setBrush(col)

    def stroke(w=6.0):
        pen = QtGui.QPen(col); pen.setWidthF(w * s)
        pen.setCapStyle(QtCore.Qt.RoundCap); pen.setJoinStyle(QtCore.Qt.RoundJoin)
        p.setPen(pen); p.setBrush(QtCore.Qt.NoBrush)

    def pt(x, y):
        return QtCore.QPointF(x * s, y * s)

    def poly(points):
        p.drawPolygon(QtGui.QPolygonF([pt(x, y) for x, y in points]))

    def polyline(points):
        p.drawPolyline(QtGui.QPolygonF([pt(x, y) for x, y in points]))

    if kind == "play":
        fill(); poly([(22, 16), (22, 48), (48, 32)])
    elif kind == "pause":
        fill()
        p.drawRoundedRect(QtCore.QRectF(pt(22, 16), pt(30, 48)), 2 * s, 2 * s)
        p.drawRoundedRect(QtCore.QRectF(pt(34, 16), pt(42, 48)), 2 * s, 2 * s)
    elif kind == "skip_start":
        fill()
        p.drawRoundedRect(QtCore.QRectF(pt(16, 17), pt(21, 47)), 2 * s, 2 * s)
        poly([(46, 16), (46, 48), (24, 32)])
    elif kind == "skip_end":
        fill()
        p.drawRoundedRect(QtCore.QRectF(pt(43, 17), pt(48, 47)), 2 * s, 2 * s)
        poly([(18, 16), (18, 48), (40, 32)])
    elif kind == "mark_in":
        stroke(6); polyline([(40, 14), (24, 14), (24, 50), (40, 50)])
    elif kind == "mark_out":
        stroke(6); polyline([(24, 14), (40, 14), (40, 50), (24, 50)])
    elif kind == "fullscreen":
        stroke(5)
        polyline([(16, 26), (16, 16), (26, 16)])
        polyline([(38, 16), (48, 16), (48, 26)])
        polyline([(16, 38), (16, 48), (26, 48)])
        polyline([(48, 38), (48, 48), (38, 48)])
    elif kind == "reset":
        stroke(5)
        r = QtCore.QRectF(pt(18, 18), pt(46, 46))
        p.drawArc(r, 75 * 16, 250 * 16)  # ~3/4 ring, gap upper-right
        fill()  # arrowhead at the ring's open (upper) end
        poly([(32, 9), (32, 23), (40, 16)])
    elif kind == "scope":
        fill()
        # Tiny histogram: four bars of varying height on a baseline.
        for bx, bh_ in ((16, 14), (24, 26), (32, 34), (40, 20)):
            p.drawRoundedRect(
                QtCore.QRectF(pt(bx, 48 - bh_), pt(bx + 6, 48)), 1.5 * s, 1.5 * s)
        stroke(4)
        polyline([(14, 50), (50, 50)])
    elif kind in ("vol_on", "vol_off"):
        fill()
        poly([(16, 26), (24, 26), (34, 17), (34, 47), (24, 38), (16, 38)])
        if kind == "vol_on":
            stroke(4)
            p.drawArc(QtCore.QRectF(pt(30, 22), pt(46, 42)), -55 * 16, 110 * 16)
            p.drawArc(QtCore.QRectF(pt(30, 16), pt(54, 48)), -50 * 16, 100 * 16)
        else:
            stroke(5)
            polyline([(40, 26), (50, 38)]); polyline([(50, 26), (40, 38)])

    p.end()
    return QtGui.QIcon(pm)
