# PyInstaller spec for MCRAWStudio.exe — single-file bundle.
#
# Build with:  python -m PyInstaller motioncam_tools.spec --noconfirm
#
# Two host-specific paths are needed; the spec defaults to the dev-box layout
# but can be overridden via env vars so other contributors don't have to edit
# this file:
#
#   VCPKG_INSTALL_DIR   -> directory containing the vcpkg "bin" folder for the
#                          x64-windows triplet (default: C:\dev\vcpkg\installed\x64-windows)
#   MSVC_REDIST_BIN_DIR -> directory containing msvcp140*.dll / vcruntime140*.dll
#                          for the MSVC version that built mcraw.pyd. We need to
#                          ship these because PyInstaller's auto-discovery misses
#                          them and Windows' system copies may be older than
#                          what mcraw.pyd was linked against (VS 18 Insiders
#                          MSVC 14.50 by default in this project).

import os
from pathlib import Path

PROJECT_ROOT = Path(SPECPATH)
BUILD_DIR    = PROJECT_ROOT / "build"
ENTRY        = PROJECT_ROOT / "gui" / "motioncam_tools.py"

VCPKG_INSTALL_DIR = Path(os.environ.get(
    "VCPKG_INSTALL_DIR",
    r"C:\dev\vcpkg\installed\x64-windows"))
VCPKG_BIN = VCPKG_INSTALL_DIR / "bin"

# Bundle every DLL from vcpkg's bin dir into a private "ffmpeg_runtime"
# subdirectory of the PyInstaller extraction tree, NOT the bundle root. The
# Python side registers that subdir via os.add_dll_directory() before
# `import mcraw`, so mcraw.pyd's implicit DLL imports (avcodec / swscale /
# x265 / OpenColorIO / etc.) resolve from our copy instead of any ffmpeg a
# user happens to have on their PATH. Without this isolation, an installed
# ffmpeg in PATH can shadow our bundled DLLs by name, leading to ABI
# mismatches that surface as ENOMEM / "external library error" / mid-encode
# allocator failures.
binaries = []
for dll in VCPKG_BIN.glob("*.dll"):
    binaries.append((str(dll), "ffmpeg_runtime"))

# mcraw.pyd stays in the bundle root so Python's standard import machinery
# can find it via sys.path. Its implicit ffmpeg dependencies are resolved
# via the USER_DIRS list (populated by os.add_dll_directory in
# _setup_paths), which Python passes to LoadLibraryExW for extension
# module loads.
mcraw_pyd = BUILD_DIR / "mcraw.cp312-win_amd64.pyd"
if mcraw_pyd.exists():
    binaries.append((str(mcraw_pyd), "."))

# Bundle the MSVC runtime DLLs that match what mcraw.pyd was built against.
# PyInstaller's auto-discovery misses MSVCP140* (C++ stdlib) which our
# parallelism code needs (std::atomic_wait, std::thread, std::mutex).
# Without these, the loader crashes with 0xC0000005 before Python even starts.
_VS_BIN_DEFAULT = (r"C:\Program Files\Microsoft Visual Studio\18\Insiders"
                   r"\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64")
VS_BIN = Path(os.environ.get("MSVC_REDIST_BIN_DIR", _VS_BIN_DEFAULT))
if VS_BIN.exists():
    for name in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
                 "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
                 "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll"):
        p = VS_BIN / name
        if p.exists():
            binaries.append((str(p), "."))
else:
    print(f"WARNING: MSVC redist dir not found: {VS_BIN}\n"
          f"  Set MSVC_REDIST_BIN_DIR env var to override. Without these DLLs,\n"
          f"  the bundled .exe may crash on startup if the user's machine\n"
          f"  doesn't already have a compatible MSVC runtime installed.")

block_cipher = None

a = Analysis(
    [str(ENTRY)],
    pathex=[str(BUILD_DIR)],
    binaries=binaries,
    datas=[
        (str(PROJECT_ROOT / "gui" / "style.qss"), "."),
        (str(PROJECT_ROOT / "gui" / "icon.png"), "."),
    ],
    hiddenimports=["mcraw"],
    hookspath=[],
    runtime_hooks=[],
    # NumPy bundled into PyInstaller crashes during pyiboot01_bootstrap on this
    # Python 3.12 + pyinstaller 6.20 + numpy 1.26 combo (0xC0000005 before any
    # Python output). Workaround: exclude numpy entirely and use the bytes-based
    # process_frame_rgb24 path on the GUI side. CLI users pip-install numpy.
    excludes=["numpy", "numpy.core", "numpy.linalg", "numpy.random",
              "numpy.fft", "numpy.polynomial", "numpy.matrixlib",
              "numpy.testing"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="MCRAWStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=str(PROJECT_ROOT / "version.txt"),
    icon=str(PROJECT_ROOT / "gui" / "icon.ico"),
)
