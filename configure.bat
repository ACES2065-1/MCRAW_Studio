@echo off
rem ===========================================================================
rem MCRAW Studio - configure step (Windows + MSVC + vcpkg + Ninja)
rem ===========================================================================
rem
rem Sets up the MSVC environment and runs CMake to configure a Ninja build
rem in .\build. Re-runnable; safe to call repeatedly.
rem
rem Required on first use:
rem   * Visual Studio 2022 (Community / Professional) OR Visual Studio 18
rem     Insiders, with the "Desktop development with C++" workload installed
rem   * vcpkg checked out somewhere (default looks in C:\dev\vcpkg)
rem
rem Override any of these via environment variables - none are required if
rem your install matches the defaults:
rem
rem   VCPKG_ROOT          Path to your vcpkg checkout.
rem                       Default: C:\dev\vcpkg
rem
rem   MSVC_VCVARSALL      Full path to vcvarsall.bat for your MSVC install.
rem                       If unset, we try VS 18 Insiders, then VS 2022
rem                       Community / Professional / Enterprise in turn.
rem
rem   CMAKE_EXE           cmake.exe to use. Defaults to whatever is on PATH;
rem                       falls back to vcpkg's bundled cmake under
rem                       %VCPKG_ROOT%\downloads\tools\cmake-*\.
rem
rem   NINJA_EXE           ninja.exe to use. Defaults to whatever is on PATH;
rem                       falls back to vcpkg's bundled ninja under
rem                       %VCPKG_ROOT%\downloads\tools\ninja-*\.
rem
rem Any extra args you pass to this script are forwarded to cmake - handy for
rem one-off overrides like:  configure.bat -DCMAKE_BUILD_TYPE=Debug
rem ===========================================================================

setlocal enabledelayedexpansion

rem ----- vcpkg ---------------------------------------------------------------
if not defined VCPKG_ROOT set "VCPKG_ROOT=C:\dev\vcpkg"
if not exist "%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" (
    echo ERROR: vcpkg toolchain file not found at:
    echo   %VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
    echo.
    echo Set VCPKG_ROOT to point at your vcpkg checkout, or install vcpkg
    echo at the default location C:\dev\vcpkg
    exit /b 1
)
set "TOOLCHAIN=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"

rem ----- CUDA Toolkit (optional, enables GPU acceleration) ------------------
rem  Auto-detect a full CUDA Toolkit install with nvcc. If we find one, CMake's
rem  enable_language(CUDA) will use it and the build picks up the .cu sources.
rem  If none is found we silently fall back to a CPU-only build.
rem
rem  Override via CUDA_TOOLKIT_ROOT (point at the dir containing bin\nvcc.exe).
rem  IMPORTANT: NVIDIA ships separate "runtime" and "full toolkit" installs
rem  under v13.0 / v12.5 / etc. The runtime install has no nvcc - make sure
rem  you have the developer toolkit (downloaded from developer.nvidia.com).
if not defined CUDA_TOOLKIT_ROOT call :find_cuda
if defined CUDA_TOOLKIT_ROOT (
    set "PATH=%CUDA_TOOLKIT_ROOT%\bin;%PATH%"
)

rem ----- MSVC vcvarsall ------------------------------------------------------
if not defined MSVC_VCVARSALL call :find_vcvars
if not defined MSVC_VCVARSALL (
    echo ERROR: could not find vcvarsall.bat. Install Visual Studio 2022 with
    echo the "Desktop development with C++" workload, or set MSVC_VCVARSALL
    echo to your install's vcvarsall.bat explicitly.
    exit /b 1
)

call "%MSVC_VCVARSALL%" x64
if errorlevel 1 exit /b %errorlevel%

rem ----- cmake.exe -----------------------------------------------------------
if not defined CMAKE_EXE call :find_cmake
if not defined CMAKE_EXE (
    echo ERROR: cmake.exe not found on PATH or in %VCPKG_ROOT%\downloads\tools.
    echo Install CMake from https://cmake.org/download/ or set CMAKE_EXE.
    exit /b 1
)

rem ----- ninja.exe -----------------------------------------------------------
if not defined NINJA_EXE call :find_ninja
if not defined NINJA_EXE (
    echo ERROR: ninja.exe not found on PATH or in %VCPKG_ROOT%\downloads\tools.
    echo Install Ninja from https://github.com/ninja-build/ninja/releases or
    echo set NINJA_EXE.
    exit /b 1
)

echo Using:
echo   VCPKG_ROOT          = %VCPKG_ROOT%
echo   MSVC_VCVARSALL      = %MSVC_VCVARSALL%
echo   CMAKE_EXE           = %CMAKE_EXE%
echo   NINJA_EXE           = %NINJA_EXE%
if defined CUDA_TOOLKIT_ROOT (
    echo   CUDA_TOOLKIT_ROOT   = %CUDA_TOOLKIT_ROOT%
) else (
    echo   CUDA_TOOLKIT_ROOT   = ^(not found - building CPU-only^)
)
echo.

if defined CUDA_TOOLKIT_ROOT (
    "%CMAKE_EXE%" -B build -S . -G Ninja ^
        -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" ^
        -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%" ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DCMAKE_CUDA_COMPILER="%CUDA_TOOLKIT_ROOT%\bin\nvcc.exe" ^
        -DCUDAToolkit_ROOT="%CUDA_TOOLKIT_ROOT%" %*
) else (
    "%CMAKE_EXE%" -B build -S . -G Ninja ^
        -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" ^
        -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%" ^
        -DCMAKE_BUILD_TYPE=Release %*
)

endlocal
goto :eof


rem ===========================================================================
rem  Subroutines (called via "call :name")
rem ===========================================================================

:find_cuda
rem  Look for a full CUDA toolkit (must have nvcc.exe). Prefer newest version.
for %%V in (v13.0 v12.9 v12.8 v12.7 v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0) do (
    set "_TRY=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V"
    if exist "!_TRY!\bin\nvcc.exe" set "CUDA_TOOLKIT_ROOT=!_TRY!" & goto :eof
)
goto :eof

:find_vcvars
set "_TRY=C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat"
if exist "%_TRY%" set "MSVC_VCVARSALL=%_TRY%" & goto :eof
set "_TRY=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
if exist "%_TRY%" set "MSVC_VCVARSALL=%_TRY%" & goto :eof
set "_TRY=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
if exist "%_TRY%" set "MSVC_VCVARSALL=%_TRY%" & goto :eof
set "_TRY=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
if exist "%_TRY%" set "MSVC_VCVARSALL=%_TRY%" & goto :eof
set "_TRY=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if exist "%_TRY%" set "MSVC_VCVARSALL=%_TRY%" & goto :eof
goto :eof

:find_cmake
where cmake.exe >nul 2>&1
if not errorlevel 1 (
    set "CMAKE_EXE=cmake.exe"
    goto :eof
)
for /d %%D in ("%VCPKG_ROOT%\downloads\tools\cmake-*") do (
    if exist "%%D\bin\cmake.exe" set "CMAKE_EXE=%%D\bin\cmake.exe"
)
if defined CMAKE_EXE goto :eof
rem  Newer vcpkg nests the binary one level deeper: cmake-X.Y.Z-windows\cmake-X.Y.Z-windows-x86_64\bin
for /d %%D in ("%VCPKG_ROOT%\downloads\tools\cmake-*") do (
    for /d %%E in ("%%D\cmake-*") do (
        if exist "%%E\bin\cmake.exe" set "CMAKE_EXE=%%E\bin\cmake.exe"
    )
)
goto :eof

:find_ninja
where ninja.exe >nul 2>&1
if not errorlevel 1 (
    set "NINJA_EXE=ninja.exe"
    goto :eof
)
for /d %%D in ("%VCPKG_ROOT%\downloads\tools\ninja-*") do (
    if exist "%%D\ninja.exe" set "NINJA_EXE=%%D\ninja.exe"
)
goto :eof
