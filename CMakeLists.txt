cmake_minimum_required(VERSION 3.18...3.29)

project(
	ExampleDecoder
	VERSION 1.0
	LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(lib/include thirdparty)

# --- Optional CUDA acceleration (Tier 2.1 GPU pipeline) ---
#
# CUDA is detected at configure time. If nvcc is available, MCRAW_HAVE_CUDA
# is defined and the lib/cuda/*.cu files are compiled into mcraw_color.
# Otherwise we silently fall back to the CPU paths everywhere — users
# without an NVIDIA toolkit can still build the project.
#
# The CUDA *runtime* is provided by the NVIDIA driver, not by the toolkit,
# so the shipped .exe only needs the user's existing graphics driver.
# Build host needs the CUDA Toolkit (12.x+) for nvcc.

option(MCRAW_ENABLE_CUDA "Enable CUDA acceleration for NVENC paths" ON)

set(MCRAW_HAVE_CUDA OFF)
if (MCRAW_ENABLE_CUDA)
    # Newer MSVC compilers (VS 17.10+, VS 18 Insiders) often ship before the
    # matching CUDA toolkit knows about them. CUDA's host_config.h then errors
    # out with "unsupported Microsoft Visual Studio version!". Pass
    # --allow-unsupported-compiler so nvcc tries anyway — the MSVC dependency
    # in nvcc is very narrow (just driving cl.exe for host code) and newer
    # MSVCs are basically always backwards-compatible enough to work. Must
    # be set BEFORE check_language so the CUDA-detect compile succeeds.
    set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} --allow-unsupported-compiler")

    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        # SASS for common consumer arches, plus PTX from the newest for
        # forward-compat with future GPUs (JIT'd at first launch).
        #   60 = Pascal (GTX 10xx, 2016)
        #   75 = Turing (RTX 20xx / GTX 16xx, 2018)
        #   86 = Ampere (RTX 30xx, 2020)
        #   89 = Ada Lovelace (RTX 40xx, 2022)
        #   89-virtual = PTX from Ada that JITs for newer GPUs (Blackwell etc.)
        # Capped at 89 because CUDA 12.5 doesn't know compute_120 (Blackwell);
        # bump this list when the project's CUDA toolkit is upgraded.
        # MUST be set BEFORE enable_language(CUDA), otherwise CMake fills
        # in its own default (typically "52") and ignores any later setting.
        set(CMAKE_CUDA_ARCHITECTURES "60-real;75-real;86-real;89-real;89-virtual")
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        set(MCRAW_HAVE_CUDA ON)
        message(STATUS "CUDA toolchain enabled (nvcc=${CMAKE_CUDA_COMPILER}, arches=${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(STATUS "CUDA toolchain not found; building CPU-only.")
    endif()
endif()

# --- Core decoder ---

add_library(motioncam_decoder
    lib/Decoder.cpp
    lib/RawData.cpp
    lib/RawData_Legacy.cpp
    lib/Trimmer.cpp
)
set_property(TARGET motioncam_decoder PROPERTY POSITION_INDEPENDENT_CODE ON)

if (MSVC)
    target_compile_options(motioncam_decoder PRIVATE /utf-8 /W4)
    target_compile_definitions(motioncam_decoder PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(motioncam_decoder PRIVATE -Wall -O3)
endif()

# --- DNG reference example ---

add_executable(example example.cpp)
target_link_libraries(example PRIVATE motioncam_decoder)

if (MSVC)
    target_compile_options(example PRIVATE /utf-8 /W4)
    target_compile_definitions(example PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(example PRIVATE -Wall -O3)
endif()

# --- Phase 1: color pipeline + EXR renderer (requires OpenEXR via vcpkg) ---

option(MCRAW_BUILD_RENDER "Build the mcraw_render tool (requires OpenEXR/Imath)" ON)

if (MCRAW_BUILD_RENDER)
    find_package(Imath CONFIG REQUIRED)
    find_package(OpenEXR CONFIG REQUIRED)
    find_package(OpenColorIO CONFIG REQUIRED)
    find_package(FFMPEG REQUIRED)

    set(MCRAW_COLOR_SOURCES
        lib/BakedTransform.cpp
        lib/ColorPipeline.cpp
        lib/Debayer.cpp
        lib/Denoise.cpp
        lib/ExrWriter.cpp
        lib/OcioTransform.cpp
        lib/MovEncoder.cpp
    )
    if (MCRAW_HAVE_CUDA)
        list(APPEND MCRAW_COLOR_SOURCES
            lib/cuda/HwHandoff.cu
            lib/cuda/BayerPipeline.cu)
    endif()

    add_library(mcraw_color ${MCRAW_COLOR_SOURCES})
    set_property(TARGET mcraw_color PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_include_directories(mcraw_color PRIVATE ${FFMPEG_INCLUDE_DIRS})
    target_link_directories(mcraw_color PRIVATE ${FFMPEG_LIBRARY_DIRS})
    target_link_libraries(mcraw_color
        PUBLIC
            motioncam_decoder
            OpenEXR::OpenEXR
            Imath::Imath
            OpenColorIO::OpenColorIO
        PRIVATE
            ${FFMPEG_LIBRARIES}
    )
    if (MCRAW_HAVE_CUDA)
        target_compile_definitions(mcraw_color PUBLIC MCRAW_HAVE_CUDA=1)
        # Static link of cudart so the shipped .pyd doesn't depend on the
        # exact CUDA Toolkit version installed on the user's machine.
        # Users' NVIDIA drivers ship a forward-compatible CUDA *driver* API
        # — the *runtime* library is what we statically bake in here.
        target_link_libraries(mcraw_color PUBLIC CUDA::cudart_static)
    endif()

    # /arch:AVX2 — lets the auto-vectorizer emit 256-bit SIMD and unlocks
    # AVX2 intrinsics in hand-written kernels (Debayer, ColorPipeline). AVX2
    # has been mainstream since Intel Haswell (2013) and AMD Excavator/Ryzen
    # (2015-2017); any 4K-video-capable Windows machine has it. We do a
    # runtime CPU-feature check at app startup (see RequireAvx2) and abort
    # with a clear message if a user has a pre-AVX2 CPU.
    #
    # /fp:fast — lets the compiler reassociate float operations
    # (a + b + c → (a + c) + b) which is needed for the auto-vectorizer to
    # turn reduction loops into SIMD. Our colour math doesn't depend on
    # bit-exact float ordering — we work in float32 throughout and the
    # output goes through codecs that quantize anyway.
    # MSVC-specific flags wrapped in generator expressions so they only
    # apply when compiling C++ files. nvcc rejects /arch:AVX2 etc. when
    # they're passed bare (it tries to interpret them as input files) —
    # gate by COMPILE_LANGUAGE:CXX so CUDA sources get a clean command line.
    if (MSVC)
        target_compile_options(mcraw_color PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
            $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>)
    else()
        target_compile_options(mcraw_color PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
            $<$<COMPILE_LANGUAGE:CXX>:-mfma>
            $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
    endif()

    add_executable(mcraw_render mcraw_render.cpp)
    target_link_libraries(mcraw_render PRIVATE mcraw_color)

    if (MSVC)
        target_compile_options(mcraw_render PRIVATE
            /utf-8 /W4 /arch:AVX2 /fp:fast)
    else()
        target_compile_options(mcraw_render PRIVATE
            -Wall -O3 -mavx2 -mfma -ffast-math)
    endif()
endif()

# --- Phase 3: Python bindings (pybind11) ---

option(MCRAW_BUILD_PYTHON "Build the mcraw Python module (requires pybind11 + Python 3.12)" ON)

if (MCRAW_BUILD_PYTHON AND MCRAW_BUILD_RENDER)
    # Prefer the user's standalone Python install over vcpkg's bundled one.
    # Override via -DPython_EXECUTABLE=... at configure time, or set the
    # MCRAW_PYTHON env var, or rely on the auto-detection paths below (which
    # cover the common Windows 64-bit Python installer location).
    if (NOT DEFINED Python_EXECUTABLE AND WIN32)
        set(_python_candidates
            "$ENV{MCRAW_PYTHON}"
            "$ENV{LOCALAPPDATA}/Programs/Python/Python312/python.exe"
            "$ENV{LOCALAPPDATA}/Programs/Python/Python313/python.exe"
            "$ENV{ProgramFiles}/Python312/python.exe"
            "$ENV{ProgramFiles}/Python313/python.exe"
        )
        foreach(_p ${_python_candidates})
            if (_p AND EXISTS "${_p}")
                set(Python_EXECUTABLE "${_p}" CACHE FILEPATH "Python executable for mcraw module")
                break()
            endif()
        endforeach()
    endif()

    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(mcraw python/mcraw_py.cpp)
    target_link_libraries(mcraw PRIVATE mcraw_color)
    if (MSVC)
        target_compile_options(mcraw PRIVATE /utf-8 /W4 /arch:AVX2 /fp:fast)
    else()
        target_compile_options(mcraw PRIVATE
            -Wall -O3 -mavx2 -mfma -ffast-math)
    endif()
endif()
