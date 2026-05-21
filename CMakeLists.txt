cmake_minimum_required(VERSION 3.15...3.29)

project(
	ExampleDecoder
	VERSION 1.0
	LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(lib/include thirdparty)

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

    add_library(mcraw_color
        lib/BakedTransform.cpp
        lib/ColorPipeline.cpp
        lib/Debayer.cpp
        lib/Denoise.cpp
        lib/ExrWriter.cpp
        lib/OcioTransform.cpp
        lib/MovEncoder.cpp
    )
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
    if (MSVC)
        target_compile_options(mcraw_color PRIVATE
            /utf-8 /W4 /arch:AVX2 /fp:fast)
    else()
        target_compile_options(mcraw_color PRIVATE
            -Wall -O3 -mavx2 -mfma -ffast-math)
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
