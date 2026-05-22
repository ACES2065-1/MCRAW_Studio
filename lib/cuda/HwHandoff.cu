// Phase A of the CUDA pipeline — toolchain proof.
//
// This file does two things:
//   1. Detect whether a CUDA-capable GPU is present at runtime
//      (cached, so we only ask the driver once per process).
//   2. Run a no-op kernel and synchronise, to prove the whole
//      build chain — nvcc compile -> link -> kernel launch ->
//      device sync — works end-to-end.
//
// As we add Phase B (RGB float -> YUV) and Phase C (bayer -> RGB) we
// extend this file with real kernels. The C++ interface lives in
// motioncam/CudaHwHandoff.hpp so the rest of the codebase never sees
// CUDA types directly.

#include <motioncam/CudaHwHandoff.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <mutex>

namespace motioncam {
namespace cuda {

namespace {

// Cached availability flag.
//   0 = unknown, not yet probed
//   1 = probed, available
//   2 = probed, NOT available
std::atomic<int> gAvailabilityCache{0};

bool ProbeCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) return false;

    // Bind to device 0. If the user has multiple GPUs we'll add a setting
    // later — for now the first one is what the NVENC encoders use too.
    err = cudaSetDevice(0);
    if (err != cudaSuccess) return false;

    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, 0);
    if (err != cudaSuccess) return false;

    return true;
}

// Placeholder kernel — empty body, takes a pointer + count so the launch
// machinery is exercised (block/grid math, parameter passing, device sync).
// Phase B replaces this with the actual RGB float -> YUV converter.
__global__ void PhaseAProbeKernel(int* /*scratch*/, int /*n*/) {
    // intentionally empty
}

}  // namespace

bool IsCudaAvailable() {
    int v = gAvailabilityCache.load(std::memory_order_acquire);
    if (v != 0) return v == 1;
    bool ok = ProbeCudaDevice();
    gAvailabilityCache.store(ok ? 1 : 2, std::memory_order_release);
    return ok;
}

bool RunPhaseAProbe() {
    if (!IsCudaAvailable()) return false;

    // Allocate a tiny scratch buffer on the device, launch the no-op
    // kernel, free. Any failure along the way returns false.
    int* d_scratch = nullptr;
    cudaError_t err = cudaMalloc(&d_scratch, sizeof(int) * 8);
    if (err != cudaSuccess) return false;

    PhaseAProbeKernel<<<1, 32>>>(d_scratch, 8);
    err = cudaDeviceSynchronize();
    cudaFree(d_scratch);
    return err == cudaSuccess;
}

// ============================================================================
// Phase B: RGB float -> NV12 (BT.709 limited, 8-bit)
// ============================================================================

namespace {

// Persistent device buffer for the H->D RGB upload. Grown on demand;
// freed via ReleaseRgbScratch() at encoder teardown. Guarded by a mutex
// because in theory two MovEncoders could share the same process — in
// practice we only have one at a time but the cost of a single lock is
// trivial vs the H->D copy that follows.
float*       gRgbScratch       = nullptr;
size_t       gRgbScratchBytes  = 0;
std::mutex   gRgbScratchMutex;

bool EnsureRgbScratch(size_t needBytes) {
    if (gRgbScratchBytes >= needBytes) return true;
    if (gRgbScratch) {
        cudaFree(gRgbScratch);
        gRgbScratch = nullptr;
        gRgbScratchBytes = 0;
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&gRgbScratch), needBytes);
    if (err != cudaSuccess) {
        gRgbScratch = nullptr;
        gRgbScratchBytes = 0;
        return false;
    }
    gRgbScratchBytes = needBytes;
    return true;
}

// BT.709 limited-range coefficients, baked at compile time. These map
// linear RGB [0,1] directly to the byte values stored in NV12:
//
//   Y'  = (0.2126*R + 0.7152*G + 0.0722*B) * 219 + 16              -> [16, 235]
//   Cb' = (-0.114572*R - 0.385428*G + 0.5*B) * 224 + 128            -> [16, 240]
//   Cr' = (0.5*R - 0.454153*G - 0.045847*B) * 224 + 128             -> [16, 240]
//
// The +0.5f added before truncation gives round-half-up, matching what
// libswscale's BT.709 fast path does.

__device__ __forceinline__ uint8_t ClampToByte(float x) {
    if (x < 0.0f) return 0;
    if (x > 255.0f) return 255;
    return static_cast<uint8_t>(x + 0.5f);
}

// One thread per Y pixel. Threads where (x even, y even) additionally
// write one chroma sample (averaging the 2x2 block of RGB above each UV
// position). Could be split into two grids — single-grid is simpler
// and the 75% chroma-no-op cost is negligible at 4K.
__global__ void RgbFloatToNv12Kernel(
    const float* __restrict__ rgb,     // width*height*3 floats interleaved
    uint8_t*     __restrict__ yPlane,
    uint8_t*     __restrict__ uvPlane,
    int width, int height,
    int yPitch, int uvPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float* px = rgb + (size_t(y) * size_t(width) + size_t(x)) * 3;
    float R = px[0];
    float G = px[1];
    float B = px[2];
    // Clamp to [0,1] BEFORE the BT.709 multiply so out-of-gamut HDR values
    // don't blow past 235.
    R = __saturatef(R);
    G = __saturatef(G);
    B = __saturatef(B);

    float Y_full =  0.2126f * R + 0.7152f * G + 0.0722f * B;
    float Y_byte = Y_full * 219.0f + 16.0f;
    yPlane[size_t(y) * size_t(yPitch) + size_t(x)] = ClampToByte(Y_byte);

    // One UV sample per 2x2 block. Only the top-left thread of each block
    // averages and stores; cuts the chroma work to 1/4.
    if ((x & 1) == 0 && (y & 1) == 0) {
        const int yx2 = (x + 1 < width)  ? x + 1 : x;
        const int yy2 = (y + 1 < height) ? y + 1 : y;
        const float* p00 = rgb + (size_t(y    ) * size_t(width) + size_t(x   )) * 3;
        const float* p01 = rgb + (size_t(y    ) * size_t(width) + size_t(yx2 )) * 3;
        const float* p10 = rgb + (size_t(yy2  ) * size_t(width) + size_t(x   )) * 3;
        const float* p11 = rgb + (size_t(yy2  ) * size_t(width) + size_t(yx2 )) * 3;
        float Ra = (p00[0] + p01[0] + p10[0] + p11[0]) * 0.25f;
        float Ga = (p00[1] + p01[1] + p10[1] + p11[1]) * 0.25f;
        float Ba = (p00[2] + p01[2] + p10[2] + p11[2]) * 0.25f;
        Ra = __saturatef(Ra);
        Ga = __saturatef(Ga);
        Ba = __saturatef(Ba);

        float Cb = -0.114572f * Ra - 0.385428f * Ga + 0.5f       * Ba;
        float Cr =  0.5f       * Ra - 0.454153f * Ga - 0.045847f * Ba;
        float Cb_byte = Cb * 224.0f + 128.0f;
        float Cr_byte = Cr * 224.0f + 128.0f;

        const int uvx = x >> 1;
        const int uvy = y >> 1;
        uint8_t* uvRow = uvPlane + size_t(uvy) * size_t(uvPitch);
        uvRow[uvx * 2 + 0] = ClampToByte(Cb_byte);
        uvRow[uvx * 2 + 1] = ClampToByte(Cr_byte);
    }
}

}  // namespace

bool RgbFloatToNv12(
    const float* rgb_host,
    void* y_device,
    void* uv_device,
    int   width,
    int   height,
    int   y_pitch_bytes,
    int   uv_pitch_bytes)
{
    if (!IsCudaAvailable() || width <= 0 || height <= 0) return false;
    if (!rgb_host || !y_device || !uv_device) return false;

    const size_t pixels   = size_t(width) * size_t(height);
    const size_t rgbBytes = pixels * 3 * sizeof(float);

    std::lock_guard<std::mutex> lock(gRgbScratchMutex);
    if (!EnsureRgbScratch(rgbBytes)) return false;

    cudaError_t err = cudaMemcpyAsync(
        gRgbScratch, rgb_host, rgbBytes, cudaMemcpyHostToDevice, 0);
    if (err != cudaSuccess) return false;

    // 32x8 block = 256 threads, decent occupancy on most arches without
    // pushing register pressure or shared-mem demands (we use neither).
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    RgbFloatToNv12Kernel<<<grid, block>>>(
        gRgbScratch,
        static_cast<uint8_t*>(y_device),
        static_cast<uint8_t*>(uv_device),
        width, height,
        y_pitch_bytes, uv_pitch_bytes);

    err = cudaGetLastError();
    if (err != cudaSuccess) return false;

    err = cudaDeviceSynchronize();
    return err == cudaSuccess;
}

void ReleaseRgbScratch() {
    std::lock_guard<std::mutex> lock(gRgbScratchMutex);
    if (gRgbScratch) {
        cudaFree(gRgbScratch);
        gRgbScratch = nullptr;
        gRgbScratchBytes = 0;
    }
}

}
}
