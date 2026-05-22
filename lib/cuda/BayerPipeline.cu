// Tier 2.1 Phase C - bayer + debayer + matrix on GPU.
//
// This file owns three kernels that together replace the CPU work in
// ColorPipeline.cpp::ProcessFrame for the common case (8-bit bayer ->
// linear RGB in a BakedTransform-compatible output space):
//
//   1. NormalizeBayerKernel   u16 raw -> float bayer (black-level, WB)
//   2. DebayerBilinearKernel  float bayer -> float RGB interleaved
//   3. ApplyMatrixCurveKernel cam-RGB -> output-RGB (matrix + opt. curve)
//
// All three are direct ports of the CPU implementations in Debayer.cpp /
// ColorPipeline.cpp / BakedTransform.cpp. Output is identical to the CPU
// path within float epsilon (verified by the Python binding in mcraw_py).
//
// Phase C.1 (this commit) leaves the result in a CPU-readable buffer so
// the Python binding can compare against the existing reference. Phase
// C.2 will hand the GPU RGB pointer directly to the existing Phase B
// RGB->NV12 kernel without a host roundtrip.

#include <motioncam/CudaHwHandoff.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>

namespace motioncam {
namespace cuda {

// ============================================================================
// Persistent device buffers (one set per process; resized on first use,
// freed via ReleaseBayerPipeline()).
// ============================================================================

namespace {

uint16_t*    gBayerU16   = nullptr;   // width*height  u16
float*       gBayerFloat = nullptr;   // width*height  float
float*       gRgbFloat   = nullptr;   // width*height*3 float
int          gBufW       = 0;
int          gBufH       = 0;
// LSM scratch: re-allocated when grid dimensions change. Tiny (a few KB
// for typical 17x13 or 65x49 grids x 4 channels) so we just realloc.
float*       gLsmDevice  = nullptr;
int          gLsmW       = 0;
int          gLsmH       = 0;
std::mutex   gBufMutex;

bool EnsureBuffers(int width, int height) {
    if (width == gBufW && height == gBufH &&
        gBayerU16 && gBayerFloat && gRgbFloat) {
        return true;
    }

    // Resize or first-allocate. Free old buffers if dimensions changed.
    if (gBayerU16)   { cudaFree(gBayerU16);   gBayerU16   = nullptr; }
    if (gBayerFloat) { cudaFree(gBayerFloat); gBayerFloat = nullptr; }
    if (gRgbFloat)   { cudaFree(gRgbFloat);   gRgbFloat   = nullptr; }
    gBufW = 0;
    gBufH = 0;

    const size_t n = size_t(width) * size_t(height);
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&gBayerU16),   n * sizeof(uint16_t));
    cudaError_t e2 = cudaMalloc(reinterpret_cast<void**>(&gBayerFloat), n * sizeof(float));
    cudaError_t e3 = cudaMalloc(reinterpret_cast<void**>(&gRgbFloat),   n * 3 * sizeof(float));
    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
        if (gBayerU16)   { cudaFree(gBayerU16);   gBayerU16   = nullptr; }
        if (gBayerFloat) { cudaFree(gBayerFloat); gBayerFloat = nullptr; }
        if (gRgbFloat)   { cudaFree(gRgbFloat);   gRgbFloat   = nullptr; }
        return false;
    }
    gBufW = width;
    gBufH = height;
    return true;
}

}  // namespace

bool SetupBayerPipeline(int width, int height) {
    if (!IsCudaAvailable() || width <= 0 || height <= 0) return false;
    std::lock_guard<std::mutex> lock(gBufMutex);
    return EnsureBuffers(width, height);
}

void ReleaseBayerPipeline() {
    std::lock_guard<std::mutex> lock(gBufMutex);
    if (gBayerU16)   { cudaFree(gBayerU16);   gBayerU16   = nullptr; }
    if (gBayerFloat) { cudaFree(gBayerFloat); gBayerFloat = nullptr; }
    if (gRgbFloat)   { cudaFree(gRgbFloat);   gRgbFloat   = nullptr; }
    if (gLsmDevice)  { cudaFree(gLsmDevice);  gLsmDevice  = nullptr; }
    gBufW = 0;
    gBufH = 0;
    gLsmW = 0;
    gLsmH = 0;
}

// ============================================================================
// Kernel 1: NormalizeBayer  -  u16 raw bayer -> float bayer in [0..1/asN_c]
// (matches Debayer.cpp::NormalizeBayer; same math the CPU version does)
// ============================================================================

// blacks[i] / scales[i] are indexed by CFA position id = (y&1)<<1 | (x&1).
// scale[i] = inv_range[i] * (1.0 / asShotNeutral[cfa_channel[i]]) so the
// kernel only multiplies once.
__global__ void NormalizeBayerKernel(
    const uint16_t* __restrict__ raw,
    float* __restrict__ out,
    int width, int height,
    float black0, float black1, float black2, float black3,
    float scale0, float scale1, float scale2, float scale3)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = ((y & 1) << 1) | (x & 1);
    float b, s;
    switch (idx) {
        case 0:  b = black0; s = scale0; break;
        case 1:  b = black1; s = scale1; break;
        case 2:  b = black2; s = scale2; break;
        default: b = black3; s = scale3; break;
    }

    float v = float(raw[size_t(y) * size_t(width) + size_t(x)]) - b;
    if (v < 0.0f) v = 0.0f;
    out[size_t(y) * size_t(width) + size_t(x)] = v * s;
}

// ============================================================================
// Kernel 2: DebayerBilinear  -  float bayer -> float RGB (interleaved)
// (matches Debayer.cpp::DebayerBilinear; same bilinear interpolation rules)
// ============================================================================

__device__ __forceinline__ float SampleClamped(
    const float* bayer, int xx, int yy, int width, int height)
{
    if (xx < 0) xx = 0; else if (xx >= width)  xx = width  - 1;
    if (yy < 0) yy = 0; else if (yy >= height) yy = height - 1;
    return bayer[size_t(yy) * size_t(width) + size_t(xx)];
}

__global__ void DebayerBilinearKernel(
    const float* __restrict__ bayer,
    float* __restrict__ rgb,            // width*height*3 interleaved
    int width, int height,
    int ch0, int ch1, int ch2, int ch3)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = ((y & 1) << 1) | (x & 1);
    int c;
    switch (idx) {
        case 0:  c = ch0; break;
        case 1:  c = ch1; break;
        case 2:  c = ch2; break;
        default: c = ch3; break;
    }

    const float self = bayer[size_t(y) * size_t(width) + size_t(x)];
    float r, g, b;

    if (c == 0) {  // R-position
        r = self;
        g = 0.25f * (SampleClamped(bayer, x-1, y,   width, height) +
                     SampleClamped(bayer, x+1, y,   width, height) +
                     SampleClamped(bayer, x,   y-1, width, height) +
                     SampleClamped(bayer, x,   y+1, width, height));
        b = 0.25f * (SampleClamped(bayer, x-1, y-1, width, height) +
                     SampleClamped(bayer, x+1, y-1, width, height) +
                     SampleClamped(bayer, x-1, y+1, width, height) +
                     SampleClamped(bayer, x+1, y+1, width, height));
    } else if (c == 2) {  // B-position
        b = self;
        g = 0.25f * (SampleClamped(bayer, x-1, y,   width, height) +
                     SampleClamped(bayer, x+1, y,   width, height) +
                     SampleClamped(bayer, x,   y-1, width, height) +
                     SampleClamped(bayer, x,   y+1, width, height));
        r = 0.25f * (SampleClamped(bayer, x-1, y-1, width, height) +
                     SampleClamped(bayer, x+1, y-1, width, height) +
                     SampleClamped(bayer, x-1, y+1, width, height) +
                     SampleClamped(bayer, x+1, y+1, width, height));
    } else {  // G-position
        g = self;
        // Identify horizontal neighbour's CFA channel (same logic as
        // Debayer.cpp): if it's R, the row carries R-on-horiz / B-on-vert;
        // if it's B, the row carries B-on-horiz / R-on-vert.
        const int h_idx = ((y & 1) << 1) | ((x + 1) & 1);
        int h_c;
        switch (h_idx) {
            case 0:  h_c = ch0; break;
            case 1:  h_c = ch1; break;
            case 2:  h_c = ch2; break;
            default: h_c = ch3; break;
        }
        if (h_c == 0) {
            r = 0.5f * (SampleClamped(bayer, x-1, y, width, height) +
                        SampleClamped(bayer, x+1, y, width, height));
            b = 0.5f * (SampleClamped(bayer, x,   y-1, width, height) +
                        SampleClamped(bayer, x,   y+1, width, height));
        } else {
            b = 0.5f * (SampleClamped(bayer, x-1, y, width, height) +
                        SampleClamped(bayer, x+1, y, width, height));
            r = 0.5f * (SampleClamped(bayer, x,   y-1, width, height) +
                        SampleClamped(bayer, x,   y+1, width, height));
        }
    }

    const size_t o = (size_t(y) * size_t(width) + size_t(x)) * 3;
    rgb[o + 0] = r;
    rgb[o + 1] = g;
    rgb[o + 2] = b;
}

// ============================================================================
// Kernel 3: ApplyMatrixCurve  -  in-place 3x3 matrix mul + optional curve
// (matches BakedTransform.cpp::ApplyBakedTransform; same math)
// ============================================================================

// Same piecewise math as BakedTransform.cpp::ApplyCurveScalar — kept
// here so the kernel is self-contained. Sign-flipped via an explicit
// abs+copysign so ptxas doesn't see "recursion" and complain about
// unbounded stack.
__device__ __forceinline__ float ApplyCurveDevice(int curve, float v) {
    if (curve == 0) return v;                                // None
    const float sign = (v < 0.0f) ? -1.0f : 1.0f;
    const float a    = (v < 0.0f) ? -v    : v;
    float r;
    switch (curve) {
        case 1: r = powf(a, 1.0f / 2.2f); break;             // Gamma22
        case 2: r = powf(a, 1.0f / 2.4f); break;             // Gamma24
        case 3:                                              // SRGB piecewise
            r = (a <= 0.0031308f)
                ? 12.92f * a
                : 1.055f * powf(a, 1.0f / 2.4f) - 0.055f;
            break;
        default: r = a; break;
    }
    return sign * r;
}

__global__ void ApplyMatrixCurveKernel(
    float* __restrict__ rgb,
    int width, int height,
    float m00, float m01, float m02,
    float m10, float m11, float m12,
    float m20, float m21, float m22,
    int curve)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t o = (size_t(y) * size_t(width) + size_t(x)) * 3;
    const float r = rgb[o + 0];
    const float g = rgb[o + 1];
    const float b = rgb[o + 2];

    float nr = m00 * r + m01 * g + m02 * b;
    float ng = m10 * r + m11 * g + m12 * b;
    float nb = m20 * r + m21 * g + m22 * b;

    if (curve != 0) {
        nr = ApplyCurveDevice(curve, nr);
        ng = ApplyCurveDevice(curve, ng);
        nb = ApplyCurveDevice(curve, nb);
    }

    rgb[o + 0] = nr;
    rgb[o + 1] = ng;
    rgb[o + 2] = nb;
}

// ============================================================================
// Kernel 1b: ApplyLensShading  -  per-pixel bilinear LSM gain multiply
// (matches Debayer.cpp::ApplyLensShading; same channel-first layout)
//
// Runs in-place on the normalised bayer buffer between NormalizeBayer and
// DebayerBilinear, mirroring the CPU order. Only applied when the frame
// metadata carries a lens shading map (typical for MotionCam clips).
// ============================================================================

__global__ void ApplyLensShadingKernel(
    float* __restrict__ bayer,
    int width, int height,
    const float* __restrict__ lsm,        // 4 * lsmW * lsmH floats, channel-first
    int lsmW, int lsmH,
    int cfa0, int cfa1, int cfa2, int cfa3)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float invXMax = 1.0f / float(width  - 1);
    const float invYMax = 1.0f / float(height - 1);
    const int lsmWMinus = lsmW - 1;
    const int lsmHMinus = lsmH - 1;
    const size_t perCh  = size_t(lsmW) * size_t(lsmH);

    const float fy = float(y) * invYMax * float(lsmHMinus);
    int yi = int(fy);
    if (yi >= lsmHMinus) yi = lsmHMinus - 1;
    const float fyf = fy - float(yi);

    const float fx = float(x) * invXMax * float(lsmWMinus);
    int xi = int(fx);
    if (xi >= lsmWMinus) xi = lsmWMinus - 1;
    const float fxf = fx - float(xi);

    const int idx = ((y & 1) << 1) | (x & 1);
    int ch;
    switch (idx) {
        case 0:  ch = cfa0; break;
        case 1:  ch = cfa1; break;
        case 2:  ch = cfa2; break;
        default: ch = cfa3; break;
    }

    const float* chPlane = lsm + size_t(ch) * perCh;
    const float v00 = chPlane[size_t(yi)     * size_t(lsmW) + size_t(xi)];
    const float v10 = chPlane[size_t(yi)     * size_t(lsmW) + size_t(xi + 1)];
    const float v01 = chPlane[size_t(yi + 1) * size_t(lsmW) + size_t(xi)];
    const float v11 = chPlane[size_t(yi + 1) * size_t(lsmW) + size_t(xi + 1)];

    const float v0 = v00 * (1.0f - fxf) + v10 * fxf;
    const float v1 = v01 * (1.0f - fxf) + v11 * fxf;
    const float gain = v0 * (1.0f - fyf) + v1 * fyf;

    bayer[size_t(y) * size_t(width) + size_t(x)] *= gain;
}

// ============================================================================
// Phase C orchestrator
// ============================================================================

bool ProcessBayerToRgb(
    const uint16_t* bayer_host,
    const float wb[3],
    const BayerPipelineConstants& C,
    float* rgb_host_out)
{
    if (!IsCudaAvailable()) return false;
    if (!bayer_host || !rgb_host_out || !wb) return false;
    if (C.width <= 0 || C.height <= 0) return false;

    const int W = C.width;
    const int H = C.height;
    const size_t n = size_t(W) * size_t(H);

    std::lock_guard<std::mutex> lock(gBufMutex);
    if (!EnsureBuffers(W, H)) return false;

    // 1. Upload bayer u16
    cudaError_t err = cudaMemcpyAsync(
        gBayerU16, bayer_host, n * sizeof(uint16_t),
        cudaMemcpyHostToDevice, 0);
    if (err != cudaSuccess) return false;

    // 2. Combine per-CFA-position normalisation constants. scale[i] folds
    //    both the dynamic-range normalise and the WB multiply into one mul,
    //    matching what Debayer.cpp::NormalizeBayer does on the CPU.
    float blacks[4];
    float scales[4];
    for (int i = 0; i < 4; ++i) {
        const float invWb_c = 1.0f / wb[C.cfa_channel[i]];
        blacks[i] = float(C.black[i]);
        scales[i] = C.inv_range[i] * invWb_c;
    }

    dim3 block(32, 8);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    NormalizeBayerKernel<<<grid, block>>>(
        gBayerU16, gBayerFloat,
        W, H,
        blacks[0], blacks[1], blacks[2], blacks[3],
        scales[0], scales[1], scales[2], scales[3]);
    err = cudaGetLastError();
    if (err != cudaSuccess) return false;

    // 2b. Optional Lens Shading Map. Uploaded each call - tiny (~3-15 KB)
    //     so we don't bother caching unless profiling shows it matters.
    if (C.lsm_w >= 2 && C.lsm_h >= 2 && C.lsm_host) {
        const size_t lsmBytes = size_t(C.lsm_w) * size_t(C.lsm_h) * 4 * sizeof(float);
        if (C.lsm_w != gLsmW || C.lsm_h != gLsmH || gLsmDevice == nullptr) {
            if (gLsmDevice) { cudaFree(gLsmDevice); gLsmDevice = nullptr; }
            err = cudaMalloc(reinterpret_cast<void**>(&gLsmDevice), lsmBytes);
            if (err != cudaSuccess) return false;
            gLsmW = C.lsm_w;
            gLsmH = C.lsm_h;
        }
        err = cudaMemcpyAsync(gLsmDevice, C.lsm_host, lsmBytes,
                              cudaMemcpyHostToDevice, 0);
        if (err != cudaSuccess) return false;

        ApplyLensShadingKernel<<<grid, block>>>(
            gBayerFloat, W, H,
            gLsmDevice, C.lsm_w, C.lsm_h,
            C.cfa_to_lsm[0], C.cfa_to_lsm[1],
            C.cfa_to_lsm[2], C.cfa_to_lsm[3]);
        err = cudaGetLastError();
        if (err != cudaSuccess) return false;
    }

    // 3. Bilinear debayer.
    DebayerBilinearKernel<<<grid, block>>>(
        gBayerFloat, gRgbFloat,
        W, H,
        C.cfa_channel[0], C.cfa_channel[1],
        C.cfa_channel[2], C.cfa_channel[3]);
    err = cudaGetLastError();
    if (err != cudaSuccess) return false;

    // 4. cam-RGB -> output-RGB in one matrix mul, plus optional curve.
    ApplyMatrixCurveKernel<<<grid, block>>>(
        gRgbFloat,
        W, H,
        C.cam_to_output[0], C.cam_to_output[1], C.cam_to_output[2],
        C.cam_to_output[3], C.cam_to_output[4], C.cam_to_output[5],
        C.cam_to_output[6], C.cam_to_output[7], C.cam_to_output[8],
        C.curve);
    err = cudaGetLastError();
    if (err != cudaSuccess) return false;

    // 5. Copy result to host for verification. Phase C.2 removes this and
    //    feeds gRgbFloat straight into the Phase B RGB->NV12 kernel.
    err = cudaMemcpy(
        rgb_host_out, gRgbFloat,
        n * 3 * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return false;

    err = cudaDeviceSynchronize();
    return err == cudaSuccess;
}

}
}
