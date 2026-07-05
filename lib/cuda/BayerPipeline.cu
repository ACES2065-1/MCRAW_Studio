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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

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
uint8_t*     gRgb8       = nullptr;   // width*height*3 u8 (preview readback)
size_t       gRgb8Bytes  = 0;
int          gBufW       = 0;
int          gBufH       = 0;
// LSM scratch: re-allocated when grid dimensions change. Tiny (a few KB
// for typical 17x13 or 65x49 grids x 4 channels) so we just realloc.
float*       gLsmDevice  = nullptr;
int          gLsmW       = 0;
int          gLsmH       = 0;
std::mutex   gBufMutex;

// Phase D: OCIO 3D LUT, uploaded once per clip via SetupLut3D(). Stored as a
// float4 cudaArray sampled with hardware trilinear filtering. The B axis is
// the array's fastest-varying dimension (matches the host bake layout), so
// the kernel samples tex3D(lut, tcB, tcG, tcR).
cudaArray_t         gLutArray = nullptr;
cudaTextureObject_t gLutTex   = 0;
int                 gLutN     = 0;
bool                gLutValid = false;

void ReleaseLut3DLocked() {
    if (gLutTex)   { cudaDestroyTextureObject(gLutTex); gLutTex = 0; }
    if (gLutArray) { cudaFreeArray(gLutArray);          gLutArray = nullptr; }
    gLutN = 0;
    gLutValid = false;
}

// Phase F: denoise scratch (output-space Y/Cb/Cr planes + one shared tmp, plus
// a tiny device buffer for the separable Gaussian weights). Allocated lazily
// when a denoise render first runs; sized to width*height.
float*  gDnY       = nullptr;
float*  gDnCb      = nullptr;
float*  gDnCr      = nullptr;
float*  gDnTmp     = nullptr;
float*  gDnWeights = nullptr;       // up to 65 Gaussian taps
size_t  gDnN       = 0;
constexpr int kDnMaxRadius = 32;    // matches Denoise.cpp's cap

void ReleaseDenoiseLocked() {
    if (gDnY)       { cudaFree(gDnY);       gDnY = nullptr; }
    if (gDnCb)      { cudaFree(gDnCb);      gDnCb = nullptr; }
    if (gDnCr)      { cudaFree(gDnCr);      gDnCr = nullptr; }
    if (gDnTmp)     { cudaFree(gDnTmp);     gDnTmp = nullptr; }
    if (gDnWeights) { cudaFree(gDnWeights); gDnWeights = nullptr; }
    gDnN = 0;
}

bool EnsureDenoiseBuffers(size_t n) {
    if (n == gDnN && gDnY && gDnCb && gDnCr && gDnTmp && gDnWeights) return true;
    ReleaseDenoiseLocked();
    const size_t bytes = n * sizeof(float);
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&gDnY),   bytes);
    cudaError_t e2 = cudaMalloc(reinterpret_cast<void**>(&gDnCb),  bytes);
    cudaError_t e3 = cudaMalloc(reinterpret_cast<void**>(&gDnCr),  bytes);
    cudaError_t e4 = cudaMalloc(reinterpret_cast<void**>(&gDnTmp), bytes);
    cudaError_t e5 = cudaMalloc(reinterpret_cast<void**>(&gDnWeights),
                                size_t(kDnMaxRadius * 2 + 1) * sizeof(float));
    if (e1 || e2 || e3 || e4 || e5) { ReleaseDenoiseLocked(); return false; }
    gDnN = n;
    return true;
}

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
    if (gRgb8)       { cudaFree(gRgb8);        gRgb8       = nullptr; gRgb8Bytes = 0; }
    if (gLsmDevice)  { cudaFree(gLsmDevice);  gLsmDevice  = nullptr; }
    ReleaseDenoiseLocked();
    gBufW = 0;
    gBufH = 0;
    gLsmW = 0;
    gLsmH = 0;
}

// ============================================================================
// Phase D: 3D LUT upload / teardown
// ============================================================================

bool SetupLut3D(const float* lut_rgba_host, int n) {
    if (!IsCudaAvailable() || !lut_rgba_host || n < 2) return false;
    std::lock_guard<std::mutex> lock(gBufMutex);

    ReleaseLut3DLocked();

    cudaChannelFormatDesc ch = cudaCreateChannelDesc<float4>();
    cudaExtent ext = make_cudaExtent(n, n, n);   // dims: B (fastest), G, R
    cudaError_t err = cudaMalloc3DArray(&gLutArray, &ch, ext, 0);
    if (err != cudaSuccess) { gLutArray = nullptr; return false; }

    cudaMemcpy3DParms cp = {};
    cp.srcPtr = make_cudaPitchedPtr(
        const_cast<float*>(lut_rgba_host),
        size_t(n) * sizeof(float4),   // row pitch (B axis * float4)
        size_t(n),                    // width  in elements (B)
        size_t(n));                   // height in rows     (G)
    cp.dstArray = gLutArray;
    cp.extent   = ext;
    cp.kind     = cudaMemcpyHostToDevice;
    err = cudaMemcpy3D(&cp);
    if (err != cudaSuccess) { ReleaseLut3DLocked(); return false; }

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = gLutArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;   // hardware trilinear
    texDesc.readMode         = cudaReadModeElementType; // float, no normalize
    texDesc.normalizedCoords = 1;

    err = cudaCreateTextureObject(&gLutTex, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) { ReleaseLut3DLocked(); return false; }

    gLutN = n;
    gLutValid = true;
    return true;
}

bool HasLut3D() {
    std::lock_guard<std::mutex> lock(gBufMutex);
    return gLutValid;
}

void ReleaseLut3D() {
    std::lock_guard<std::mutex> lock(gBufMutex);
    ReleaseLut3DLocked();
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
// Kernel 4 (Phase D): ApplyLut3D  -  in-place ACEScg -> target via 3D LUT
//
// Replaces the curve step for OCIO-only targets. Input is ACEScg scene-linear
// (cam_to_output was cam->ACEScg, curve None). Each channel is run through the
// asinh shaper to get a [0,1] cube coordinate, then the float4 LUT is sampled
// with hardware trilinear filtering. The B axis is the cube's fastest-varying
// dimension, so the fetch order is tex3D(lut, tcB, tcG, tcR).
//
// shaperK / sLo / invSpan are precomputed on the host from kLutShaper*:
//   sLo = asinh(lo/K),  invSpan = 1 / (asinh(hi/K) - sLo)
// ============================================================================

__device__ __forceinline__ float LinToShaperT(float L, float K, float sLo, float invSpan) {
    float t = (asinhf(L / K) - sLo) * invSpan;
    return fminf(fmaxf(t, 0.0f), 1.0f);
}

__global__ void ApplyLut3DKernel(
    float* __restrict__ rgb,
    int width, int height,
    cudaTextureObject_t lut, int n,
    float shaperK, float sLo, float invSpan)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t o = (size_t(y) * size_t(width) + size_t(x)) * 3;
    const float R = rgb[o + 0];
    const float G = rgb[o + 1];
    const float B = rgb[o + 2];

    const float tR = LinToShaperT(R, shaperK, sLo, invSpan);
    const float tG = LinToShaperT(G, shaperK, sLo, invSpan);
    const float tB = LinToShaperT(B, shaperK, sLo, invSpan);

    // [0,1] grid coordinate -> normalized texture coordinate with the
    // half-texel offset so t = i/(n-1) lands on texel center (i+0.5)/n.
    const float scale = float(n - 1) / float(n);
    const float bias  = 0.5f / float(n);
    const float u = tB * scale + bias;   // array dim 0 (fastest) = B
    const float v = tG * scale + bias;   // array dim 1            = G
    const float w = tR * scale + bias;   // array dim 2 (slowest)  = R

    const float4 c = tex3D<float4>(lut, u, v, w);
    rgb[o + 0] = c.x;
    rgb[o + 1] = c.y;
    rgb[o + 2] = c.z;
}

// ============================================================================
// Kernel 5 (Phase E.3): NeutraliseClippedHighlights  -  pre-matrix de-magenta
// (matches Debayer.cpp::NeutraliseClippedHighlights; cam-RGB, WB-applied)
// ============================================================================

__global__ void NeutraliseClippedHighlightsKernel(
    float* __restrict__ rgb,
    int width, int height,
    float wb0, float wb1, float wb2)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t o = (size_t(y) * size_t(width) + size_t(x)) * 3;
    const float r = rgb[o + 0];
    const float g = rgb[o + 1];
    const float b = rgb[o + 2];

    const float rn = r * wb0;
    const float gn = g * wb1;
    const float bn = b * wb2;
    const float maxN = fmaxf(rn, fmaxf(gn, bn));
    const float minN = fminf(rn, fminf(gn, bn));

    float t_max = (maxN - 0.5f) * 2.0f;           // smoothstep(0.5, 1.0, maxN)
    if (t_max <= 0.0f) return;
    if (t_max > 1.0f) t_max = 1.0f;
    t_max = t_max * t_max * (3.0f - 2.0f * t_max);

    float t_gate = (minN - 0.2f) * 5.0f;          // smoothstep(0.2, 0.4, minN)
    if (t_gate <= 0.0f) return;
    if (t_gate > 1.0f) t_gate = 1.0f;
    t_gate = t_gate * t_gate * (3.0f - 2.0f * t_gate);

    const float t = t_max * t_gate;
    const float maxCh = fmaxf(r, fmaxf(g, b));
    rgb[o + 0] = r + (maxCh - r) * t;
    rgb[o + 1] = g + (maxCh - g) * t;
    rgb[o + 2] = b + (maxCh - b) * t;
}

// ============================================================================
// Kernel 6 (Phase E.3): HighlightRolloff  -  post-transform shoulder
// (matches Debayer.cpp::HighlightRolloff; display-encoded outputs only)
// ============================================================================

__global__ void HighlightRolloffKernel(
    float* __restrict__ rgb,
    int width, int height,
    float kneeStart, float kneeEnd, float invKneeRange)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t o = (size_t(y) * size_t(width) + size_t(x)) * 3;
    float r = rgb[o + 0];
    float g = rgb[o + 1];
    float b = rgb[o + 2];
    const float maxCh = fmaxf(r, fmaxf(g, b));
    if (maxCh <= kneeStart) return;

    float t = (maxCh - kneeStart) * invKneeRange;
    if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
    t = t * t * (3.0f - 2.0f * t);
    r = r + (maxCh - r) * t;
    g = g + (maxCh - g) * t;
    b = b + (maxCh - b) * t;
    if (r > kneeEnd) r = kneeEnd;
    if (g > kneeEnd) g = kneeEnd;
    if (b > kneeEnd) b = kneeEnd;
    rgb[o + 0] = r;
    rgb[o + 1] = g;
    rgb[o + 2] = b;
}

// ============================================================================
// Kernels 7-10 (Phase F): output-space denoise
// (matches Denoise.cpp::DenoiseRgb — same Y/Cb=B-Y/Cr=R-Y decomposition,
//  separable Gaussian on chroma, 5x5 bilateral on luma, recombine.)
// ============================================================================

namespace {
constexpr float kDnRy = 0.2126f;
constexpr float kDnGy = 0.7152f;
constexpr float kDnBy = 0.0722f;
}

__global__ void SplitYCbCrKernel(
    const float* __restrict__ rgb,
    float* __restrict__ Y, float* __restrict__ Cb, float* __restrict__ Cr,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const size_t i = size_t(y) * size_t(width) + size_t(x);
    const float r = rgb[i*3 + 0], g = rgb[i*3 + 1], b = rgb[i*3 + 2];
    const float yv = kDnRy * r + kDnGy * g + kDnBy * b;
    Y[i]  = yv;
    Cb[i] = b - yv;
    Cr[i] = r - yv;
}

// Separable Gaussian, one axis per launch. horizontal != 0 -> blur along x.
// Clamps at the border (matches the CPU edge handling).
__global__ void GaussianBlurAxisKernel(
    const float* __restrict__ src, float* __restrict__ dst,
    int width, int height,
    const float* __restrict__ w, int radius, int horizontal)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int t = -radius; t <= radius; ++t) {
        int xi = x, yi = y;
        if (horizontal) { xi = x + t; if (xi < 0) xi = 0; else if (xi >= width)  xi = width  - 1; }
        else            { yi = y + t; if (yi < 0) yi = 0; else if (yi >= height) yi = height - 1; }
        sum += src[size_t(yi) * size_t(width) + size_t(xi)] * w[t + radius];
    }
    dst[size_t(y) * size_t(width) + size_t(x)] = sum;
}

// 5x5 edge-preserving bilateral on a single plane. Spatial + range weights
// computed directly (the CPU LUTs the range; direct expf is marginally more
// accurate and the difference is far below a code value).
__global__ void Bilateral5x5Kernel(
    const float* __restrict__ src, float* __restrict__ dst,
    int width, int height, float invSp2, float invRn2)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float c = src[size_t(y) * size_t(width) + size_t(x)];
    float wsum = 0.0f, vsum = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        int yi = y + dy;
        if (yi < 0) yi = 0; else if (yi >= height) yi = height - 1;
        for (int dx = -2; dx <= 2; ++dx) {
            int xi = x + dx;
            if (xi < 0) xi = 0; else if (xi >= width) xi = width - 1;
            const float v = src[size_t(yi) * size_t(width) + size_t(xi)];
            float dv = v - c;
            const float ws = __expf(-float(dx*dx + dy*dy) * invSp2);
            const float wr = __expf(-dv * dv * invRn2);
            const float wgt = ws * wr;
            wsum += wgt;
            vsum += wgt * v;
        }
    }
    dst[size_t(y) * size_t(width) + size_t(x)] = (wsum > 0.0f) ? (vsum / wsum) : c;
}

__global__ void CombineYCbCrKernel(
    float* __restrict__ rgb,
    const float* __restrict__ Y, const float* __restrict__ Cb, const float* __restrict__ Cr,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const size_t i = size_t(y) * size_t(width) + size_t(x);
    const float yv = Y[i], cb = Cb[i], cr = Cr[i];
    const float r = yv + cr;
    const float b = yv + cb;
    const float g = (yv - kDnRy * r - kDnBy * b) / kDnGy;
    rgb[i*3 + 0] = r;
    rgb[i*3 + 1] = g;
    rgb[i*3 + 2] = b;
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

// Phase D: optional ACEScg->target 3D LUT after the matrix. Caller must hold
// gBufMutex and have just left ACEScg in gRgbFloat. Returns false only on a
// kernel launch error; a no-op (LUT disabled / not loaded) returns true.
namespace {
bool MaybeApplyLut3D(const BayerPipelineConstants& C, int W, int H,
                     dim3 grid, dim3 block) {
    if (!C.use_lut3d || !gLutValid) return true;
    const float K       = kLutShaperK;
    const float sLo     = std::asinh(kLutShaperLo / K);
    const float sHi     = std::asinh(kLutShaperHi / K);
    const float invSpan = 1.0f / (sHi - sLo);
    ApplyLut3DKernel<<<grid, block>>>(gRgbFloat, W, H, gLutTex, gLutN,
                                      K, sLo, invSpan);
    return cudaGetLastError() == cudaSuccess;
}

// Host copy of Denoise.cpp::GaussianKernel — builds a normalised 1D Gaussian
// covering ~3 sigma each side (radius capped at kDnMaxRadius). Returns the
// weights and sets `radius`. radius 0 (sigma < 0.05) means a no-op {1.0}.
std::vector<float> BuildGaussianKernel(float sigma, int& radius) {
    if (sigma < 0.05f) { radius = 0; return {1.0f}; }
    radius = std::max(1, int(std::ceil(sigma * 3.0f)));
    if (radius > kDnMaxRadius) radius = kDnMaxRadius;
    std::vector<float> k(size_t(radius) * 2 + 1);
    const float inv2sig2 = 1.0f / (2.0f * sigma * sigma);
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        float v = std::exp(-float(i * i) * inv2sig2);
        k[size_t(i + radius)] = v;
        sum += v;
    }
    const float invSum = 1.0f / sum;
    for (auto& v : k) v *= invSum;
    return k;
}

// Phase F: output-space denoise on gRgbFloat. Caller holds gBufMutex and has
// left output-space RGB in gRgbFloat (after matrix/LUT/rolloff). No-op when
// both strengths are 0. Returns false on any launch/alloc error.
bool RunDenoiseLocked(const BayerPipelineConstants& C, int W, int H,
                      dim3 grid, dim3 block) {
    int chroma = C.denoise_chroma, luma = C.denoise_luma;
    if (chroma <= 0 && luma <= 0) return true;
    if (chroma < 0) chroma = 0; else if (chroma > 100) chroma = 100;
    if (luma   < 0) luma   = 0; else if (luma   > 100) luma   = 100;
    if (W < 3 || H < 3) return true;

    if (!EnsureDenoiseBuffers(size_t(W) * size_t(H))) return false;

    SplitYCbCrKernel<<<grid, block>>>(gRgbFloat, gDnY, gDnCb, gDnCr, W, H);
    if (cudaGetLastError() != cudaSuccess) return false;

    if (chroma > 0) {
        const float sigma = float(chroma) * 0.04f;
        int radius = 0;
        std::vector<float> k = BuildGaussianKernel(sigma, radius);
        if (radius > 0) {
            if (cudaMemcpyAsync(gDnWeights, k.data(), k.size() * sizeof(float),
                                cudaMemcpyHostToDevice, 0) != cudaSuccess)
                return false;
            // Cb: H -> tmp, V -> Cb.  Cr: H -> tmp, V -> Cr.
            GaussianBlurAxisKernel<<<grid, block>>>(gDnCb, gDnTmp, W, H, gDnWeights, radius, 1);
            GaussianBlurAxisKernel<<<grid, block>>>(gDnTmp, gDnCb, W, H, gDnWeights, radius, 0);
            GaussianBlurAxisKernel<<<grid, block>>>(gDnCr, gDnTmp, W, H, gDnWeights, radius, 1);
            GaussianBlurAxisKernel<<<grid, block>>>(gDnTmp, gDnCr, W, H, gDnWeights, radius, 0);
            if (cudaGetLastError() != cudaSuccess) return false;
        }
    }

    const float* Yptr = gDnY;
    if (luma > 0) {
        const float ssig = float(luma) * 0.025f;
        const float rsig = 0.005f + float(luma) * 0.0005f;
        if (ssig >= 0.05f) {  // matches Bilateral5x5's early-out (memcpy)
            const float invSp2 = 1.0f / (2.0f * ssig * ssig);
            const float invRn2 = 1.0f / (2.0f * rsig * rsig);
            Bilateral5x5Kernel<<<grid, block>>>(gDnY, gDnTmp, W, H, invSp2, invRn2);
            if (cudaGetLastError() != cudaSuccess) return false;
            Yptr = gDnTmp;  // denoised luma now in tmp
        }
    }

    CombineYCbCrKernel<<<grid, block>>>(gRgbFloat, Yptr, gDnCb, gDnCr, W, H);
    return cudaGetLastError() == cudaSuccess;
}
}  // namespace

// Shared bayer->ACEScg/target chain: upload -> normalise -> (LSM) -> debayer
// -> matrix(+curve) -> (3D LUT). Leaves the result in gRgbFloat. Caller must
// hold gBufMutex. Does NOT synchronise — the tail step (host copy / NV12 /
// P010) handles that. Returns false on any launch/copy error.
namespace {
bool RunBayerChainLocked(const uint16_t* bayer_host, const float wb[3],
                         const BayerPipelineConstants& C, int W, int H) {
    const size_t n = size_t(W) * size_t(H);

    cudaError_t err = cudaMemcpyAsync(
        gBayerU16, bayer_host, n * sizeof(uint16_t),
        cudaMemcpyHostToDevice, 0);
    if (err != cudaSuccess) return false;

    // Fold dynamic-range normalise + WB multiply into one scale per CFA
    // position (matches Debayer.cpp::NormalizeBayer).
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
        gBayerU16, gBayerFloat, W, H,
        blacks[0], blacks[1], blacks[2], blacks[3],
        scales[0], scales[1], scales[2], scales[3]);
    if (cudaGetLastError() != cudaSuccess) return false;

    // Optional lens shading map (uploaded each call - tiny).
    if (C.lsm_w >= 2 && C.lsm_h >= 2 && C.lsm_host) {
        const size_t lsmBytes = size_t(C.lsm_w) * size_t(C.lsm_h) * 4 * sizeof(float);
        if (C.lsm_w != gLsmW || C.lsm_h != gLsmH || gLsmDevice == nullptr) {
            if (gLsmDevice) { cudaFree(gLsmDevice); gLsmDevice = nullptr; }
            if (cudaMalloc(reinterpret_cast<void**>(&gLsmDevice), lsmBytes) != cudaSuccess)
                return false;
            gLsmW = C.lsm_w;
            gLsmH = C.lsm_h;
        }
        if (cudaMemcpyAsync(gLsmDevice, C.lsm_host, lsmBytes,
                            cudaMemcpyHostToDevice, 0) != cudaSuccess)
            return false;

        ApplyLensShadingKernel<<<grid, block>>>(
            gBayerFloat, W, H,
            gLsmDevice, C.lsm_w, C.lsm_h,
            C.cfa_to_lsm[0], C.cfa_to_lsm[1],
            C.cfa_to_lsm[2], C.cfa_to_lsm[3]);
        if (cudaGetLastError() != cudaSuccess) return false;
    }

    DebayerBilinearKernel<<<grid, block>>>(
        gBayerFloat, gRgbFloat, W, H,
        C.cfa_channel[0], C.cfa_channel[1],
        C.cfa_channel[2], C.cfa_channel[3]);
    if (cudaGetLastError() != cudaSuccess) return false;

    // Phase E.3: pre-matrix highlight recovery (cam-RGB, WB-applied).
    if (C.highlight_recovery) {
        NeutraliseClippedHighlightsKernel<<<grid, block>>>(
            gRgbFloat, W, H, wb[0], wb[1], wb[2]);
        if (cudaGetLastError() != cudaSuccess) return false;
    }

    ApplyMatrixCurveKernel<<<grid, block>>>(
        gRgbFloat, W, H,
        C.cam_to_output[0], C.cam_to_output[1], C.cam_to_output[2],
        C.cam_to_output[3], C.cam_to_output[4], C.cam_to_output[5],
        C.cam_to_output[6], C.cam_to_output[7], C.cam_to_output[8],
        C.curve);
    if (cudaGetLastError() != cudaSuccess) return false;

    // Phase D: optional ACEScg->target 3D LUT (OCIO targets).
    if (!MaybeApplyLut3D(C, W, H, grid, block)) return false;

    // Phase E.3: post-transform highlight rolloff (display-encoded targets).
    if (C.highlight_recovery && C.highlight_rolloff) {
        const float kneeStart = 1.0f, kneeEnd = 1.4f;
        HighlightRolloffKernel<<<grid, block>>>(
            gRgbFloat, W, H, kneeStart, kneeEnd, 1.0f / (kneeEnd - kneeStart));
        if (cudaGetLastError() != cudaSuccess) return false;
    }

    // Phase F: output-space denoise (chroma Gaussian + luma bilateral).
    return RunDenoiseLocked(C, W, H, grid, block);
}
}  // namespace

bool ProcessBayerToRgb(
    const uint16_t* bayer_host,
    const float wb[3],
    const BayerPipelineConstants& C,
    float* rgb_host_out)
{
    if (!IsCudaAvailable()) return false;
    if (!bayer_host || !rgb_host_out || !wb) return false;
    if (C.width <= 0 || C.height <= 0) return false;

    const int W = C.width, H = C.height;
    const size_t n = size_t(W) * size_t(H);

    std::lock_guard<std::mutex> lock(gBufMutex);
    if (!EnsureBuffers(W, H)) return false;
    if (!RunBayerChainLocked(bayer_host, wb, C, W, H)) return false;

    // Copy the result to host for the Python correctness bindings.
    if (cudaMemcpy(rgb_host_out, gRgbFloat, n * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    return cudaDeviceSynchronize() == cudaSuccess;
}

// Preview: clamp float RGB [0,1] -> uint8 RGB888 (n = pixels*3 components).
__global__ void RgbFloatToU8Kernel(const float* __restrict__ rgb,
                                   uint8_t* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = rgb[i];
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    out[i] = static_cast<uint8_t>(v * 255.0f + 0.5f);
}

bool ProcessBayerToRgb8(
    const uint16_t* bayer_host,
    const float wb[3],
    const BayerPipelineConstants& C,
    uint8_t* rgb8_host_out)
{
    // Preview fast path: full bayer chain on GPU, then clamp to 8-bit RGB888
    // and read back (≈ width*height*3 bytes — far less than the float buffer).
    if (!IsCudaAvailable()) return false;
    if (!bayer_host || !rgb8_host_out || !wb) return false;
    if (C.width <= 0 || C.height <= 0) return false;

    const int W = C.width, H = C.height;
    const size_t n = size_t(W) * size_t(H);
    const size_t bytes = n * 3;

    std::lock_guard<std::mutex> lock(gBufMutex);
    if (!EnsureBuffers(W, H)) return false;
    if (gRgb8Bytes != bytes || !gRgb8) {
        if (gRgb8) { cudaFree(gRgb8); gRgb8 = nullptr; }
        if (cudaMalloc(reinterpret_cast<void**>(&gRgb8), bytes) != cudaSuccess) {
            gRgb8 = nullptr; gRgb8Bytes = 0; return false;
        }
        gRgb8Bytes = bytes;
    }
    if (!RunBayerChainLocked(bayer_host, wb, C, W, H)) return false;

    const int total = int(bytes);
    const int block = 256;
    const int grid = (total + block - 1) / block;
    RgbFloatToU8Kernel<<<grid, block>>>(gRgbFloat, gRgb8, total);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (cudaMemcpy(rgb8_host_out, gRgb8, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool ProcessBayerToNv12(
    const uint16_t* bayer_host,
    const float wb[3],
    const BayerPipelineConstants& C,
    void* y_device,
    void* uv_device,
    int   y_pitch_bytes,
    int   uv_pitch_bytes)
{
    // Phase C.2 fast path: run the shared chain, then hand the resident
    // gRgbFloat to the Phase B NV12 converter. One bayer upload per frame,
    // no RGB copy back to host. RgbFloatToNv12FromDevice does the final sync.
    if (!IsCudaAvailable()) return false;
    if (!bayer_host || !y_device || !uv_device || !wb) return false;
    if (C.width <= 0 || C.height <= 0) return false;

    const int W = C.width, H = C.height;
    std::lock_guard<std::mutex> lock(gBufMutex);
    if (!EnsureBuffers(W, H)) return false;
    if (!RunBayerChainLocked(bayer_host, wb, C, W, H)) return false;

    return RgbFloatToNv12FromDevice(
        gRgbFloat, y_device, uv_device, W, H,
        y_pitch_bytes, uv_pitch_bytes, C.yuv_matrix);
}

bool ProcessBayerToP010(
    const uint16_t* bayer_host,
    const float wb[3],
    const BayerPipelineConstants& C,
    void* y_device,
    void* uv_device,
    int   y_pitch_bytes,
    int   uv_pitch_bytes)
{
    // Phase E.1: same as ProcessBayerToNv12 but feeds the 10-bit P010
    // converter for Main10 NVENC output.
    if (!IsCudaAvailable()) return false;
    if (!bayer_host || !y_device || !uv_device || !wb) return false;
    if (C.width <= 0 || C.height <= 0) return false;

    const int W = C.width, H = C.height;
    std::lock_guard<std::mutex> lock(gBufMutex);
    if (!EnsureBuffers(W, H)) return false;
    if (!RunBayerChainLocked(bayer_host, wb, C, W, H)) return false;

    return RgbFloatToP010FromDevice(
        gRgbFloat, y_device, uv_device, W, H,
        y_pitch_bytes, uv_pitch_bytes, C.yuv_matrix);
}

}
}
