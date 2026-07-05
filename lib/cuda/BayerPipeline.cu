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

// Phase H: CUDA-GL interop for the zero-readback preview. Windows-only
// (this project ships a Windows GUI); other platforms get stub fallbacks.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace motioncam {
namespace cuda {

// ============================================================================
// Persistent device buffers — Phase G: one full set per *context*.
//
// Two contexts exist:
//   gRenderCtx  — the batch NVENC render path (ProcessBayerToRgb / Nv12 /
//                 P010). Runs on the legacy default stream, exactly as the
//                 pre-Phase-G code did, so the NV12/P010 converters and
//                 their cudaDeviceSynchronize keep their ordering.
//   gPreviewCtx — the real-time player (ProcessBayerToRgb8). Has its OWN
//                 mutex, buffer set, pinned host staging, and a
//                 non-blocking CUDA stream, so a 4K render and a proxy-res
//                 preview no longer serialize on one lock, thrash each
//                 other's buffer allocations, or stall each other with
//                 device-wide syncs.
// ============================================================================

namespace {

struct PipelineCtx {
    uint16_t* bayerU16   = nullptr;   // width*height  u16
    float*    bayerFloat = nullptr;   // width*height  float
    float*    rgbFloat   = nullptr;   // width*height*3 float
    uint8_t*  rgb8       = nullptr;   // width*height*3 u8 (preview readback)
    size_t    rgb8Bytes  = 0;
    int       bufW       = 0;
    int       bufH       = 0;
    // LSM scratch: re-allocated when grid dimensions change. Tiny (a few KB
    // for typical 17x13 or 65x49 grids x 4 channels) so we just realloc.
    float*    lsmDevice  = nullptr;
    int       lsmW       = 0;
    int       lsmH       = 0;
    // Phase F denoise scratch (render ctx only in practice).
    float*    dnY        = nullptr;
    float*    dnCb       = nullptr;
    float*    dnCr       = nullptr;
    float*    dnTmp      = nullptr;
    float*    dnWeights  = nullptr;   // up to 65 Gaussian taps
    size_t    dnN        = 0;
    // Pinned host staging (preview ctx only): upload bounce + readback
    // target. Pinned pages DMA at full PCIe rate and make the async
    // copies on `stream` genuinely asynchronous.
    uint16_t* pinnedBayer      = nullptr;
    size_t    pinnedBayerBytes = 0;
    uint8_t*  pinnedRgb8       = nullptr;
    size_t    pinnedRgb8Bytes  = 0;
    bool      usePinned        = false;
    // Scope histogram accumulator (kScopeSlots uint32, preview ctx only).
    uint32_t* scopeDev         = nullptr;
    // Phase H display buffer: RGBA8 frame that stays on the device for the
    // GL blit. Guarded by mtx like everything else; rgbaValid flips true
    // once a frame has been published.
    uint8_t*  rgbaDev          = nullptr;
    size_t    rgbaBytes        = 0;
    int       rgbaW            = 0;
    int       rgbaH            = 0;
    bool      rgbaValid        = false;
    // 0 = legacy default stream (render). The preview ctx lazily creates a
    // cudaStreamNonBlocking stream so it never implicitly syncs with the
    // render's default-stream work.
    cudaStream_t stream        = 0;
    bool         streamCreated = false;
    std::mutex   mtx;
};

PipelineCtx gRenderCtx;
PipelineCtx gPreviewCtx;

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

constexpr int kDnMaxRadius = 32;    // matches Denoise.cpp's cap

void ReleaseDenoiseLocked(PipelineCtx& ctx) {
    if (ctx.dnY)       { cudaFree(ctx.dnY);       ctx.dnY = nullptr; }
    if (ctx.dnCb)      { cudaFree(ctx.dnCb);      ctx.dnCb = nullptr; }
    if (ctx.dnCr)      { cudaFree(ctx.dnCr);      ctx.dnCr = nullptr; }
    if (ctx.dnTmp)     { cudaFree(ctx.dnTmp);     ctx.dnTmp = nullptr; }
    if (ctx.dnWeights) { cudaFree(ctx.dnWeights); ctx.dnWeights = nullptr; }
    ctx.dnN = 0;
}

bool EnsureDenoiseBuffers(PipelineCtx& ctx, size_t n) {
    if (n == ctx.dnN && ctx.dnY && ctx.dnCb && ctx.dnCr && ctx.dnTmp &&
        ctx.dnWeights) return true;
    ReleaseDenoiseLocked(ctx);
    const size_t bytes = n * sizeof(float);
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&ctx.dnY),   bytes);
    cudaError_t e2 = cudaMalloc(reinterpret_cast<void**>(&ctx.dnCb),  bytes);
    cudaError_t e3 = cudaMalloc(reinterpret_cast<void**>(&ctx.dnCr),  bytes);
    cudaError_t e4 = cudaMalloc(reinterpret_cast<void**>(&ctx.dnTmp), bytes);
    cudaError_t e5 = cudaMalloc(reinterpret_cast<void**>(&ctx.dnWeights),
                                size_t(kDnMaxRadius * 2 + 1) * sizeof(float));
    if (e1 || e2 || e3 || e4 || e5) { ReleaseDenoiseLocked(ctx); return false; }
    ctx.dnN = n;
    return true;
}

bool EnsureBuffers(PipelineCtx& ctx, int width, int height) {
    if (width == ctx.bufW && height == ctx.bufH &&
        ctx.bayerU16 && ctx.bayerFloat && ctx.rgbFloat) {
        return true;
    }

    // Resize or first-allocate. Free old buffers if dimensions changed.
    if (ctx.bayerU16)   { cudaFree(ctx.bayerU16);   ctx.bayerU16   = nullptr; }
    if (ctx.bayerFloat) { cudaFree(ctx.bayerFloat); ctx.bayerFloat = nullptr; }
    if (ctx.rgbFloat)   { cudaFree(ctx.rgbFloat);   ctx.rgbFloat   = nullptr; }
    ctx.bufW = 0;
    ctx.bufH = 0;

    const size_t n = size_t(width) * size_t(height);
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&ctx.bayerU16),   n * sizeof(uint16_t));
    cudaError_t e2 = cudaMalloc(reinterpret_cast<void**>(&ctx.bayerFloat), n * sizeof(float));
    cudaError_t e3 = cudaMalloc(reinterpret_cast<void**>(&ctx.rgbFloat),   n * 3 * sizeof(float));
    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
        if (ctx.bayerU16)   { cudaFree(ctx.bayerU16);   ctx.bayerU16   = nullptr; }
        if (ctx.bayerFloat) { cudaFree(ctx.bayerFloat); ctx.bayerFloat = nullptr; }
        if (ctx.rgbFloat)   { cudaFree(ctx.rgbFloat);   ctx.rgbFloat   = nullptr; }
        return false;
    }
    ctx.bufW = width;
    ctx.bufH = height;
    return true;
}

// Preview ctx: lazily create the non-blocking stream and size the pinned
// host staging buffers. Caller holds ctx.mtx.
bool EnsurePreviewTransport(PipelineCtx& ctx, size_t bayerBytes, size_t rgb8Bytes) {
    if (!ctx.streamCreated) {
        if (cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking)
                != cudaSuccess) {
            ctx.stream = 0;   // degrade to the default stream, still correct
        }
        ctx.streamCreated = true;
    }
    if (ctx.pinnedBayerBytes != bayerBytes || !ctx.pinnedBayer) {
        if (ctx.pinnedBayer) { cudaFreeHost(ctx.pinnedBayer); ctx.pinnedBayer = nullptr; }
        ctx.pinnedBayerBytes = 0;
        if (cudaMallocHost(reinterpret_cast<void**>(&ctx.pinnedBayer),
                           bayerBytes) != cudaSuccess) {
            ctx.pinnedBayer = nullptr;
        } else {
            ctx.pinnedBayerBytes = bayerBytes;
        }
    }
    if (ctx.pinnedRgb8Bytes != rgb8Bytes || !ctx.pinnedRgb8) {
        if (ctx.pinnedRgb8) { cudaFreeHost(ctx.pinnedRgb8); ctx.pinnedRgb8 = nullptr; }
        ctx.pinnedRgb8Bytes = 0;
        if (cudaMallocHost(reinterpret_cast<void**>(&ctx.pinnedRgb8),
                           rgb8Bytes) != cudaSuccess) {
            ctx.pinnedRgb8 = nullptr;
        } else {
            ctx.pinnedRgb8Bytes = rgb8Bytes;
        }
    }
    // Pinned staging is an optimisation, not a correctness requirement —
    // if either alloc failed we fall back to pageable copies.
    ctx.usePinned = (ctx.pinnedBayer != nullptr);
    return true;
}

void ReleaseCtxLocked(PipelineCtx& ctx) {
    if (ctx.bayerU16)    { cudaFree(ctx.bayerU16);        ctx.bayerU16    = nullptr; }
    if (ctx.bayerFloat)  { cudaFree(ctx.bayerFloat);      ctx.bayerFloat  = nullptr; }
    if (ctx.rgbFloat)    { cudaFree(ctx.rgbFloat);        ctx.rgbFloat    = nullptr; }
    if (ctx.rgb8)        { cudaFree(ctx.rgb8);            ctx.rgb8        = nullptr; ctx.rgb8Bytes = 0; }
    if (ctx.lsmDevice)   { cudaFree(ctx.lsmDevice);       ctx.lsmDevice   = nullptr; }
    if (ctx.pinnedBayer) { cudaFreeHost(ctx.pinnedBayer); ctx.pinnedBayer = nullptr; ctx.pinnedBayerBytes = 0; }
    if (ctx.pinnedRgb8)  { cudaFreeHost(ctx.pinnedRgb8);  ctx.pinnedRgb8  = nullptr; ctx.pinnedRgb8Bytes = 0; }
    if (ctx.scopeDev)    { cudaFree(ctx.scopeDev);        ctx.scopeDev    = nullptr; }
    if (ctx.rgbaDev)     { cudaFree(ctx.rgbaDev);         ctx.rgbaDev     = nullptr;
                           ctx.rgbaBytes = 0; ctx.rgbaW = 0; ctx.rgbaH = 0;
                           ctx.rgbaValid = false; }
    ReleaseDenoiseLocked(ctx);
    ctx.bufW = 0;
    ctx.bufH = 0;
    ctx.lsmW = 0;
    ctx.lsmH = 0;
}

}  // namespace

bool SetupBayerPipeline(int width, int height) {
    if (!IsCudaAvailable() || width <= 0 || height <= 0) return false;
    std::lock_guard<std::mutex> lock(gRenderCtx.mtx);
    return EnsureBuffers(gRenderCtx, width, height);
}

void ReleaseBayerPipeline() {
    // Encoder teardown releases only the RENDER context. The preview
    // context belongs to the GUI player and survives renders finishing —
    // pre-Phase-G this call would yank the shared buffers out from under
    // a live preview and force a realloc on its next frame.
    std::lock_guard<std::mutex> lock(gRenderCtx.mtx);
    ReleaseCtxLocked(gRenderCtx);
}

// ============================================================================
// Phase D: 3D LUT upload / teardown
// ============================================================================

bool SetupLut3D(const float* lut_rgba_host, int n) {
    // The 3D LUT is a render-path feature (OCIO targets); the preview only
    // runs baked targets, so the LUT globals live under the render lock.
    if (!IsCudaAvailable() || !lut_rgba_host || n < 2) return false;
    std::lock_guard<std::mutex> lock(gRenderCtx.mtx);

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
    std::lock_guard<std::mutex> lock(gRenderCtx.mtx);
    return gLutValid;
}

void ReleaseLut3D() {
    std::lock_guard<std::mutex> lock(gRenderCtx.mtx);
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
// Kernel 2b (Tier 3a): DebayerMalvar  -  Malvar-He-Cutler 5x5 demosaic
// (tap-for-tap copy of Debayer.cpp::DebayerMalvar — keep in lockstep).
// The pipeline default since v0.7; DebayerBilinearKernel stays for reference.
// ============================================================================

__global__ void DebayerMalvarKernel(
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

    const float C = bayer[size_t(y) * size_t(width) + size_t(x)];

    // Shared cross taps (clamped like the CPU version).
    const float axmm = SampleClamped(bayer, x-2, y, width, height);
    const float axpp = SampleClamped(bayer, x+2, y, width, height);
    const float aymm = SampleClamped(bayer, x, y-2, width, height);
    const float aypp = SampleClamped(bayer, x, y+2, width, height);
    float r, g, b;

    if (c == 0 || c == 2) {
        const float cross = SampleClamped(bayer, x-1, y,   width, height)
                          + SampleClamped(bayer, x+1, y,   width, height)
                          + SampleClamped(bayer, x,   y-1, width, height)
                          + SampleClamped(bayer, x,   y+1, width, height);
        const float diag  = SampleClamped(bayer, x-1, y-1, width, height)
                          + SampleClamped(bayer, x+1, y-1, width, height)
                          + SampleClamped(bayer, x-1, y+1, width, height)
                          + SampleClamped(bayer, x+1, y+1, width, height);
        const float axial = axmm + axpp + aymm + aypp;
        const float gv = (4.0f * C + 2.0f * cross - axial) * 0.125f;
        const float ov = (6.0f * C + 2.0f * diag - 1.5f * axial) * 0.125f;
        g = gv;
        if (c == 0) { r = C;  b = ov; }
        else        { b = C;  r = ov; }
    }
    else {
        g = C;
        const float corners = SampleClamped(bayer, x-1, y-1, width, height)
                            + SampleClamped(bayer, x+1, y-1, width, height)
                            + SampleClamped(bayer, x-1, y+1, width, height)
                            + SampleClamped(bayer, x+1, y+1, width, height);
        const float hv = (5.0f * C
                          + 4.0f * (SampleClamped(bayer, x-1, y, width, height)
                                    + SampleClamped(bayer, x+1, y, width, height))
                          - (corners + axmm + axpp)
                          + 0.5f * (aymm + aypp)) * 0.125f;
        const float vv = (5.0f * C
                          + 4.0f * (SampleClamped(bayer, x, y-1, width, height)
                                    + SampleClamped(bayer, x, y+1, width, height))
                          - (corners + aymm + aypp)
                          + 0.5f * (axmm + axpp)) * 0.125f;
        const int h_idx = ((y & 1) << 1) | ((x + 1) & 1);
        int h_c;
        switch (h_idx) {
            case 0:  h_c = ch0; break;
            case 1:  h_c = ch1; break;
            case 2:  h_c = ch2; break;
            default: h_c = ch3; break;
        }
        if (h_c == 0) { r = hv; b = vv; }
        else          { b = hv; r = vv; }
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
// ctx.mtx and have just left ACEScg in ctx.rgbFloat. Returns false only on a
// kernel launch error; a no-op (LUT disabled / not loaded) returns true.
// Only the render ctx ever sets use_lut3d, so the LUT globals stay safely
// under the render lock.
namespace {
bool MaybeApplyLut3D(PipelineCtx& ctx, const BayerPipelineConstants& C,
                     int W, int H, dim3 grid, dim3 block) {
    if (!C.use_lut3d || !gLutValid) return true;
    const float K       = kLutShaperK;
    const float sLo     = std::asinh(kLutShaperLo / K);
    const float sHi     = std::asinh(kLutShaperHi / K);
    const float invSpan = 1.0f / (sHi - sLo);
    ApplyLut3DKernel<<<grid, block, 0, ctx.stream>>>(
        ctx.rgbFloat, W, H, gLutTex, gLutN, K, sLo, invSpan);
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

// Phase F: output-space denoise on ctx.rgbFloat. Caller holds ctx.mtx and has
// left output-space RGB in ctx.rgbFloat (after matrix/LUT/rolloff). No-op when
// both strengths are 0. Returns false on any launch/alloc error.
bool RunDenoiseLocked(PipelineCtx& ctx, const BayerPipelineConstants& C,
                      int W, int H, dim3 grid, dim3 block) {
    int chroma = C.denoise_chroma, luma = C.denoise_luma;
    if (chroma <= 0 && luma <= 0) return true;
    if (chroma < 0) chroma = 0; else if (chroma > 100) chroma = 100;
    if (luma   < 0) luma   = 0; else if (luma   > 100) luma   = 100;
    if (W < 3 || H < 3) return true;

    if (!EnsureDenoiseBuffers(ctx, size_t(W) * size_t(H))) return false;
    const cudaStream_t s = ctx.stream;

    SplitYCbCrKernel<<<grid, block, 0, s>>>(ctx.rgbFloat, ctx.dnY, ctx.dnCb,
                                            ctx.dnCr, W, H);
    if (cudaGetLastError() != cudaSuccess) return false;

    if (chroma > 0) {
        const float sigma = float(chroma) * 0.04f;
        int radius = 0;
        std::vector<float> k = BuildGaussianKernel(sigma, radius);
        if (radius > 0) {
            if (cudaMemcpyAsync(ctx.dnWeights, k.data(), k.size() * sizeof(float),
                                cudaMemcpyHostToDevice, s) != cudaSuccess)
                return false;
            // Cb: H -> tmp, V -> Cb.  Cr: H -> tmp, V -> Cr.
            GaussianBlurAxisKernel<<<grid, block, 0, s>>>(ctx.dnCb, ctx.dnTmp, W, H, ctx.dnWeights, radius, 1);
            GaussianBlurAxisKernel<<<grid, block, 0, s>>>(ctx.dnTmp, ctx.dnCb, W, H, ctx.dnWeights, radius, 0);
            GaussianBlurAxisKernel<<<grid, block, 0, s>>>(ctx.dnCr, ctx.dnTmp, W, H, ctx.dnWeights, radius, 1);
            GaussianBlurAxisKernel<<<grid, block, 0, s>>>(ctx.dnTmp, ctx.dnCr, W, H, ctx.dnWeights, radius, 0);
            if (cudaGetLastError() != cudaSuccess) return false;
        }
    }

    const float* Yptr = ctx.dnY;
    if (luma > 0) {
        const float ssig = float(luma) * 0.025f;
        const float rsig = 0.005f + float(luma) * 0.0005f;
        if (ssig >= 0.05f) {  // matches Bilateral5x5's early-out (memcpy)
            const float invSp2 = 1.0f / (2.0f * ssig * ssig);
            const float invRn2 = 1.0f / (2.0f * rsig * rsig);
            Bilateral5x5Kernel<<<grid, block, 0, s>>>(ctx.dnY, ctx.dnTmp, W, H, invSp2, invRn2);
            if (cudaGetLastError() != cudaSuccess) return false;
            Yptr = ctx.dnTmp;  // denoised luma now in tmp
        }
    }

    CombineYCbCrKernel<<<grid, block, 0, s>>>(ctx.rgbFloat, Yptr, ctx.dnCb,
                                              ctx.dnCr, W, H);
    return cudaGetLastError() == cudaSuccess;
}
}  // namespace

// Shared bayer->ACEScg/target chain: upload -> normalise -> (LSM) -> debayer
// -> matrix(+curve) -> (3D LUT). Leaves the result in ctx.rgbFloat. Caller
// must hold ctx.mtx. Does NOT synchronise — the tail step (host copy / NV12 /
// P010) handles that. Returns false on any launch/copy error.
namespace {
bool RunBayerChainLocked(PipelineCtx& ctx, const uint16_t* bayer_host,
                         const float wb[3],
                         const BayerPipelineConstants& C, int W, int H) {
    const size_t n = size_t(W) * size_t(H);
    const cudaStream_t s = ctx.stream;

    // Upload. The preview ctx bounces through pinned host memory so the
    // async copy DMAs at full rate and genuinely overlaps on its stream;
    // the render ctx keeps the original direct (pageable) copy.
    const void* src = bayer_host;
    if (ctx.usePinned && ctx.pinnedBayerBytes >= n * sizeof(uint16_t)) {
        std::memcpy(ctx.pinnedBayer, bayer_host, n * sizeof(uint16_t));
        src = ctx.pinnedBayer;
    }
    cudaError_t err = cudaMemcpyAsync(
        ctx.bayerU16, src, n * sizeof(uint16_t),
        cudaMemcpyHostToDevice, s);
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

    NormalizeBayerKernel<<<grid, block, 0, s>>>(
        ctx.bayerU16, ctx.bayerFloat, W, H,
        blacks[0], blacks[1], blacks[2], blacks[3],
        scales[0], scales[1], scales[2], scales[3]);
    if (cudaGetLastError() != cudaSuccess) return false;

    // Optional lens shading map (uploaded each call - tiny).
    if (C.lsm_w >= 2 && C.lsm_h >= 2 && C.lsm_host) {
        const size_t lsmBytes = size_t(C.lsm_w) * size_t(C.lsm_h) * 4 * sizeof(float);
        if (C.lsm_w != ctx.lsmW || C.lsm_h != ctx.lsmH || ctx.lsmDevice == nullptr) {
            if (ctx.lsmDevice) { cudaFree(ctx.lsmDevice); ctx.lsmDevice = nullptr; }
            if (cudaMalloc(reinterpret_cast<void**>(&ctx.lsmDevice), lsmBytes) != cudaSuccess)
                return false;
            ctx.lsmW = C.lsm_w;
            ctx.lsmH = C.lsm_h;
        }
        if (cudaMemcpyAsync(ctx.lsmDevice, C.lsm_host, lsmBytes,
                            cudaMemcpyHostToDevice, s) != cudaSuccess)
            return false;

        ApplyLensShadingKernel<<<grid, block, 0, s>>>(
            ctx.bayerFloat, W, H,
            ctx.lsmDevice, C.lsm_w, C.lsm_h,
            C.cfa_to_lsm[0], C.cfa_to_lsm[1],
            C.cfa_to_lsm[2], C.cfa_to_lsm[3]);
        if (cudaGetLastError() != cudaSuccess) return false;
    }

    // Tier 3a: Malvar-He-Cutler demosaic (matches the CPU pipeline default).
    DebayerMalvarKernel<<<grid, block, 0, s>>>(
        ctx.bayerFloat, ctx.rgbFloat, W, H,
        C.cfa_channel[0], C.cfa_channel[1],
        C.cfa_channel[2], C.cfa_channel[3]);
    if (cudaGetLastError() != cudaSuccess) return false;

    // Phase E.3: pre-matrix highlight recovery (cam-RGB, WB-applied).
    if (C.highlight_recovery) {
        NeutraliseClippedHighlightsKernel<<<grid, block, 0, s>>>(
            ctx.rgbFloat, W, H, wb[0], wb[1], wb[2]);
        if (cudaGetLastError() != cudaSuccess) return false;
    }

    ApplyMatrixCurveKernel<<<grid, block, 0, s>>>(
        ctx.rgbFloat, W, H,
        C.cam_to_output[0], C.cam_to_output[1], C.cam_to_output[2],
        C.cam_to_output[3], C.cam_to_output[4], C.cam_to_output[5],
        C.cam_to_output[6], C.cam_to_output[7], C.cam_to_output[8],
        C.curve);
    if (cudaGetLastError() != cudaSuccess) return false;

    // Phase D: optional ACEScg->target 3D LUT (OCIO targets).
    if (!MaybeApplyLut3D(ctx, C, W, H, grid, block)) return false;

    // Phase E.3: post-transform highlight rolloff (display-encoded targets).
    if (C.highlight_recovery && C.highlight_rolloff) {
        const float kneeStart = 1.0f, kneeEnd = 1.4f;
        HighlightRolloffKernel<<<grid, block, 0, s>>>(
            ctx.rgbFloat, W, H, kneeStart, kneeEnd, 1.0f / (kneeEnd - kneeStart));
        if (cudaGetLastError() != cudaSuccess) return false;
    }

    // Phase F: output-space denoise (chroma Gaussian + luma bilateral).
    return RunDenoiseLocked(ctx, C, W, H, grid, block);
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

    std::lock_guard<std::mutex> lock(gRenderCtx.mtx);
    if (!EnsureBuffers(gRenderCtx, W, H)) return false;
    if (!RunBayerChainLocked(gRenderCtx, bayer_host, wb, C, W, H)) return false;

    // Copy the result to host for the Python correctness bindings.
    if (cudaMemcpy(rgb_host_out, gRenderCtx.rgbFloat, n * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    return cudaDeviceSynchronize() == cudaSuccess;
}

// Scopes: RGB histogram + clip counters over the pre-quantize float output.
// Layout matches kScopeSlots: 3x256 bins (R,G,B) then [768]=any-channel<=0
// pixel count, [769]=any-channel>=1 pixel count. Shared-memory privatised
// per block, merged with global atomics. Bin formula matches the u8 clamp
// (round(clamp(v,0,1)*255)) so the histogram is exactly the distribution of
// the displayed image, while the clip counters see the un-clamped floats.
__global__ void ScopeHistogramKernel(const float* __restrict__ rgb,
                                     int pixels,
                                     uint32_t* __restrict__ out) {
    __shared__ uint32_t sh[770];
    for (int i = threadIdx.x; i < 770; i += blockDim.x) sh[i] = 0;
    __syncthreads();

    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pixels; i += stride) {
        bool lo = false, hi = false;
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            const float v = rgb[size_t(i) * 3 + c];
            if (v <= 0.0f) lo = true;
            if (v >= 1.0f) hi = true;
            float t = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            const int b = int(t * 255.0f + 0.5f);
            atomicAdd(&sh[c * 256 + b], 1u);
        }
        if (lo) atomicAdd(&sh[768], 1u);
        if (hi) atomicAdd(&sh[769], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 770; i += blockDim.x)
        if (sh[i]) atomicAdd(&out[i], sh[i]);
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
    uint8_t* rgb8_host_out,
    uint32_t* scope_out)
{
    // Preview fast path (Phase G): full bayer chain on the PREVIEW context —
    // its own buffers, its own non-blocking stream, pinned staging both ways —
    // then clamp to 8-bit RGB888 and read back (≈ width*height*3 bytes, far
    // less than the float buffer). Synchronises only its own stream, so a
    // concurrent NVENC render on the default stream is never stalled.
    if (!IsCudaAvailable()) return false;
    if (!bayer_host || !rgb8_host_out || !wb) return false;
    if (C.width <= 0 || C.height <= 0) return false;

    const int W = C.width, H = C.height;
    const size_t n = size_t(W) * size_t(H);
    const size_t bytes = n * 3;

    PipelineCtx& ctx = gPreviewCtx;
    std::lock_guard<std::mutex> lock(ctx.mtx);
    if (!EnsureBuffers(ctx, W, H)) return false;
    EnsurePreviewTransport(ctx, n * sizeof(uint16_t), bytes);
    if (ctx.rgb8Bytes != bytes || !ctx.rgb8) {
        if (ctx.rgb8) { cudaFree(ctx.rgb8); ctx.rgb8 = nullptr; }
        if (cudaMalloc(reinterpret_cast<void**>(&ctx.rgb8), bytes) != cudaSuccess) {
            ctx.rgb8 = nullptr; ctx.rgb8Bytes = 0; return false;
        }
        ctx.rgb8Bytes = bytes;
    }
    if (!RunBayerChainLocked(ctx, bayer_host, wb, C, W, H)) return false;

    const int total = int(bytes);
    const int block = 256;
    const int grid = (total + block - 1) / block;
    RgbFloatToU8Kernel<<<grid, block, 0, ctx.stream>>>(ctx.rgbFloat, ctx.rgb8, total);
    if (cudaGetLastError() != cudaSuccess) return false;

    // Optional scopes: histogram the float output on-device (tiny readback).
    if (scope_out) {
        const size_t scopeBytes = size_t(kScopeSlots) * sizeof(uint32_t);
        if (!ctx.scopeDev &&
            cudaMalloc(reinterpret_cast<void**>(&ctx.scopeDev), scopeBytes)
                != cudaSuccess) {
            ctx.scopeDev = nullptr;
            return false;
        }
        if (cudaMemsetAsync(ctx.scopeDev, 0, scopeBytes, ctx.stream) != cudaSuccess)
            return false;
        const int hgrid = std::min(256, (int(n) + block - 1) / block);
        ScopeHistogramKernel<<<hgrid, block, 0, ctx.stream>>>(
            ctx.rgbFloat, int(n), ctx.scopeDev);
        if (cudaGetLastError() != cudaSuccess) return false;
    }

    uint8_t* dst = (ctx.pinnedRgb8 && ctx.pinnedRgb8Bytes >= bytes)
                       ? ctx.pinnedRgb8 : rgb8_host_out;
    if (cudaMemcpyAsync(dst, ctx.rgb8, bytes, cudaMemcpyDeviceToHost,
                        ctx.stream) != cudaSuccess)
        return false;
    if (cudaStreamSynchronize(ctx.stream) != cudaSuccess) return false;
    if (dst != rgb8_host_out)
        std::memcpy(rgb8_host_out, dst, bytes);
    if (scope_out) {
        // ~3 KB, stream already idle — a plain sync copy is fine here.
        if (cudaMemcpy(scope_out, ctx.scopeDev,
                       size_t(kScopeSlots) * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
    }
    return true;
}

// Phase H: clamp float RGB -> RGBA8 (alpha 255), pixel-indexed. Same
// quantise as RgbFloatToU8Kernel so the GL path matches the readback path.
__global__ void RgbFloatToRgba8Kernel(const float* __restrict__ rgb,
                                      uchar4* __restrict__ out, int pixels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels) return;
    float r = rgb[size_t(i) * 3 + 0];
    float g = rgb[size_t(i) * 3 + 1];
    float b = rgb[size_t(i) * 3 + 2];
    r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
    g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
    b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);
    out[i] = make_uchar4(uint8_t(r * 255.0f + 0.5f),
                         uint8_t(g * 255.0f + 0.5f),
                         uint8_t(b * 255.0f + 0.5f), 255u);
}

bool ProcessBayerToRgbaDevice(
    const uint16_t* bayer_host,
    const float wb[3],
    const BayerPipelineConstants& C,
    int* out_w,
    int* out_h,
    uint32_t* scope_out)
{
    if (!IsCudaAvailable()) return false;
    if (!bayer_host || !wb || !out_w || !out_h) return false;
    if (C.width <= 0 || C.height <= 0) return false;

    const int W = C.width, H = C.height;
    const size_t n = size_t(W) * size_t(H);
    const size_t rgbaBytes = n * 4;

    PipelineCtx& ctx = gPreviewCtx;
    std::lock_guard<std::mutex> lock(ctx.mtx);
    if (!EnsureBuffers(ctx, W, H)) return false;
    EnsurePreviewTransport(ctx, n * sizeof(uint16_t), n * 3);
    if (ctx.rgbaBytes != rgbaBytes || !ctx.rgbaDev) {
        ctx.rgbaValid = false;
        if (ctx.rgbaDev) { cudaFree(ctx.rgbaDev); ctx.rgbaDev = nullptr; }
        if (cudaMalloc(reinterpret_cast<void**>(&ctx.rgbaDev), rgbaBytes)
                != cudaSuccess) {
            ctx.rgbaDev = nullptr; ctx.rgbaBytes = 0; return false;
        }
        ctx.rgbaBytes = rgbaBytes;
    }
    if (!RunBayerChainLocked(ctx, bayer_host, wb, C, W, H)) return false;

    const int block = 256;
    const int grid = (int(n) + block - 1) / block;
    RgbFloatToRgba8Kernel<<<grid, block, 0, ctx.stream>>>(
        ctx.rgbFloat, reinterpret_cast<uchar4*>(ctx.rgbaDev), int(n));
    if (cudaGetLastError() != cudaSuccess) return false;

    if (scope_out) {
        const size_t scopeBytes = size_t(kScopeSlots) * sizeof(uint32_t);
        if (!ctx.scopeDev &&
            cudaMalloc(reinterpret_cast<void**>(&ctx.scopeDev), scopeBytes)
                != cudaSuccess) {
            ctx.scopeDev = nullptr;
            return false;
        }
        if (cudaMemsetAsync(ctx.scopeDev, 0, scopeBytes, ctx.stream) != cudaSuccess)
            return false;
        const int hgrid = std::min(256, (int(n) + block - 1) / block);
        ScopeHistogramKernel<<<hgrid, block, 0, ctx.stream>>>(
            ctx.rgbFloat, int(n), ctx.scopeDev);
        if (cudaGetLastError() != cudaSuccess) return false;
    }

    if (cudaStreamSynchronize(ctx.stream) != cudaSuccess) return false;
    if (scope_out &&
        cudaMemcpy(scope_out, ctx.scopeDev,
                   size_t(kScopeSlots) * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;

    ctx.rgbaW = W;
    ctx.rgbaH = H;
    ctx.rgbaValid = true;
    *out_w = W;
    *out_h = H;
    return true;
}

#ifdef _WIN32
namespace {
// GL texture id -> cached CUDA registration. Guarded by gPreviewCtx.mtx
// (blit / unregister / release all take it).
std::unordered_map<unsigned int, cudaGraphicsResource_t> gGlTexRes;
}

bool BlitPreviewRgbaToGLTexture(unsigned int gl_texture, int* out_w, int* out_h) {
    if (!IsCudaAvailable() || !out_w || !out_h) return false;
    PipelineCtx& ctx = gPreviewCtx;
    std::lock_guard<std::mutex> lock(ctx.mtx);
    if (!ctx.rgbaValid || !ctx.rgbaDev) return false;

    cudaGraphicsResource_t res = nullptr;
    auto it = gGlTexRes.find(gl_texture);
    if (it != gGlTexRes.end()) {
        res = it->second;
    } else {
        if (cudaGraphicsGLRegisterImage(
                &res, gl_texture, GL_TEXTURE_2D,
                cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess)
            return false;
        gGlTexRes.emplace(gl_texture, res);
    }

    if (cudaGraphicsMapResources(1, &res, 0) != cudaSuccess) return false;
    cudaArray_t arr = nullptr;
    bool ok = cudaGraphicsSubResourceGetMappedArray(&arr, res, 0, 0) == cudaSuccess;
    if (ok) {
        ok = cudaMemcpy2DToArray(
                 arr, 0, 0, ctx.rgbaDev,
                 size_t(ctx.rgbaW) * 4, size_t(ctx.rgbaW) * 4,
                 size_t(ctx.rgbaH), cudaMemcpyDeviceToDevice) == cudaSuccess;
    }
    cudaGraphicsUnmapResources(1, &res, 0);
    if (!ok) return false;
    *out_w = ctx.rgbaW;
    *out_h = ctx.rgbaH;
    return true;
}

void UnregisterPreviewGLTexture(unsigned int gl_texture) {
    std::lock_guard<std::mutex> lock(gPreviewCtx.mtx);
    auto it = gGlTexRes.find(gl_texture);
    if (it != gGlTexRes.end()) {
        cudaGraphicsUnregisterResource(it->second);
        gGlTexRes.erase(it);
    }
}

void ReleaseGlInterop() {
    std::lock_guard<std::mutex> lock(gPreviewCtx.mtx);
    for (auto& kv : gGlTexRes) cudaGraphicsUnregisterResource(kv.second);
    gGlTexRes.clear();
    if (gPreviewCtx.rgbaDev) {
        cudaFree(gPreviewCtx.rgbaDev);
        gPreviewCtx.rgbaDev = nullptr;
        gPreviewCtx.rgbaBytes = 0;
        gPreviewCtx.rgbaValid = false;
    }
}
#else
bool BlitPreviewRgbaToGLTexture(unsigned int, int*, int*) { return false; }
void UnregisterPreviewGLTexture(unsigned int) {}
void ReleaseGlInterop() {}
#endif

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
    std::lock_guard<std::mutex> lock(gRenderCtx.mtx);
    if (!EnsureBuffers(gRenderCtx, W, H)) return false;
    if (!RunBayerChainLocked(gRenderCtx, bayer_host, wb, C, W, H)) return false;

    return RgbFloatToNv12FromDevice(
        gRenderCtx.rgbFloat, y_device, uv_device, W, H,
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
    std::lock_guard<std::mutex> lock(gRenderCtx.mtx);
    if (!EnsureBuffers(gRenderCtx, W, H)) return false;
    if (!RunBayerChainLocked(gRenderCtx, bayer_host, wb, C, W, H)) return false;

    return RgbFloatToP010FromDevice(
        gRenderCtx.rgbFloat, y_device, uv_device, W, H,
        y_pitch_bytes, uv_pitch_bytes, C.yuv_matrix);
}

}
}
