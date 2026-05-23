#ifndef CudaHwHandoff_hpp
#define CudaHwHandoff_hpp

// Phase A of the CUDA pipeline.
//
// This header is the C++ entry point for our CUDA code. It's deliberately
// kept tiny so the rest of the codebase doesn't need to know about CUDA
// types — the only thing the C++ side does is call IsCudaAvailable() to
// decide whether to take the GPU path or the CPU path.
//
// As we add Phase B / C / D kernels, this header grows to expose them via
// plain C++ signatures. The CUDA-specific types stay inside the .cu file.

namespace motioncam {
namespace cuda {

// Probes for a usable CUDA-capable device at runtime. Returns true if at
// least one device is visible and we successfully retrieved its properties.
// Cached after the first call; safe to call from anywhere.
//
// Use this at NVENC-codec setup time to decide whether to allocate a CUDA
// hardware-frames context for the encoder. If false, fall back to the
// scalar CPU YUV path.
bool IsCudaAvailable();

// Sanity probe: launches a small no-op kernel and synchronises. Returns
// true if the kernel completed without error. Used by the smoke-test
// script to verify the CUDA toolchain is wired correctly end-to-end.
//
// This is the Phase A milestone — proves the build chain works without
// changing any pixel math. Phase B replaces this with the real
// RGB float -> YUV converter.
bool RunPhaseAProbe();

// ---------- Phase B: RGB float -> NV12 (BT.709 limited, 8-bit) ----------
//
// Wraps the GPU side of "feed NVENC from our linear RGB float buffer":
//   1) async H->D copy of width*height*3 floats into a private device
//      buffer (managed inside the .cu file, lazily resized)
//   2) launches the RGB->NV12 conversion kernel directly into the caller-
//      supplied Y / UV device pointers (these come from the NVENC hwframe)
//   3) blocks on cudaDeviceSynchronize before returning so the encoder
//      sees a fully-written frame.
//
// Output convention:
//   - 8-bit NV12 (Y plane + interleaved UV, 4:2:0 chroma subsampling)
//   - BT.709 RGB -> Y'CbCr, limited range (Y in [16,235], CbCr in [16,240])
//   - Input RGB is clamped to [0,1] inside the kernel
//
// y_device / uv_device are CUDA device pointers (the values stored in
// AVFrame::data[0/1] when the frame's hwframes context is CUDA). The
// pitches are AVFrame::linesize[0/1] in bytes.
//
// Returns false if any CUDA call fails (e.g. OOM, kernel launch error);
// caller is expected to fall back to the CPU YUV path in that case.
bool RgbFloatToNv12(
    const float* rgb_host,
    void* y_device,
    void* uv_device,
    int   width,
    int   height,
    int   y_pitch_bytes,
    int   uv_pitch_bytes);

// Release the lazily-allocated device scratch buffer. Call this once at
// MovEncoder destruction so the GPU memory isn't held for the lifetime
// of the process.
void ReleaseRgbScratch();

// Variant of RgbFloatToNv12 that takes a device-side RGB pointer instead
// of a host pointer. Used by Phase C to chain the output of the bayer
// kernels straight into NVENC without a CPU roundtrip. Pitch must equal
// width*sizeof(float3) (we don't currently use a strided RGB layout).
bool RgbFloatToNv12FromDevice(
    const void* rgb_device,
    void* y_device,
    void* uv_device,
    int   width,
    int   height,
    int   y_pitch_bytes,
    int   uv_pitch_bytes);

// ---------- Phase E.1: RGB float -> P010 (BT.709 limited, 10-bit) -------
//
// 10-bit siblings of the NV12 converters. Output is P010: 16-bit samples
// with the 10-bit value in the high bits (value << 6), Y plane + interleaved
// UV, 4:2:0, BT.709 limited range. y_device / uv_device are the P010 hwframe
// planes (AVFrame data[0/1]); pitches are linesize[0/1] in bytes.
bool RgbFloatToP010(
    const float* rgb_host,
    void* y_device,
    void* uv_device,
    int   width,
    int   height,
    int   y_pitch_bytes,
    int   uv_pitch_bytes);

bool RgbFloatToP010FromDevice(
    const void* rgb_device,
    void* y_device,
    void* uv_device,
    int   width,
    int   height,
    int   y_pitch_bytes,
    int   uv_pitch_bytes);

// ---------- Phase C: bayer -> RGB on GPU --------------------------------
//
// Per-clip constants (built once when the encoder starts, identical across
// frames). The combined matrix M is (BakedTransform * Cam->ACEScg) so a
// single 3x3 mul gets us from cam-RGB straight to the target output space.
// `curve` encodes the optional gamma the BakedTransform applies after the
// matrix (matches BakedTransform.cpp's Curve enum, kept as plain int so we
// don't have to drag a header into CUDA code).
struct BayerPipelineConstants {
    float    cam_to_output[9];
    uint16_t black[4];          // per-CFA-position black level
    float    inv_range[4];      // 1.0 / (whiteLevel - black[i])
    int      cfa_channel[4];    // 0=R, 1=G, 2=B per CFA position (matches CfaPattern)
    int      cfa_to_lsm[4];     // CFA position -> LSM channel index (R, Gr, Gb, B = 0..3)
    int      curve;             // 0=None, 1=Gamma22, 2=Gamma24, 3=SRGB
    int      width;
    int      height;

    // Phase D: when non-zero, run the uploaded ACEScg->target 3D LUT after
    // the cam_to_output matrix (which must then be cam->ACEScg, curve None).
    // Requires a prior successful SetupLut3D(). Ignored if no LUT is loaded.
    int      use_lut3d = 0;

    // Optional lens-shading map. If lsm_w > 0 && lsm_h > 0 && lsm_host is
    // non-null, the kernel multiplies each bayer pixel by the bilinearly
    // sampled gain from lsm_host[cfa_to_lsm[idx]] grid. lsm_host points to
    // a channel-first buffer of size 4 * lsm_w * lsm_h floats (R, Gr, Gb, B).
    int          lsm_w;
    int          lsm_h;
    const float* lsm_host;
};

// One-time setup. Allocates persistent GPU buffers (bayer + intermediate
// RGB) sized for `width * height`. Safe to call repeatedly with the same
// dimensions — only the first call actually allocates.
bool SetupBayerPipeline(int width, int height);

// Frees the bayer / intermediate-RGB device buffers. Call at encoder
// destruction.
void ReleaseBayerPipeline();

// Per-frame call. Uploads `bayer_host`, runs the three kernels, leaves
// the result in the persistent GPU RGB buffer. For now, also copies the
// result to `rgb_host` so the Python binding can verify against CPU
// reference (will become a no-op + hwframe handoff in Phase C.2).
//
// `wb` is per-frame (the asShotNeutral from the MotionCam metadata).
//
// Returns false on any CUDA error.
bool ProcessBayerToRgb(
    const uint16_t* bayer_host,
    const float wb[3],
    const BayerPipelineConstants& consts,
    float* rgb_host_out);

// Phase C.2 entry point. Same bayer pipeline as ProcessBayerToRgb but
// instead of copying the result to host, hands the GPU RGB straight to
// the Phase B RGB->NV12 kernel and writes the result into the caller-
// supplied NVENC hwframe Y/UV device pointers. No host roundtrip.
//
// When a 3D LUT has been uploaded via SetupLut3D() AND consts.use_lut3d is
// non-zero, an extra ACEScg->target LUT step runs after the matrix
// (consts.cam_to_output should then be cam->ACEScg with curve == None).
bool ProcessBayerToNv12(
    const uint16_t* bayer_host,
    const float wb[3],
    const BayerPipelineConstants& consts,
    void* y_device,
    void* uv_device,
    int   y_pitch_bytes,
    int   uv_pitch_bytes);

// Phase E.1: 10-bit sibling of ProcessBayerToNv12. Runs the same bayer
// chain (incl. optional 3D LUT) then writes P010 instead of NV12, for
// Main10 NVENC output. y_device / uv_device are the P010 hwframe planes.
bool ProcessBayerToP010(
    const uint16_t* bayer_host,
    const float wb[3],
    const BayerPipelineConstants& consts,
    void* y_device,
    void* uv_device,
    int   y_pitch_bytes,
    int   uv_pitch_bytes);

// ---------- Phase D: OCIO on GPU via 3D LUT -----------------------------
//
// OCIO-only output targets (ACEScct, S-Log3, DaVinci Intermediate, Rec.2020
// PQ/HLG, ...) have no baked matrix+curve. Instead we bake the OCIO
// ACEScg->target transform into a kLutSize^3 float RGB cube on the host
// (see color::BakeAcesCgToTargetLut3D) and sample it on the GPU.
//
// The cube is addressed through an asinh "shaper" that maps ACEScg
// scene-linear (unbounded, can go slightly negative) into the LUT's [0,1]
// domain: linear near black so sub-black survives, log-like in the
// highlights so HDR compresses. These constants are shared by the host bake
// and the device sampler so the two stay in lockstep.
constexpr int   kLutSize     = 65;
constexpr float kLutShaperK  = 0.01f;   // toe width (linear region ~ |L| < K)
constexpr float kLutShaperLo = -0.35f;  // ACEScg value mapped to t = 0
constexpr float kLutShaperHi = 64.0f;   // ACEScg value mapped to t = 1

// Upload an n^3 float LUT. `lut_rgba_host` is n*n*n*4 floats, RGBA, with the
// B axis fastest-varying: index = ((iR*n + iG)*n + iB)*4 (matches
// color::BakeAcesCgToTargetLut3D). The alpha channel is ignored. Allocates a
// cudaArray + texture object with hardware trilinear filtering and replaces
// any previously uploaded LUT. Returns false on any CUDA error.
bool SetupLut3D(const float* lut_rgba_host, int n);

// True once a LUT has been successfully uploaded.
bool HasLut3D();

// Free the LUT cudaArray + texture object. Call at encoder destruction.
void ReleaseLut3D();

}
}

#endif
