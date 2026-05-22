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

}
}

#endif
