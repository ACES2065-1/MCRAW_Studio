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

}
}

#endif
