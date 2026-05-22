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

}
}
