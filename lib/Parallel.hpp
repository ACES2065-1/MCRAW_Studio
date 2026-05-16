#ifndef Parallel_hpp
#define Parallel_hpp

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace motioncam {
namespace internal {

// Splits [0, total) into ~hardware_concurrency contiguous chunks and runs
// each on its own std::thread. fn is called as fn(begin, end) per chunk
// with [begin, end) being a half-open sub-range.
//
// Designed for compute-heavy embarrassingly-parallel loops (debayer rows,
// per-pixel matrix multiplies). Spawns threads per call — ~100 µs overhead
// per thread, fine when the work is millions of pixels.
template <typename F>
inline void ParallelForRange(std::size_t total, F&& fn) {
    if (total == 0) return;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t threads = std::min<std::size_t>(hw, total);
    if (threads <= 1) {
        fn(std::size_t(0), total);
        return;
    }
    const std::size_t chunk = (total + threads - 1) / threads;
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
        const std::size_t i0 = t * chunk;
        const std::size_t i1 = std::min(total, i0 + chunk);
        if (i0 >= i1) break;
        workers.emplace_back([i0, i1, &fn]() { fn(i0, i1); });
    }
    for (auto& w : workers) w.join();
}

}
}

#endif
