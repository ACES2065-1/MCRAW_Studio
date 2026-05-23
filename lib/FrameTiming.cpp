#include <motioncam/FrameTiming.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace motioncam {
namespace video {

namespace {

struct Candidate { int num; int den; };

// Standard capture / delivery rates, integer and NTSC-fractional.
constexpr Candidate kCandidates[] = {
    {24000, 1001}, {24, 1}, {25, 1},
    {30000, 1001}, {30, 1},
    {48, 1}, {50, 1},
    {60000, 1001}, {60, 1},
    {100, 1}, {120, 1},
};

// Snap an fps value to the closest candidate within 1% (relative); else
// round to an integer rate. The 1% window comfortably covers timestamp jitter
// while still distinguishing 30.000 from 29.970 (0.1% apart) by closeness.
FrameRate SnapInternal(double fps) {
    if (fps <= 0.0) return FrameRate{30, 1};
    double bestDiff = 1e30;
    FrameRate best{};
    bool found = false;
    for (const auto& c : kCandidates) {
        const double cf = double(c.num) / double(c.den);
        const double d = std::abs(cf - fps);
        if (d < 0.01 * cf && d < bestDiff) {
            bestDiff = d;
            best = FrameRate{c.num, c.den};
            found = true;
        }
    }
    if (!found) best = FrameRate{int(fps + 0.5), 1};
    return best;
}

}  // namespace

FrameRate SnapFrameRate(double fps) { return SnapInternal(fps); }

FrameRate DetectFrameRate(const std::vector<int64_t>& ts) {
    if (ts.size() < 2) return FrameRate{30, 1};

    std::vector<int64_t> gaps;
    gaps.reserve(ts.size() - 1);
    for (size_t i = 1; i < ts.size(); ++i) {
        const int64_t d = ts[i] - ts[i - 1];
        if (d > 0) gaps.push_back(d);
    }
    if (gaps.empty()) return FrameRate{30, 1};

    std::sort(gaps.begin(), gaps.end());
    const double medianNs = double(gaps[gaps.size() / 2]);
    if (medianNs <= 0.0) return FrameRate{30, 1};
    return SnapInternal(1.0e9 / medianNs);
}

FramePlan BuildFramePlan(const std::vector<int64_t>& ts,
                         int start, int end, double targetFps) {
    FramePlan plan;
    plan.srcRate = DetectFrameRate(ts);

    const int n = int(ts.size());
    start = std::max(0, std::min(start, n));
    end   = std::max(start, std::min(end, n));
    const int count = end - start;

    // Identity mapping (no conversion): 1:1 at the detected source rate.
    if (targetFps <= 0.0 || count <= 0) {
        plan.outRate = plan.srcRate;
        plan.srcIndex.reserve(count > 0 ? count : 0);
        for (int i = start; i < end; ++i) plan.srcIndex.push_back(i);
        return plan;
    }

    plan.outRate = SnapInternal(targetFps);

    // A single frame, or a zero/negative time span: nothing to resample.
    const int64_t t0 = ts[start];
    const int64_t span = ts[end - 1] - t0;
    if (count == 1 || span <= 0) {
        plan.srcIndex.push_back(start);
        plan.outRate = plan.srcRate;
        return plan;
    }

    // Constant-rate output covering the source time span. +1 so the final
    // source moment gets a slot (inclusive endpoint).
    const double spanSec = double(span) / 1.0e9;
    const int outCount = std::max(1, int(std::llround(spanSec * targetFps)) + 1);

    plan.srcIndex.reserve(size_t(outCount));
    std::vector<char> used(size_t(n), 0);
    int cursor = start;       // monotonic — timestamps are sorted
    int lastEmitted = -1;
    for (int j = 0; j < outCount; ++j) {
        const int64_t tjns = t0 + int64_t(std::llround(double(j) / targetFps * 1.0e9));
        while (cursor + 1 < end &&
               std::llabs(ts[cursor + 1] - tjns) <= std::llabs(ts[cursor] - tjns)) {
            ++cursor;
        }
        plan.srcIndex.push_back(cursor);
        if (cursor == lastEmitted) ++plan.duplicated;
        used[size_t(cursor)] = 1;
        lastEmitted = cursor;
    }
    for (int i = start; i < end; ++i) if (!used[size_t(i)]) ++plan.dropped;
    return plan;
}

}  // namespace video
}  // namespace motioncam
