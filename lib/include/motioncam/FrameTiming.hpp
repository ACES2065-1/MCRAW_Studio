#ifndef FrameTiming_hpp
#define FrameTiming_hpp

#include <cstdint>
#include <vector>

namespace motioncam {
namespace video {

// Rational frame rate (num/den), e.g. 30/1 or 30000/1001 (29.97).
struct FrameRate {
    int num = 30;
    int den = 1;
    double fps() const { return double(num) / double(den); }
};

// Detect the source frame rate from per-frame capture timestamps (nanoseconds),
// using the MEDIAN inter-frame interval — robust to dropped frames, unlike a
// plain average — then snapping to the nearest standard rate (integer or NTSC
// fraction) when within ~1%. Falls back to round(median fps) otherwise.
// Returns 30/1 for fewer than 2 timestamps.
FrameRate DetectFrameRate(const std::vector<int64_t>& timestampsNs);

// Snap an arbitrary fps value to the nearest standard rate (same table as
// DetectFrameRate); used for a user-chosen conversion target.
FrameRate SnapFrameRate(double fps);

// Source -> output frame mapping for the output frames covering source range
// [start, end). One srcIndex entry per OUTPUT frame.
//
//  - targetFps <= 0  : identity (1:1 at the detected source rate; no resample).
//  - targetFps  > 0  : a constant-rate stream at targetFps, built by nearest-
//                      timestamp resampling — source frames are duplicated or
//                      dropped so the output stays time-aligned with the source
//                      (and therefore the audio). This is the "frame rate
//                      conversion" the GUI exposes.
struct FramePlan {
    std::vector<int> srcIndex;   // size == number of output frames
    FrameRate srcRate;           // detected source rate
    FrameRate outRate;           // == srcRate for identity, else SnapFrameRate(targetFps)
    int duplicated = 0;          // output frames that repeat the previous source frame
    int dropped = 0;             // source frames in [start,end) never emitted
};

FramePlan BuildFramePlan(const std::vector<int64_t>& timestampsNs,
                         int start, int end, double targetFps);

}  // namespace video
}  // namespace motioncam

#endif
