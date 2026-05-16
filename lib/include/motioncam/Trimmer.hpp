#ifndef Trimmer_hpp
#define Trimmer_hpp

#include <cstdint>
#include <functional>
#include <string>

namespace motioncam {

// Trim an MCRAW file: write a new MCRAW containing frames [startFrame, endFrame)
// from `inputPath` to `outputPath`. The output is a fully-valid MCRAW that any
// MotionCam-aware tool (including this library) can decode.
//
// Implementation does bit-perfect byte copies of the compressed bayer + frame
// metadata — no decode/re-encode of the raw — so the output is identical to the
// source for the trimmed range. Audio chunks whose timestamp falls in or near
// the trimmed range are also copied. Original frame timestamps are preserved.
//
// `progress` (optional) is called with (current, total) frame counts as the
// trim runs. `cancel` (optional) is called once per frame; if it returns true,
// the trim aborts cleanly (output file is left in whatever partial state it
// reached — caller is responsible for deleting it if desired). Throws
// IOException on read/write errors.
void TrimMcraw(
    const std::string& inputPath,
    const std::string& outputPath,
    int startFrame,
    int endFrame,
    std::function<void(int current, int total)> progress = {},
    std::function<bool()> cancel = {});

}  // namespace motioncam

#endif
