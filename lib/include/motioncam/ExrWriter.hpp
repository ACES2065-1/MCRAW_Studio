#ifndef ExrWriter_hpp
#define ExrWriter_hpp

#include <cstdint>
#include <string>

namespace motioncam {
namespace color {

// EXR compression methods. None / RLE / ZIPS / ZIP / PIZ are lossless; the rest
// are lossy in the half-float domain (visually high quality, much smaller).
enum class ExrCompression {
    None,    // uncompressed, fastest write, biggest files
    RLE,     // lossless, ~30% smaller — great for synthetic / mostly-flat images
    ZIPS,    // lossless, per-line zip — slightly larger than ZIP, friendlier to parallel readers
    ZIP,     // lossless, 16-line-block zip — classic default
    PIZ,     // lossless, wavelet — usually smallest lossless for noisy / camera footage
    PXR24,   // lossy 24-bit (rounds half precision) — small, still very high quality
    B44,     // lossy, fixed ~2.28x over ZIP, fast
    B44A,    // B44 with run-length on flat areas
    DWAA,    // lossy DCT (ILM/DWA) — small, very high quality
    DWAB,    // DWA with bigger blocks — even smaller, marginal quality drop
};

// Parse a CLI/JSON name (case-insensitive) into the enum. Returns false if unknown.
// Accepted: none, rle, zips, zip, piz, pxr24, b44, b44a, dwaa, dwab.
bool ParseExrCompression(const std::string& s, ExrCompression& out);
const char* ExrCompressionName(ExrCompression c);

// chromaticities[8] = { rX, rY, gX, gY, bX, bY, wX, wY } in CIE xy.
// Pass nullptr to omit the chromaticities attribute.
void WriteExrHalfRgb(
    const std::string& path,
    const float* rgbInterleaved,
    uint32_t width,
    uint32_t height,
    const std::string& ocioColorSpace,
    const float* chromaticities,
    ExrCompression compression = ExrCompression::ZIP);

}
}

#endif
