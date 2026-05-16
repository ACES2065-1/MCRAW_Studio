#ifndef Denoise_hpp
#define Denoise_hpp

#include <cstdint>

namespace motioncam {
namespace color {

// Two-axis denoise on float-RGB-interleaved buffer (in/out, in-place).
// Both strengths in [0..100]; 0 = no-op for that axis.
//
//   chromaStrength : Camera Raw "Color" slider equivalent.
//                    Aggressive Gaussian blur on Cb/Cr planes — kills the
//                    magenta/green chroma blobs in shadows and skin without
//                    softening edges or detail.
//   lumaStrength   : Camera Raw "Luminance" slider equivalent.
//                    Edge-preserving bilateral smooth on Y — gentler, can
//                    soften fine detail at high settings.
//
// Operates in BT.709-weighted Y/Cb/Cr; that approximation is fine for noise
// separation across our supported output spaces (rec709, sRGB, ACEScg). The
// denoise is applied AFTER the OCIO transform so it sees the same image the
// encoder will see.
void DenoiseRgb(
    float* rgbInterleaved,
    uint32_t width,
    uint32_t height,
    int chromaStrength,
    int lumaStrength);

}
}

#endif
