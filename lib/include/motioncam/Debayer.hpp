#ifndef Debayer_hpp
#define Debayer_hpp

#include <cstdint>
#include <string>

namespace motioncam {
namespace color {

enum class CfaPattern { RGGB, BGGR, GRBG, GBRG };

CfaPattern ParseCfa(const std::string& arrangement);

void NormalizeBayer(
    const uint16_t* raw,
    float* out,
    uint32_t width,
    uint32_t height,
    const uint16_t blackPerPosition[4],
    double whiteLevel,
    const float wbRgb[3],
    CfaPattern pattern);

// Per-pixel highlight neutralisation in CAMERA-RGB space (call after
// DebayerBilinear and before the camera→output matrix). Detects pixels at
// sensor saturation and lifts the dimmer channels up to the brightest one,
// producing neutral camera RGB at the clipped brightness. Matrix and OCIO
// stages then preserve the neutrality, so the magenta-sun artifact renders
// as bright neutral white in any output space.
void NeutraliseClippedHighlights(
    float* rgbInterleaved,
    uint32_t width,
    uint32_t height,
    const float wbRgb[3]);

// Smooth highlight rolloff applied AFTER the camera→output color transform.
// For each pixel, anything above `kneeStart` is gradually pushed toward
// neutral as it brightens, with a hard ceiling at `kneeEnd`. Used for
// display-encoded outputs (Rec.709 / sRGB / gamma) to neutralise residual
// channel imbalances above 1.0; not applied to scene-referred outputs
// (EXR linear, ACEScg) where users expect raw HDR values.
void HighlightRolloff(
    float* rgbInterleaved,
    uint32_t width,
    uint32_t height,
    float kneeStart = 1.00f,
    float kneeEnd   = 1.40f);

void DebayerBilinear(
    const float* bayer,
    float* outRgbInterleaved,
    uint32_t width,
    uint32_t height,
    CfaPattern pattern);

// Apply a lens shading map (per-pixel multipliers) to a normalized float Bayer.
// Layout follows Android CameraMetadata LENS_SHADING_MAP: lsmWidth * lsmHeight
// grid points, each holding 4 floats in [R, Gr, Gb, B] order (regardless of
// CFA pattern). We bilinearly upsample the grid to full resolution, picking
// the right channel per CFA position.
void ApplyLensShading(
    float* bayer,
    uint32_t width,
    uint32_t height,
    const float* lsm,
    uint32_t lsmWidth,
    uint32_t lsmHeight,
    CfaPattern pattern);

}
}

#endif
