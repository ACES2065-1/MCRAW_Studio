#ifndef ColorPipeline_hpp
#define ColorPipeline_hpp

#include <motioncam/Debayer.hpp>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace motioncam {
namespace color {

enum class OutputColorSpace {
    // Pipeline-native (no OCIO): pipeline produces these directly via hardcoded matrices.
    ACEScg,
    LinearRec709,
    // OCIO-routed: pipeline produces ACEScg, then OcioTransform converts to target.
    ACES2065_1,
    ACEScct,
    SLog3SGamut3Cine,
    SRGB,                  // sRGB curve (piecewise), Rec.709 primaries
    Rec709Gamma22,         // pure 2.2 gamma, Rec.709 primaries
    Rec709Display,         // pure 2.4 gamma, Rec.709 primaries (BT.1886-ish display)
    // Wide-gamut + HDR + grading additions (Phase A.3):
    Rec2020Linear,         // Linear Rec.2020 (BT.2020 primaries, scene-referred)
    Rec2020PQ,             // Rec.2100-PQ Display (BT.2020 + ST2084) — HDR10 delivery
    Rec2020HLG,            // Rec.2100-HLG Display (BT.2020 + ARIB B67) — HDR HLG delivery
    DaVinciWGLinear,       // Linear DaVinci WideGamut — grading working space
    DaVinciIntermediate,   // DaVinci Intermediate (DaVinci WG + DI log)
    ADX10,                 // ADX10 (AP0 + Cineon log, 10-bit) — Academy Density Exchange
};

struct OutputColorSpaceInfo {
    const char* ocioName;
    const char* shortName;
    bool requiresOcio;
    float chromaticities[8];
    // QuickTime 'nclc' atom values, matching FFmpeg's AVCOL_* enums
    // (ITU-T H.273 / ISO 23091-2). 2 = UNSPECIFIED for spaces with no clean
    // QT mapping (ACEScg, S-Log3, ACEScct).
    int qtPrimaries;   // 1 = BT.709, 9 = BT.2020, 12 = SMPTE432 (P3-D65), 2 = unspecified
    int qtTransfer;    // 1 = BT.709 OETF, 8 = linear, 13 = IEC61966-2-1 (sRGB), 2 = unspecified
    int qtMatrix;      // 1 = BT.709, 9 = BT.2020 NCL, 2 = unspecified
};

const OutputColorSpaceInfo& GetInfo(OutputColorSpace cs);
bool ParseOutputColorSpace(const std::string& s, OutputColorSpace& out);

struct FrameParams {
    uint32_t width;
    uint32_t height;
    uint16_t blackPerPosition[4];
    double whiteLevel;
    float asShotNeutral[3];
    float forwardMatrix2[9];
    CfaPattern cfa;
    // Optional lens shading map. Stored channel-first as MotionCam writes it:
    //   index = ch * (lsmWidth * lsmHeight) + y * lsmWidth + x
    // ch order: 0=R, 1=Gr, 2=Gb, 3=B. Total length: 4 * lsmWidth * lsmHeight.
    // Empty when source frame doesn't carry one.
    std::vector<float> lensShadingMap;
    uint32_t lsmWidth = 0;
    uint32_t lsmHeight = 0;
};

FrameParams BuildFrameParams(
    const nlohmann::json& frameMetadata,
    const nlohmann::json& containerMetadata);

void ProcessFrame(
    const uint16_t* rawBayer,
    const FrameParams& params,
    OutputColorSpace output,
    std::vector<float>& outRgbInterleaved,
    // Highlight recovery: when true, the pipeline neutralises pixels that
    // were at sensor saturation (kills the magenta-sun artifact) and applies
    // a soft highlight rolloff for display-encoded outputs. No-op for
    // scene-referred outputs except for the saturation neutralisation.
    bool highlightRecovery = false);

// True if the named output is "display-encoded" (carries a non-linear
// transfer like Rec.709 / sRGB / PQ / HLG). When highlight recovery is
// requested AND the output is display-encoded, the pipeline applies the
// soft rolloff after the OCIO step.
bool IsDisplayEncoded(OutputColorSpace cs);

}
}

#endif
