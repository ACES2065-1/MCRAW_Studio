#include <motioncam/ColorPipeline.hpp>
#include "Parallel.hpp"

#include <stdexcept>

namespace motioncam {
namespace color {

namespace {

// Bradford chromatic adaptation: XYZ D50 -> XYZ D65.
// DNG's ForwardMatrix is defined as camera -> XYZ D50 (PCS), so we need D50->D65
// before the standard XYZ D65 -> linear Rec.709 matrix.
const float kBradfordD50toD65[9] = {
     0.9555766f, -0.0230393f,  0.0631636f,
    -0.0282895f,  1.0099416f,  0.0210077f,
     0.0122982f, -0.0204830f,  1.3299098f,
};

const float kXyzD65toRec709[9] = {
     3.2404542f, -1.5371385f, -0.4985314f,
    -0.9692660f,  1.8760108f,  0.0415560f,
     0.0556434f, -0.2040259f,  1.0572252f,
};

// Bradford D50 -> ACES "D60" white (CIE x=0.32168 y=0.33767), derived from the
// standard Bradford response matrix and the published ACES whitepoint.
const float kBradfordD50toD60[9] = {
     0.96766f, -0.01686f,  0.04424f,
    -0.02099f,  1.00778f,  0.01477f,
     0.00853f, -0.01415f,  1.22963f,
};

// XYZ (ACES D60) -> linear ACEScg (AP1 primaries, AP1-D60 white).
// Inverse of the standardized AP1 RGB->XYZ matrix.
const float kXyzD60toAP1[9] = {
     1.6410233797f, -0.3248032942f, -0.2364246952f,
    -0.6636628587f,  1.6153315917f,  0.0167563477f,
     0.0117218943f, -0.0082844420f,  0.9883948585f,
};

void Mat3Mul(const float A[9], const float B[9], float out[9]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i*3 + j] = A[i*3+0]*B[0*3+j] + A[i*3+1]*B[1*3+j] + A[i*3+2]*B[2*3+j];
        }
    }
}

void BuildCameraToRec709(const float forwardMatrixD50[9], float out[9]) {
    float tmp[9];
    Mat3Mul(kBradfordD50toD65, forwardMatrixD50, tmp);
    Mat3Mul(kXyzD65toRec709, tmp, out);
}

void BuildCameraToAcescg(const float forwardMatrixD50[9], float out[9]) {
    float tmp[9];
    Mat3Mul(kBradfordD50toD60, forwardMatrixD50, tmp);
    Mat3Mul(kXyzD60toAP1, tmp, out);
}

void ApplyMatrixInPlace(float* rgb, size_t numPixels, const float M[9]) {
    // NOTE: the AoS interleaved-RGB layout means an explicit AVX2 path here
    // pays 9 vpermps + 4 vblendps for the deinterleave + reinterleave per
    // 8-pixel batch, which on Skylake / Zen-class cores costs roughly as
    // much as the 9 FMAs it saves. We benchmarked the explicit SIMD version
    // and it came out flat-to-slightly-worse against /arch:AVX2 scalar code
    // (the auto-vectorizer + FMA contraction does fine here). Keeping the
    // simple scalar loop — when we eventually move to a planar RGB layout
    // through the pipeline, this kernel becomes trivially SIMD-friendly.
    motioncam::internal::ParallelForRange(numPixels, [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i) {
            const float r = rgb[i*3 + 0];
            const float g = rgb[i*3 + 1];
            const float b = rgb[i*3 + 2];
            rgb[i*3 + 0] = M[0]*r + M[1]*g + M[2]*b;
            rgb[i*3 + 1] = M[3]*r + M[4]*g + M[5]*b;
            rgb[i*3 + 2] = M[6]*r + M[7]*g + M[8]*b;
        }
    });
}

}

namespace {

// Permissive number reader. nlohmann::json is strict about integer / unsigned /
// float subtypes — `.get<uint32_t>()` throws type_error 302 if the value is
// stored as signed int or float, even though it's "a number". MotionCam's
// metadata writer varies across app versions on what subtype it uses for
// width / height / black levels, so we go through `double` to coerce uniformly.
// Treat null / missing as the supplied default so optional / sparse fields
// don't blow up the render.
double ReadNumber(const nlohmann::json& j, double fallback = 0.0) {
    if (j.is_null()) return fallback;
    if (j.is_number()) return j.get<double>();
    if (j.is_boolean()) return j.get<bool>() ? 1.0 : 0.0;
    if (j.is_string()) {
        try { return std::stod(j.get<std::string>()); } catch (...) { return fallback; }
    }
    return fallback;
}

double ReadField(const nlohmann::json& obj, const char* key, double fallback = 0.0) {
    auto it = obj.find(key);
    if (it == obj.end()) return fallback;
    return ReadNumber(*it, fallback);
}

template <typename T>
void ReadNumberArray(const nlohmann::json& arr, T* out, int expected, const char* fieldName) {
    if (!arr.is_array()) {
        throw std::runtime_error(std::string(fieldName) + " must be an array");
    }
    if (int(arr.size()) != expected) {
        throw std::runtime_error(std::string(fieldName) + " must have "
                                 + std::to_string(expected) + " entries");
    }
    for (int i = 0; i < expected; ++i) {
        out[i] = static_cast<T>(ReadNumber(arr[i], 0.0));
    }
}

}  // namespace

FrameParams BuildFrameParams(
    const nlohmann::json& frameMeta,
    const nlohmann::json& containerMeta)
{
    FrameParams p{};
    p.width  = static_cast<uint32_t>(ReadField(frameMeta, "width"));
    p.height = static_cast<uint32_t>(ReadField(frameMeta, "height"));
    if (p.width == 0 || p.height == 0) {
        throw std::runtime_error("frame width/height missing or zero in metadata");
    }

    auto blackIt = containerMeta.find("blackLevel");
    if (blackIt == containerMeta.end()) {
        throw std::runtime_error("blackLevel missing in container metadata");
    }
    ReadNumberArray<uint16_t>(*blackIt, p.blackPerPosition, 4, "blackLevel");

    p.whiteLevel = ReadField(containerMeta, "whiteLevel", 0.0);
    if (p.whiteLevel <= 0.0) {
        throw std::runtime_error("whiteLevel missing or non-positive in container metadata");
    }

    auto wbIt = frameMeta.find("asShotNeutral");
    if (wbIt == frameMeta.end()) {
        throw std::runtime_error("asShotNeutral missing in frame metadata");
    }
    ReadNumberArray<float>(*wbIt, p.asShotNeutral, 3, "asShotNeutral");

    auto fmIt = containerMeta.find("forwardMatrix2");
    if (fmIt == containerMeta.end()) {
        throw std::runtime_error("forwardMatrix2 missing in container metadata");
    }
    ReadNumberArray<float>(*fmIt, p.forwardMatrix2, 9, "forwardMatrix2");

    auto cfaIt = containerMeta.find("sensorArrangment");
    if (cfaIt == containerMeta.end() || !cfaIt->is_string()) {
        throw std::runtime_error("sensorArrangment missing in container metadata");
    }
    p.cfa = ParseCfa(cfaIt->get<std::string>());

    // Optional lens shading map. MotionCam writes it as a channel-first JSON
    // array: lensShadingMap = [R_grid, Gr_grid, Gb_grid, B_grid], each a flat
    // list of lsmW*lsmH floats. Concatenate into one contiguous buffer so
    // ApplyLensShading sees [ch][y*lsmW+x] cleanly. Any parse failure is
    // silently swallowed — vignette correction is non-critical.
    if (frameMeta.contains("lensShadingMap") &&
        frameMeta.contains("lensShadingMapWidth") &&
        frameMeta.contains("lensShadingMapHeight"))
    {
        try {
            const uint32_t lw = static_cast<uint32_t>(ReadField(frameMeta, "lensShadingMapWidth"));
            const uint32_t lh = static_cast<uint32_t>(ReadField(frameMeta, "lensShadingMapHeight"));
            const auto& maps = frameMeta["lensShadingMap"];
            if (lw >= 2 && lh >= 2 && maps.is_array() && maps.size() == 4) {
                const size_t perCh = size_t(lw) * size_t(lh);
                std::vector<float> flat(perCh * 4);
                bool ok = true;
                for (int ch = 0; ch < 4 && ok; ++ch) {
                    if (!maps[ch].is_array() || maps[ch].size() != perCh) { ok = false; break; }
                    for (size_t k = 0; k < perCh && ok; ++k) {
                        flat[ch * perCh + k] = static_cast<float>(ReadNumber(maps[ch][k], 1.0));
                    }
                }
                if (ok) {
                    p.lensShadingMap = std::move(flat);
                    p.lsmWidth = lw;
                    p.lsmHeight = lh;
                }
            }
        } catch (...) {
            // Wrong type or shape — leave blank, vignette stays uncorrected.
        }
    }

    return p;
}

void ProcessFrame(
    const uint16_t* rawBayer,
    const FrameParams& params,
    OutputColorSpace output,
    std::vector<float>& outRgb,
    bool highlightRecovery)
{
    const size_t numPixels = size_t(params.width) * size_t(params.height);
    std::vector<float> bayerFloat(numPixels);

    NormalizeBayer(
        rawBayer, bayerFloat.data(),
        params.width, params.height,
        params.blackPerPosition, params.whiteLevel, params.asShotNeutral,
        params.cfa);

    if (!params.lensShadingMap.empty()) {
        ApplyLensShading(
            bayerFloat.data(),
            params.width, params.height,
            params.lensShadingMap.data(),
            params.lsmWidth, params.lsmHeight,
            params.cfa);
    }

    outRgb.resize(numPixels * 3);

    DebayerBilinear(
        bayerFloat.data(), outRgb.data(),
        params.width, params.height,
        params.cfa);

    if (highlightRecovery) {
        // In camera-RGB space, before the cam→output matrix. This keeps the
        // neutralised pixels truly neutral after the matrix transform.
        NeutraliseClippedHighlights(
            outRgb.data(), params.width, params.height,
            params.asShotNeutral);
    }

    float M[9];
    switch (output) {
        case OutputColorSpace::LinearRec709:
            BuildCameraToRec709(params.forwardMatrix2, M);
            break;
        default:
            BuildCameraToAcescg(params.forwardMatrix2, M);
            break;
    }
    ApplyMatrixInPlace(outRgb.data(), numPixels, M);
    // Note: NeutraliseClippedHighlights ran above (camera-RGB space). The
    // matrix preserved neutrality, so saturated pixels are now bright neutral
    // in the output space. Display-encoded callers additionally run
    // HighlightRolloff() after the OCIO/baked output transform to compress
    // the >1.0 neutrals into [0, kneeEnd]. Scene-referred outputs keep the
    // full HDR brightness at the neutralised hue.
}

bool IsDisplayEncoded(OutputColorSpace cs) {
    // True for spaces that carry a non-linear display transfer (gamma /
    // PQ / HLG / sRGB curve). False for scene-referred / linear / log
    // workspaces — those keep their HDR values for downstream grading.
    switch (cs) {
        case OutputColorSpace::SRGB:
        case OutputColorSpace::Rec709Gamma22:
        case OutputColorSpace::Rec709Display:
        case OutputColorSpace::Rec2020PQ:
        case OutputColorSpace::Rec2020HLG:
            return true;
        case OutputColorSpace::ACEScg:
        case OutputColorSpace::LinearRec709:
        case OutputColorSpace::ACES2065_1:
        case OutputColorSpace::ACEScct:
        case OutputColorSpace::SLog3SGamut3Cine:
        case OutputColorSpace::Rec2020Linear:
        case OutputColorSpace::DaVinciWGLinear:
        case OutputColorSpace::DaVinciIntermediate:
        case OutputColorSpace::ADX10:
            return false;
    }
    return false;
}

const OutputColorSpaceInfo& GetInfo(OutputColorSpace cs) {
    static const OutputColorSpaceInfo acescg = {
        "ACEScg", "acescg", false,
        { 0.713f, 0.293f, 0.165f, 0.830f, 0.128f, 0.044f, 0.32168f, 0.33767f },
        2, 2, 2  // unspecified — ACEScg has no QT mapping
    };
    static const OutputColorSpaceInfo rec709 = {
        "Linear Rec.709 (sRGB)", "rec709", false,
        { 0.640f, 0.330f, 0.300f, 0.600f, 0.150f, 0.060f, 0.3127f, 0.3290f },
        1, 8, 1  // BT.709 primaries, linear transfer, BT.709 matrix
    };
    static const OutputColorSpaceInfo aces2065 = {
        "ACES2065-1", "aces2065-1", true,
        { 0.7347f, 0.2653f, 0.0000f, 1.0000f, 0.0001f, -0.0770f, 0.32168f, 0.33767f },
        2, 2, 2
    };
    static const OutputColorSpaceInfo acescct = {
        "ACEScct", "acescct", true,
        { 0.713f, 0.293f, 0.165f, 0.830f, 0.128f, 0.044f, 0.32168f, 0.33767f },
        2, 2, 2
    };
    static const OutputColorSpaceInfo slog3 = {
        "S-Log3 S-Gamut3.Cine", "slog3-sgamut3cine", true,
        { 0.766f, 0.275f, 0.225f, 0.800f, 0.089f, -0.087f, 0.3127f, 0.3290f },
        2, 2, 2  // S-Log3 has no QT nclc mapping; let NLE auto-detect from the codec/metadata path
    };
    static const OutputColorSpaceInfo srgb = {
        "sRGB Encoded Rec.709 (sRGB)", "srgb", true,
        { 0.640f, 0.330f, 0.300f, 0.600f, 0.150f, 0.060f, 0.3127f, 0.3290f },
        1, 13, 1  // BT.709 primaries, IEC61966-2-1 (sRGB) transfer, BT.709 matrix
    };
    static const OutputColorSpaceInfo rec709g22 = {
        "Gamma 2.2 Encoded Rec.709", "rec709-2.2", true,
        { 0.640f, 0.330f, 0.300f, 0.600f, 0.150f, 0.060f, 0.3127f, 0.3290f },
        1, 4, 1  // BT.709 primaries, BT.470M (gamma 2.2) transfer, BT.709 matrix
    };
    static const OutputColorSpaceInfo rec709disp = {
        "Gamma 2.4 Encoded Rec.709", "rec709-display", true,
        { 0.640f, 0.330f, 0.300f, 0.600f, 0.150f, 0.060f, 0.3127f, 0.3290f },
        1, 1, 1  // full BT.709 (gamma 2.4-ish)
    };

    // BT.2020 primaries (HDR / wide gamut)
    static const OutputColorSpaceInfo rec2020lin = {
        "Linear Rec.2020", "rec2020-linear", true,
        { 0.708f, 0.292f, 0.170f, 0.797f, 0.131f, 0.046f, 0.3127f, 0.3290f },
        9, 8, 9   // BT.2020 primaries, LINEAR transfer, BT.2020 NCL matrix
    };
    static const OutputColorSpaceInfo rec2020pq = {
        "Rec.2100-PQ - Display", "rec2020-pq", true,
        { 0.708f, 0.292f, 0.170f, 0.797f, 0.131f, 0.046f, 0.3127f, 0.3290f },
        9, 16, 9  // SMPTE ST2084 transfer (HDR10 delivery)
    };
    static const OutputColorSpaceInfo rec2020hlg = {
        "Rec.2100-HLG - Display", "rec2020-hlg", true,
        { 0.708f, 0.292f, 0.170f, 0.797f, 0.131f, 0.046f, 0.3127f, 0.3290f },
        9, 18, 9  // ARIB STD-B67 (HLG) transfer
    };
    // DaVinci Wide Gamut — primaries are partly imaginary (outside the human
    // visible region, that's by design for headroom). No clean nclc mapping;
    // tag UNSPECIFIED so downstream tools fall back to defaults.
    static const OutputColorSpaceInfo dvwgLin = {
        "Linear DaVinci WideGamut", "davinci-wg-linear", true,
        { 0.8f, -0.1f, 0.265f, 1.5f, 0.0925f, -0.075f, 0.3127f, 0.3290f },
        2, 8, 2
    };
    static const OutputColorSpaceInfo dvi = {
        "DaVinci Intermediate WideGamut", "davinci-intermediate", true,
        { 0.8f, -0.1f, 0.265f, 1.5f, 0.0925f, -0.075f, 0.3127f, 0.3290f },
        2, 2, 2
    };
    // ADX10 — Academy Density Exchange (10-bit). Cineon-style log on AP0.
    static const OutputColorSpaceInfo adx = {
        "ADX10", "adx10", true,
        { 0.7347f, 0.2653f, 0.0f, 1.0f, 0.0001f, -0.077f, 0.32168f, 0.33767f },
        2, 2, 2
    };

    switch (cs) {
        case OutputColorSpace::ACEScg:              return acescg;
        case OutputColorSpace::LinearRec709:        return rec709;
        case OutputColorSpace::ACES2065_1:          return aces2065;
        case OutputColorSpace::ACEScct:             return acescct;
        case OutputColorSpace::SLog3SGamut3Cine:    return slog3;
        case OutputColorSpace::SRGB:                return srgb;
        case OutputColorSpace::Rec709Gamma22:       return rec709g22;
        case OutputColorSpace::Rec709Display:       return rec709disp;
        case OutputColorSpace::Rec2020Linear:       return rec2020lin;
        case OutputColorSpace::Rec2020PQ:           return rec2020pq;
        case OutputColorSpace::Rec2020HLG:          return rec2020hlg;
        case OutputColorSpace::DaVinciWGLinear:     return dvwgLin;
        case OutputColorSpace::DaVinciIntermediate: return dvi;
        case OutputColorSpace::ADX10:               return adx;
    }
    return acescg;
}

bool ParseOutputColorSpace(const std::string& s, OutputColorSpace& out) {
    if (s == "acescg" || s == "ACEScg" || s == "aces-cg") {
        out = OutputColorSpace::ACEScg; return true;
    }
    if (s == "rec709" || s == "Rec.709" || s == "linear-rec709" || s == "rec709-linear") {
        out = OutputColorSpace::LinearRec709; return true;
    }
    if (s == "aces2065-1" || s == "ACES2065-1" || s == "ap0") {
        out = OutputColorSpace::ACES2065_1; return true;
    }
    if (s == "acescct" || s == "ACEScct") {
        out = OutputColorSpace::ACEScct; return true;
    }
    if (s == "slog3-sgamut3cine" || s == "slog3" || s == "sony-slog3" || s == "slog3-cine") {
        out = OutputColorSpace::SLog3SGamut3Cine; return true;
    }
    if (s == "srgb" || s == "sRGB" || s == "srgb-texture") {
        out = OutputColorSpace::SRGB; return true;
    }
    if (s == "rec709-2.2" || s == "rec709-gamma22" || s == "gamma22") {
        out = OutputColorSpace::Rec709Gamma22; return true;
    }
    if (s == "rec709-display" || s == "rec709-2.4" || s == "rec709-output" || s == "gamma24") {
        out = OutputColorSpace::Rec709Display; return true;
    }
    if (s == "rec2020-linear" || s == "rec2020" || s == "linear-rec2020") {
        out = OutputColorSpace::Rec2020Linear; return true;
    }
    if (s == "rec2020-pq" || s == "rec2100-pq" || s == "hdr10" || s == "pq") {
        out = OutputColorSpace::Rec2020PQ; return true;
    }
    if (s == "rec2020-hlg" || s == "rec2100-hlg" || s == "hlg") {
        out = OutputColorSpace::Rec2020HLG; return true;
    }
    if (s == "davinci-wg-linear" || s == "linear-davinci" || s == "davinci-wg") {
        out = OutputColorSpace::DaVinciWGLinear; return true;
    }
    if (s == "davinci-intermediate" || s == "davinci-int" || s == "di") {
        out = OutputColorSpace::DaVinciIntermediate; return true;
    }
    if (s == "adx10" || s == "cineon" || s == "cineon-log") {
        out = OutputColorSpace::ADX10; return true;
    }
    return false;
}

}
}
