#include <motioncam/BakedTransform.hpp>
#include "Parallel.hpp"

#include <array>
#include <cmath>
#include <mutex>

namespace motioncam {
namespace color {

namespace {

// ----- Matrices --------------------------------------------------------------
//
// Each matrix M is row-major and applied as: out = M * in (column vectors).
// All matrices map FROM ACEScg (AP1, linear, scene-referred) to the named
// destination's linear scene-referred form. The curve (if any) is applied
// after the matrix.

// AP1 (ACEScg) → Rec.709 primaries. From the ACES specification.
constexpr float M_AP1_to_Rec709[9] = {
     1.7050514f, -0.6217908f, -0.0832606f,
    -0.1302561f,  1.1408047f, -0.0105486f,
    -0.0240083f, -0.1289693f,  1.1529777f,
};

// AP1 → AP0 (ACES2065-1). From the ACES specification.
constexpr float M_AP1_to_AP0[9] = {
     0.6954522f,  0.1406787f,  0.1638691f,
     0.0447946f,  0.8596711f,  0.0955343f,
    -0.0055258f,  0.0040252f,  1.0015007f,
};

// Identity for "ACEScg" output (no transform needed) — kept for symmetry.
constexpr float M_Identity[9] = {
     1.0f, 0.0f, 0.0f,
     0.0f, 1.0f, 0.0f,
     0.0f, 0.0f, 1.0f,
};

// ----- Curves ----------------------------------------------------------------

enum class Curve {
    None,        // linear scene-referred, no encoding
    Gamma22,     // power 1/2.2
    Gamma24,     // power 1/2.4 (a.k.a. BT.1886-ish display gamma)
    SRGB,        // piecewise sRGB encoding (linear toe + ~2.4 power above 0.0031308)
};

// LUT-backed curve evaluator. We sample the curve at LUT_N points across
// [0, 1] and linearly interpolate. Negative inputs use symmetric power so
// the transform stays well-defined for HDR-ish content.
constexpr int   LUT_N    = 4097;
constexpr float LUT_MAXV = 1.0f;

struct CurveLut {
    std::array<float, LUT_N> data;
    bool ready = false;
};

float ApplyCurveScalar(Curve c, float v) {
    switch (c) {
        case Curve::None:    return v;
        case Curve::Gamma22: return std::pow(v, 1.0f / 2.2f);
        case Curve::Gamma24: return std::pow(v, 1.0f / 2.4f);
        case Curve::SRGB:
            if (v <= 0.0031308f) return 12.92f * v;
            return 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    }
    return v;
}

const CurveLut& GetLut(Curve c) {
    static CurveLut luts[4];
    static std::once_flag once[4];
    int idx = int(c);
    std::call_once(once[idx], [idx]() {
        Curve cv = static_cast<Curve>(idx);
        for (int i = 0; i < LUT_N; ++i) {
            const float v = (float(i) / float(LUT_N - 1)) * LUT_MAXV;
            luts[idx].data[i] = ApplyCurveScalar(cv, v);
        }
        luts[idx].ready = true;
    });
    return luts[idx];
}

// Evaluate the curve at v with linear interpolation between LUT bins.
// Out-of-range inputs use symmetric continuation (negative) and direct
// pow() fall-through (above 1.0) so we never silently clamp HDR pixels.
inline float ApplyCurveLut(const CurveLut& lut, Curve c, float v) {
    if (c == Curve::None) return v;

    if (v < 0.0f) return -ApplyCurveLut(lut, c, -v);
    if (v >= LUT_MAXV) {
        // Above-1 input: fall back to scalar evaluator. Rare in practice (we
        // expect normalized [0,1] post-tone-map), but correct for VFX content.
        return ApplyCurveScalar(c, v);
    }

    const float pos = v * float(LUT_N - 1) / LUT_MAXV;
    const int   i0  = int(pos);
    const int   i1  = (i0 + 1 < LUT_N) ? i0 + 1 : i0;
    const float f   = pos - float(i0);
    return lut.data[i0] * (1.0f - f) + lut.data[i1] * f;
}

// ----- Plan table ------------------------------------------------------------

struct Plan {
    const float* matrix;   // nullptr → identity (skip mul)
    Curve        curve;
};

bool LookupPlan(OutputColorSpace target, Plan& out) {
    switch (target) {
        case OutputColorSpace::ACEScg:
            // Identity — pipeline already produces ACEScg, baked is a no-op.
            out = {nullptr, Curve::None}; return true;
        case OutputColorSpace::LinearRec709:
            out = {M_AP1_to_Rec709, Curve::None}; return true;
        case OutputColorSpace::ACES2065_1:
            out = {M_AP1_to_AP0, Curve::None}; return true;
        case OutputColorSpace::Rec709Gamma22:
            out = {M_AP1_to_Rec709, Curve::Gamma22}; return true;
        case OutputColorSpace::Rec709Display:    // gamma 2.4
            out = {M_AP1_to_Rec709, Curve::Gamma24}; return true;
        case OutputColorSpace::SRGB:
            out = {M_AP1_to_Rec709, Curve::SRGB}; return true;
        // Log-encoded and HDR / wide-gamut targets stay on the OCIO path —
        // their curves and matrices are too involved to bake quickly and
        // they're not common enough to be worth the engineering today.
        case OutputColorSpace::ACEScct:
        case OutputColorSpace::SLog3SGamut3Cine:
        case OutputColorSpace::Rec2020Linear:
        case OutputColorSpace::Rec2020PQ:
        case OutputColorSpace::Rec2020HLG:
        case OutputColorSpace::DaVinciWGLinear:
        case OutputColorSpace::DaVinciIntermediate:
        case OutputColorSpace::ADX10:
            return false;
    }
    return false;
}

}  // namespace

bool HasBakedTransform(OutputColorSpace target) {
    Plan p{};
    return LookupPlan(target, p);
}

void ApplyBakedTransform(
    OutputColorSpace target,
    float* rgb,
    uint32_t width,
    uint32_t height)
{
    Plan plan{};
    if (!LookupPlan(target, plan)) return;
    const size_t numPixels = size_t(width) * size_t(height);
    if (numPixels == 0) return;

    // Identity + None = early-out.
    if (plan.matrix == nullptr && plan.curve == Curve::None) return;
    if (plan.matrix == M_Identity) plan.matrix = nullptr;

    const CurveLut& lut = (plan.curve == Curve::None)
        ? GetLut(Curve::None)         // unused but avoids branch in inner loop
        : GetLut(plan.curve);
    const Curve curve = plan.curve;
    const float* M = plan.matrix;

    motioncam::internal::ParallelForRange(numPixels, [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i) {
            float r = rgb[i*3 + 0];
            float g = rgb[i*3 + 1];
            float b = rgb[i*3 + 2];

            if (M) {
                const float nr = M[0]*r + M[1]*g + M[2]*b;
                const float ng = M[3]*r + M[4]*g + M[5]*b;
                const float nb = M[6]*r + M[7]*g + M[8]*b;
                r = nr; g = ng; b = nb;
            }

            if (curve != Curve::None) {
                r = ApplyCurveLut(lut, curve, r);
                g = ApplyCurveLut(lut, curve, g);
                b = ApplyCurveLut(lut, curve, b);
            }

            rgb[i*3 + 0] = r;
            rgb[i*3 + 1] = g;
            rgb[i*3 + 2] = b;
        }
    });
}

}
}
