#include <motioncam/Debayer.hpp>
#include "Parallel.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace motioncam {
namespace color {

namespace {

void CfaChannelMap(CfaPattern p, int map[4]) {
    switch (p) {
        case CfaPattern::RGGB: { const int m[] = {0,1,1,2}; std::memcpy(map, m, sizeof(m)); break; }
        case CfaPattern::BGGR: { const int m[] = {2,1,1,0}; std::memcpy(map, m, sizeof(m)); break; }
        case CfaPattern::GRBG: { const int m[] = {1,0,2,1}; std::memcpy(map, m, sizeof(m)); break; }
        case CfaPattern::GBRG: { const int m[] = {1,2,0,1}; std::memcpy(map, m, sizeof(m)); break; }
    }
}

}

CfaPattern ParseCfa(const std::string& arrangement) {
    if (arrangement == "rggb") return CfaPattern::RGGB;
    if (arrangement == "bggr") return CfaPattern::BGGR;
    if (arrangement == "grbg") return CfaPattern::GRBG;
    if (arrangement == "gbrg") return CfaPattern::GBRG;
    throw std::runtime_error("Unknown CFA arrangement: " + arrangement);
}

void NormalizeBayer(
    const uint16_t* raw,
    float* out,
    uint32_t width,
    uint32_t height,
    const uint16_t black[4],
    double whiteLevel,
    const float wb[3],
    CfaPattern pattern)
{
    // Highlight handling is done per-pixel in NeutraliseClippedHighlights
    // (post-debayer, pre-matrix). Doing it per-bayer-position is wrong:
    // green-position saturation hits its natural ceiling (1.0) earlier than
    // red (~2.2 in daylight), so a per-position constant rewrite causes a
    // channel-asymmetric over-amplification at the saturation boundary.

    int ch[4];
    CfaChannelMap(pattern, ch);

    const float invWb[3] = { 1.0f / wb[0], 1.0f / wb[1], 1.0f / wb[2] };

    // Precompute per-CFA-position constants: blackF[i] for subtraction, and
    // scale[i] = invRange * wbMul collapses two pre-pixel multiplies into
    // one. Trivially auto-vectorisable straight-line per-row loop once we
    // pull the CFA-position branch out into per-row constants.
    float blackF[4];
    float scale[4];
    for (int i = 0; i < 4; ++i) {
        double denom = whiteLevel - double(black[i]);
        float invRange = denom > 0.0 ? float(1.0 / denom) : 0.0f;
        blackF[i] = float(black[i]);
        scale[i]  = invRange * invWb[ch[i]];
    }

    motioncam::internal::ParallelForRange(height, [&](size_t y0, size_t y1) {
        for (size_t y = y0; y < y1; ++y) {
            const size_t row = y * size_t(width);
            const int yParity = int(y & 1u) << 1;
            const float b_even = blackF[yParity | 0];
            const float b_odd  = blackF[yParity | 1];
            const float s_even = scale [yParity | 0];
            const float s_odd  = scale [yParity | 1];

            // Process pixel pairs so each iteration has fully resolved
            // constants — the compiler unrolls + vectorises this with
            // /arch:AVX2 / /fp:fast on MSVC and -O3 -mavx2 -mfma on Clang.
            uint32_t x = 0;
            for (; x + 1 < width; x += 2) {
                float v_even = float(raw[row + x    ]) - b_even;
                float v_odd  = float(raw[row + x + 1]) - b_odd;
                if (v_even < 0.0f) v_even = 0.0f;
                if (v_odd  < 0.0f) v_odd  = 0.0f;
                out[row + x    ] = v_even * s_even;
                out[row + x + 1] = v_odd  * s_odd;
            }
            // Width was odd — handle the trailing column (rare).
            if (x < width) {
                const int idx = yParity | int(x & 1u);
                float v = float(raw[row + x]) - blackF[idx];
                if (v < 0.0f) v = 0.0f;
                out[row + x] = v * scale[idx];
            }
        }
    });
}

void NeutraliseClippedHighlights(
    float* rgbInterleaved,
    uint32_t width,
    uint32_t height,
    const float wbRgb[3])
{
    // Pre-matrix per-pixel highlight de-magenta-ing.
    //
    // We work in NORMALISED de-WB space where each channel sits in [0, 1]
    // and 1.0 means that channel reached sensor saturation:
    //     n_c = post_WB_c * asShotNeutral_c  (= post_WB_c / ceiling_c)
    //
    // For each pixel:
    //   maxN = max(n_R, n_G, n_B)  — how saturated the most-loaded channel is
    //   minN = min(n_R, n_G, n_B)  — how much signal the dimmest carries
    //
    // SAT trigger (`t_max`): smoothstep(0.5, 1.0, maxN). Starts lifting at
    // 50% sat so the magenta HALO around the sun (G clipped, R/B not yet)
    // gets reach, not just the fully-clipped core. Without this the halo
    // shows up as magenta when the user pulls -5 stops in compositing.
    //
    // COLOUR GATE (`t_gate`): smoothstep(0.2, 0.4, minN). Kills the
    // recovery for pure-coloured highlights (red sodium lamp, pure green
    // laser, etc.) where one channel is at sensor sat but the others are
    // near zero — those are real saturated colours, not clipped neutrals.
    //
    // LIFT TARGET: `maxCh` (the brightest of R/G/B in cam-RGB). For a
    // neutral pixel that's partially clipped, the still-unclipped channel
    // carries the true scene brightness; lifting the clipped channel(s) to
    // match recovers neutral. For a fully-clipped pixel, max channel = R's
    // ceiling (1/min(asN)) which is the highest bound we can reconstruct
    // — we cap there since we have no signal beyond. The colour gate is
    // what makes "lerp to maxCh" safe: it suppresses firing on pure
    // coloured highlights where the dim channels really are near zero.

    if (wbRgb[0] <= 0.0f || wbRgb[1] <= 0.0f || wbRgb[2] <= 0.0f) return;

    const size_t numPixels = size_t(width) * size_t(height);

    motioncam::internal::ParallelForRange(numPixels, [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i) {
            const float r = rgbInterleaved[i * 3 + 0];
            const float g = rgbInterleaved[i * 3 + 1];
            const float b = rgbInterleaved[i * 3 + 2];

            const float rn = r * wbRgb[0];
            const float gn = g * wbRgb[1];
            const float bn = b * wbRgb[2];
            const float maxN = std::max(rn, std::max(gn, bn));
            const float minN = std::min(rn, std::min(gn, bn));

            float t_max = (maxN - 0.5f) * 2.0f;  // (maxN - 0.5) / (1.0 - 0.5)
            if (t_max <= 0.0f) continue;
            if (t_max > 1.0f) t_max = 1.0f;
            t_max = t_max * t_max * (3.0f - 2.0f * t_max);

            float t_gate = (minN - 0.2f) * 5.0f;  // smoothstep(0.2, 0.4, minN)
            if (t_gate <= 0.0f) continue;
            if (t_gate > 1.0f) t_gate = 1.0f;
            t_gate = t_gate * t_gate * (3.0f - 2.0f * t_gate);

            const float t = t_max * t_gate;
            const float maxCh = std::max(r, std::max(g, b));
            rgbInterleaved[i * 3 + 0] = r + (maxCh - r) * t;
            rgbInterleaved[i * 3 + 1] = g + (maxCh - g) * t;
            rgbInterleaved[i * 3 + 2] = b + (maxCh - b) * t;
        }
    });
}

void HighlightRolloff(
    float* rgb,
    uint32_t width,
    uint32_t height,
    float kneeStart,
    float kneeEnd)
{
    if (kneeEnd <= kneeStart) return;
    const float invKneeRange = 1.0f / (kneeEnd - kneeStart);
    const size_t numPixels = size_t(width) * size_t(height);

    motioncam::internal::ParallelForRange(numPixels, [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i) {
            float r = rgb[i * 3 + 0];
            float g = rgb[i * 3 + 1];
            float b = rgb[i * 3 + 2];
            const float maxCh = std::max(r, std::max(g, b));
            if (maxCh > kneeStart) {
                // Smooth blend factor in [0, 1] across the knee region.
                float t = (maxCh - kneeStart) * invKneeRange;
                if (t < 0.0f) t = 0.0f;
                else if (t > 1.0f) t = 1.0f;
                // Smoothstep for a softer, more cinematic shoulder.
                t = t * t * (3.0f - 2.0f * t);
                // Lerp each channel toward maxCh (neutral at this brightness).
                r = r + (maxCh - r) * t;
                g = g + (maxCh - g) * t;
                b = b + (maxCh - b) * t;
                // Hard ceiling at kneeEnd so display-space output doesn't
                // explode further.
                if (r > kneeEnd) r = kneeEnd;
                if (g > kneeEnd) g = kneeEnd;
                if (b > kneeEnd) b = kneeEnd;
                rgb[i * 3 + 0] = r;
                rgb[i * 3 + 1] = g;
                rgb[i * 3 + 2] = b;
            }
        }
    });
}

void ApplyLensShading(
    float* bayer,
    uint32_t width,
    uint32_t height,
    const float* lsm,
    uint32_t lsmWidth,
    uint32_t lsmHeight,
    CfaPattern pattern)
{
    if (lsm == nullptr || lsmWidth < 2 || lsmHeight < 2 || width == 0 || height == 0) return;

    // CFA position (yParity*2 + xParity) -> LSM channel index in [R, Gr, Gb, B].
    int cfaToLsm[4];
    switch (pattern) {
        case CfaPattern::RGGB: cfaToLsm[0]=0; cfaToLsm[1]=1; cfaToLsm[2]=2; cfaToLsm[3]=3; break;
        case CfaPattern::BGGR: cfaToLsm[0]=3; cfaToLsm[1]=2; cfaToLsm[2]=1; cfaToLsm[3]=0; break;
        case CfaPattern::GRBG: cfaToLsm[0]=1; cfaToLsm[1]=0; cfaToLsm[2]=3; cfaToLsm[3]=2; break;
        case CfaPattern::GBRG: cfaToLsm[0]=2; cfaToLsm[1]=3; cfaToLsm[2]=0; cfaToLsm[3]=1; break;
    }

    const float invXMax = 1.0f / float(width  - 1);
    const float invYMax = 1.0f / float(height - 1);
    const int lsmWMinus = int(lsmWidth)  - 1;
    const int lsmHMinus = int(lsmHeight) - 1;
    const size_t perCh = size_t(lsmWidth) * size_t(lsmHeight);

    motioncam::internal::ParallelForRange(height, [&](size_t y0, size_t y1) {
        for (size_t y = y0; y < y1; ++y) {
            const float fy = float(y) * invYMax * float(lsmHMinus);
            int yi = int(fy);
            if (yi >= lsmHMinus) yi = lsmHMinus - 1;
            const float fyf = fy - float(yi);

            const int yParity = int(y & 1u) << 1;
            const size_t row = y * size_t(width);

            for (uint32_t x = 0; x < width; ++x) {
                const float fx = float(x) * invXMax * float(lsmWMinus);
                int xi = int(fx);
                if (xi >= lsmWMinus) xi = lsmWMinus - 1;
                const float fxf = fx - float(xi);

                const int idx = yParity | int(x & 1u);
                const float* chPlane = lsm + size_t(cfaToLsm[idx]) * perCh;

                const float v00 = chPlane[size_t(yi)     * size_t(lsmWidth) + size_t(xi)];
                const float v10 = chPlane[size_t(yi)     * size_t(lsmWidth) + size_t(xi + 1)];
                const float v01 = chPlane[size_t(yi + 1) * size_t(lsmWidth) + size_t(xi)];
                const float v11 = chPlane[size_t(yi + 1) * size_t(lsmWidth) + size_t(xi + 1)];

                const float v0 = v00 * (1.0f - fxf) + v10 * fxf;
                const float v1 = v01 * (1.0f - fxf) + v11 * fxf;
                const float gain = v0 * (1.0f - fyf) + v1 * fyf;

                bayer[row + x] *= gain;
            }
        }
    });
}

void DebayerBilinear(
    const float* bayer,
    float* rgb,
    uint32_t width,
    uint32_t height,
    CfaPattern pattern)
{
    int ch[4];
    CfaChannelMap(pattern, ch);

    const int32_t w = int32_t(width);
    const int32_t h = int32_t(height);

    auto at = [&](int32_t x, int32_t y) -> float {
        x = std::max(0, std::min(w - 1, x));
        y = std::max(0, std::min(h - 1, y));
        return bayer[size_t(y) * size_t(width) + size_t(x)];
    };

    motioncam::internal::ParallelForRange(size_t(h), [&](size_t y0, size_t y1) {
        for (int32_t y = int32_t(y0); y < int32_t(y1); ++y) {
            for (int32_t x = 0; x < w; ++x) {
                const int idx = ((y & 1) << 1) | (x & 1);
                const int c = ch[idx];
                const float self = bayer[size_t(y) * size_t(width) + size_t(x)];
                float r, g, b;

                if (c == 0) {
                    r = self;
                    g = 0.25f * (at(x-1,y) + at(x+1,y) + at(x,y-1) + at(x,y+1));
                    b = 0.25f * (at(x-1,y-1) + at(x+1,y-1) + at(x-1,y+1) + at(x+1,y+1));
                }
                else if (c == 2) {
                    b = self;
                    g = 0.25f * (at(x-1,y) + at(x+1,y) + at(x,y-1) + at(x,y+1));
                    r = 0.25f * (at(x-1,y-1) + at(x+1,y-1) + at(x-1,y+1) + at(x+1,y+1));
                }
                else {
                    g = self;
                    const int hIdx = ((y & 1) << 1) | ((x + 1) & 1);
                    if (ch[hIdx] == 0) {
                        r = 0.5f * (at(x-1,y) + at(x+1,y));
                        b = 0.5f * (at(x,y-1) + at(x,y+1));
                    } else {
                        b = 0.5f * (at(x-1,y) + at(x+1,y));
                        r = 0.5f * (at(x,y-1) + at(x,y+1));
                    }
                }

                const size_t o = (size_t(y) * size_t(width) + size_t(x)) * 3;
                rgb[o + 0] = r;
                rgb[o + 1] = g;
                rgb[o + 2] = b;
            }
        }
    });
}

}
}
