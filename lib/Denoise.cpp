#include <motioncam/Denoise.hpp>
#include "Parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace motioncam {
namespace color {

namespace {

// BT.709 luma weights. These are exact for Rec.709 / sRGB primaries, and
// approximate-but-fine for ACEScg primaries (we just need a luma-ish channel).
constexpr float kRy = 0.2126f;
constexpr float kGy = 0.7152f;
constexpr float kBy = 0.0722f;
// Inverse matrix coefficients for Y, Cb=B-Y, Cr=R-Y.
//   R = Y + Cr
//   B = Y + Cb
//   G = (Y - kRy*R - kBy*B) / kGy   <- substitute for plain Y/Cb/Cr below.

// Build a 1D Gaussian kernel that covers ~3*sigma each side.
std::vector<float> GaussianKernel(float sigma) {
    if (sigma < 0.05f) return {1.0f};   // no-op kernel
    int radius = std::max(1, int(std::ceil(sigma * 3.0f)));
    if (radius > 32) radius = 32;
    std::vector<float> k(size_t(radius) * 2 + 1);
    const float inv2sig2 = 1.0f / (2.0f * sigma * sigma);
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        float v = std::exp(-float(i * i) * inv2sig2);
        k[size_t(i + radius)] = v;
        sum += v;
    }
    const float invSum = 1.0f / sum;
    for (auto& v : k) v *= invSum;
    return k;
}

void SeparableGaussianBlur(float* plane, uint32_t W, uint32_t H, float sigma) {
    if (sigma < 0.05f || W < 3 || H < 3) return;
    const std::vector<float> k = GaussianKernel(sigma);
    const int radius = int(k.size()) / 2;
    std::vector<float> tmp(size_t(W) * size_t(H));

    // Horizontal pass: plane -> tmp
    motioncam::internal::ParallelForRange(H, [&](size_t y0, size_t y1) {
        for (size_t y = y0; y < y1; ++y) {
            const float* row = plane + y * W;
            float* dst = tmp.data() + y * W;
            for (uint32_t x = 0; x < W; ++x) {
                float sum = 0.0f;
                for (int i = -radius; i <= radius; ++i) {
                    int xi = int(x) + i;
                    if (xi < 0) xi = 0;
                    else if (xi >= int(W)) xi = int(W) - 1;
                    sum += row[xi] * k[size_t(i + radius)];
                }
                dst[x] = sum;
            }
        }
    });

    // Vertical pass: tmp -> plane
    motioncam::internal::ParallelForRange(H, [&](size_t y0, size_t y1) {
        for (size_t y = y0; y < y1; ++y) {
            float* dst = plane + y * W;
            for (uint32_t x = 0; x < W; ++x) {
                float sum = 0.0f;
                for (int i = -radius; i <= radius; ++i) {
                    int yi = int(y) + i;
                    if (yi < 0) yi = 0;
                    else if (yi >= int(H)) yi = int(H) - 1;
                    sum += tmp[size_t(yi) * W + x] * k[size_t(i + radius)];
                }
                dst[x] = sum;
            }
        }
    });
}

// 5x5 bilateral filter on a single-channel plane. Edge-preserving — pixels
// only contribute their weight if their value is close to the center pixel
// (within ~rangeSigma in linear-light units).
//
// Performance: the naive form calls std::exp() per neighbour-pixel, which is
// ~25× per output pixel — that dominates wall-clock at 4K. We replace it with
// a 1024-entry LUT keyed on |dv| / RANGE_MAX. exp() is then called only once
// per LUT entry at setup. Speeds the filter ~5×.
void Bilateral5x5(const float* src, float* dst,
                  uint32_t W, uint32_t H,
                  float spatialSigma, float rangeSigma)
{
    if (spatialSigma < 0.05f) {
        std::memcpy(dst, src, size_t(W) * H * sizeof(float));
        return;
    }
    const int radius = 2;  // 5x5 kernel
    float spatial[5][5];
    const float invSp2 = 1.0f / (2.0f * spatialSigma * spatialSigma);
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
            spatial[dy + radius][dx + radius] =
                std::exp(-float(dx*dx + dy*dy) * invSp2);

    // Range LUT: 1024 bins covering |dv| ∈ [0, RANGE_MAX]. For Y in our
    // post-OCIO float images, values stay roughly in [0, 1] with a long tail
    // of speculars; capping at 1.0 lets dv span ±1 which more than covers the
    // realistic "different pixel" case (anything beyond it gets weight ~0).
    constexpr int LUT_N = 1024;
    constexpr float RANGE_MAX = 1.0f;
    float rangeLut[LUT_N];
    const float invRn2 = 1.0f / (2.0f * rangeSigma * rangeSigma);
    for (int i = 0; i < LUT_N; ++i) {
        const float dv = (float(i) / float(LUT_N - 1)) * RANGE_MAX;
        rangeLut[i] = std::exp(-dv * dv * invRn2);
    }
    const float lutScale = float(LUT_N - 1) / RANGE_MAX;

    motioncam::internal::ParallelForRange(H, [&](size_t y0, size_t y1) {
        for (size_t y = y0; y < y1; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                const float c = src[y * W + x];
                float wsum = 0.0f, vsum = 0.0f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int yi = int(y) + dy;
                    if (yi < 0) yi = 0;
                    else if (yi >= int(H)) yi = int(H) - 1;
                    const float* row = src + size_t(yi) * W;
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int xi = int(x) + dx;
                        if (xi < 0) xi = 0;
                        else if (xi >= int(W)) xi = int(W) - 1;
                        const float v = row[xi];
                        float adv = v - c;
                        if (adv < 0.0f) adv = -adv;
                        if (adv > RANGE_MAX) adv = RANGE_MAX;
                        const int idx = int(adv * lutScale);
                        const float w = spatial[dy + radius][dx + radius]
                                      * rangeLut[idx];
                        wsum += w;
                        vsum += w * v;
                    }
                }
                dst[y * W + x] = (wsum > 0.0f) ? (vsum / wsum) : c;
            }
        }
    });
}

}  // namespace

void DenoiseRgb(
    float* rgb,
    uint32_t width,
    uint32_t height,
    int chromaStrength,
    int lumaStrength)
{
    chromaStrength = std::clamp(chromaStrength, 0, 100);
    lumaStrength   = std::clamp(lumaStrength,   0, 100);
    if (chromaStrength == 0 && lumaStrength == 0) return;
    if (width < 3 || height < 3) return;

    const size_t N = size_t(width) * size_t(height);
    std::vector<float> Y(N), Cb(N), Cr(N);

    // RGB -> Y / Cb=B-Y / Cr=R-Y
    motioncam::internal::ParallelForRange(height, [&](size_t y0, size_t y1) {
        for (size_t y = y0; y < y1; ++y) {
            const float* p = rgb + y * width * 3;
            float* py  = Y.data()  + y * width;
            float* pcb = Cb.data() + y * width;
            float* pcr = Cr.data() + y * width;
            for (uint32_t x = 0; x < width; ++x) {
                const float r = p[0], g = p[1], b = p[2];
                const float yv = kRy * r + kGy * g + kBy * b;
                py[x]  = yv;
                pcb[x] = b - yv;
                pcr[x] = r - yv;
                p += 3;
            }
        }
    });

    // Chroma denoise: separable Gaussian on Cb/Cr.
    // sigma 0..4 pixels at strength 0..100 — enough to obliterate phone chroma
    // noise at full strength while staying fast at moderate settings (sigma=2
    // at strength=50 → 13-tap kernel, ~10ms multi-threaded at 4K).
    if (chromaStrength > 0) {
        const float chromaSigma = float(chromaStrength) * 0.04f;
        SeparableGaussianBlur(Cb.data(), width, height, chromaSigma);
        SeparableGaussianBlur(Cr.data(), width, height, chromaSigma);
    }

    // Luma denoise: gentler edge-preserving bilateral on Y.
    // spatialSigma 0..2.5 (kernel stays 5x5), rangeSigma 0.005..0.055 — at
    // strength 100 it'll smooth flat areas considerably while leaving real
    // edges sharp.
    if (lumaStrength > 0) {
        const float ssig = float(lumaStrength) * 0.025f;
        const float rsig = 0.005f + float(lumaStrength) * 0.0005f;
        std::vector<float> Yf(N);
        Bilateral5x5(Y.data(), Yf.data(), width, height, ssig, rsig);
        Y.swap(Yf);
    }

    // Recombine. From Y, Cb=B-Y, Cr=R-Y:
    //   R = Y + Cr
    //   B = Y + Cb
    //   G = (Y - kRy*R - kBy*B) / kGy
    motioncam::internal::ParallelForRange(height, [&](size_t y0, size_t y1) {
        for (size_t y = y0; y < y1; ++y) {
            float* p = rgb + y * width * 3;
            const float* py  = Y.data()  + y * width;
            const float* pcb = Cb.data() + y * width;
            const float* pcr = Cr.data() + y * width;
            for (uint32_t x = 0; x < width; ++x) {
                const float yv = py[x];
                const float cb = pcb[x];
                const float cr = pcr[x];
                const float r = yv + cr;
                const float b = yv + cb;
                const float g = (yv - kRy * r - kBy * b) / kGy;
                p[0] = r;
                p[1] = g;
                p[2] = b;
                p += 3;
            }
        }
    });
}

}
}
