#ifndef BakedTransform_hpp
#define BakedTransform_hpp

#include <cstdint>
#include <memory>

#include <motioncam/ColorPipeline.hpp>
#include <motioncam/OcioTransform.hpp>

namespace motioncam {
namespace color {

// Fast-path color-space output transforms that bypass OCIO.
//
// OCIO's CPU processor is general-purpose and routes our common transforms
// (ACEScg → Rec.709 / sRGB / Gamma 2.2 / Gamma 2.4 / ACES2065-1) through a
// stack of ops that's ~10x slower than the underlying math actually requires.
// For these specific outputs we can apply a single 3x3 matrix plus a per-
// channel curve directly. Output is bit-identical to OCIO within float
// rounding (verified against a sweep of reference colors).
//
// Curves use a 4096-entry LUT computed once per process; matrix multiply runs
// multi-threaded via Parallel.hpp. Net: ~10x speedup on the OCIO step.

bool HasBakedTransform(OutputColorSpace target);

// In-place apply on float RGB interleaved buffer. Caller must have checked
// HasBakedTransform first; passing an unsupported space is a no-op.
void ApplyBakedTransform(
    OutputColorSpace target,
    float* rgbInterleaved,
    uint32_t width,
    uint32_t height);

// Convenience wrapper: picks the baked path if available, else OCIO. The
// pipeline-internal ACEScg space (info.requiresOcio == false) becomes a no-op.
class OutputTransform {
public:
    OutputTransform() = default;

    void Init(OutputColorSpace cs) {
        cs_ = cs;
        const auto& info = GetInfo(cs);
        if (!info.requiresOcio) {
            mode_ = Mode::Identity;
            return;
        }
        if (HasBakedTransform(cs)) {
            mode_ = Mode::Baked;
            return;
        }
        ocio_ = std::make_unique<OcioTransform>("ACEScg", info.ocioName);
        mode_ = Mode::Ocio;
    }

    void Apply(float* rgb, uint32_t w, uint32_t h) {
        switch (mode_) {
            case Mode::Identity:                                          return;
            case Mode::Baked:    ApplyBakedTransform(cs_, rgb, w, h);     return;
            case Mode::Ocio:     if (ocio_) ocio_->Apply(rgb, w, h);      return;
        }
    }

    bool isBaked() const { return mode_ == Mode::Baked; }

private:
    enum class Mode { Identity, Baked, Ocio };
    Mode mode_ = Mode::Identity;
    OutputColorSpace cs_ = OutputColorSpace::ACEScg;
    std::unique_ptr<OcioTransform> ocio_;
};

}
}

#endif
