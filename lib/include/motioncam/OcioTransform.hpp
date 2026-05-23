#ifndef OcioTransform_hpp
#define OcioTransform_hpp

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace motioncam {
namespace color {

// Bake the OCIO "ACEScg -> dstColorSpace" transform into an n^3 float RGBA
// cube, addressed through an asinh input shaper:
//
//   s(L) = asinh(L / shaperK)
//   t    = (s(L) - s(shaperLo)) / (s(shaperHi) - s(shaperLo))   in [0,1]
//   L    = shaperK * sinh(s(shaperLo) + t * (s(shaperHi) - s(shaperLo)))  (inverse)
//
// so the LUT covers ACEScg's unbounded, slightly-negative scene-linear range.
// Output layout matches the GPU uploader (cuda::SetupLut3D): the B axis is
// fastest-varying, index = ((iR*n + iG)*n + iB)*4, alpha = 0. The shaper
// params come from cuda::kLutShaper* / kLutSize so host and device agree.
//
// Throws std::runtime_error on OCIO failure (unknown space, etc.).
std::vector<float> BakeAcesCgToTargetLut3D(
    const std::string& dstColorSpace,
    int   n,
    float shaperK,
    float shaperLo,
    float shaperHi);

// Wraps an OCIO color-space conversion as a CPU processor.
// Loads the OCIO 2.x built-in studio-config which ships ACES + camera log spaces
// (S-Log3, LogC, V-Log, Rec.709, sRGB, etc.) — no external config file needed.
class OcioTransform {
public:
    OcioTransform(const std::string& srcColorSpace, const std::string& dstColorSpace);
    ~OcioTransform();

    OcioTransform(const OcioTransform&) = delete;
    OcioTransform& operator=(const OcioTransform&) = delete;

    void Apply(float* rgbInterleaved, uint32_t width, uint32_t height) const;

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

}
}

#endif
