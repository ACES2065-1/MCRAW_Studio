#ifndef OcioTransform_hpp
#define OcioTransform_hpp

#include <cstdint>
#include <memory>
#include <string>

namespace motioncam {
namespace color {

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
