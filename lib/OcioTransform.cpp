#include <motioncam/OcioTransform.hpp>

#include <OpenColorIO/OpenColorIO.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace OCIO = OCIO_NAMESPACE;

namespace motioncam {
namespace color {

struct OcioTransform::Impl {
    OCIO::ConstCPUProcessorRcPtr cpu;
};

namespace {
OCIO::ConstConfigRcPtr LoadStudioConfig() {
    const char* candidates[] = {
        "studio-config-latest",
        "studio-config-v2.1.0_aces-v1.3_ocio-v2.3",
        "studio-config-v1.0.0_aces-v1.3_ocio-v2.1",
        "default",
    };
    for (const char* name : candidates) {
        try {
            auto cfg = OCIO::Config::CreateFromBuiltinConfig(name);
            if (cfg) return cfg;
        } catch (const OCIO::Exception&) {
            // try next
        }
    }
    throw std::runtime_error("OCIO: no built-in config available");
}
}

OcioTransform::OcioTransform(const std::string& srcCs, const std::string& dstCs)
    : p(std::make_unique<Impl>())
{
    auto config = LoadStudioConfig();

    OCIO::ConstProcessorRcPtr proc;
    try {
        proc = config->getProcessor(srcCs.c_str(), dstCs.c_str());
    } catch (const OCIO::Exception& e) {
        std::string msg = std::string("OCIO transform failed: ") + e.what()
            + " (src='" + srcCs + "', dst='" + dstCs + "')\n"
            + "Available color spaces in built-in config (" + config->getName() + "):\n";
        const int n = config->getNumColorSpaces();
        for (int i = 0; i < n; ++i) {
            msg += "  ";
            msg += config->getColorSpaceNameByIndex(i);
            msg += "\n";
        }
        throw std::runtime_error(msg);
    }
    if (!proc) {
        throw std::runtime_error("OCIO: null processor for '" + srcCs + "' -> '" + dstCs + "'");
    }

    p->cpu = proc->getDefaultCPUProcessor();
    if (!p->cpu) {
        throw std::runtime_error("OCIO: failed to build CPU processor");
    }
}

OcioTransform::~OcioTransform() = default;

std::vector<float> BakeAcesCgToTargetLut3D(
    const std::string& dstColorSpace,
    int n, float shaperK, float shaperLo, float shaperHi)
{
    if (n < 2) throw std::runtime_error("BakeAcesCgToTargetLut3D: n must be >= 2");

    auto config = LoadStudioConfig();
    OCIO::ConstProcessorRcPtr proc;
    try {
        proc = config->getProcessor("ACEScg", dstColorSpace.c_str());
    } catch (const OCIO::Exception& e) {
        throw std::runtime_error(
            std::string("BakeAcesCgToTargetLut3D: OCIO transform failed: ")
            + e.what() + " (ACEScg -> '" + dstColorSpace + "')");
    }
    if (!proc) {
        throw std::runtime_error(
            "BakeAcesCgToTargetLut3D: null processor for ACEScg -> '"
            + dstColorSpace + "'");
    }
    auto cpu = proc->getDefaultCPUProcessor();
    if (!cpu) throw std::runtime_error("BakeAcesCgToTargetLut3D: CPU processor build failed");

    // Inverse asinh shaper: grid coordinate t in [0,1] -> ACEScg linear.
    const double sLo = std::asinh(double(shaperLo) / double(shaperK));
    const double sHi = std::asinh(double(shaperHi) / double(shaperK));
    auto tToLin = [&](double t) -> float {
        return float(double(shaperK) * std::sinh(sLo + t * (sHi - sLo)));
    };

    // Sample the cube in packed RGB (B fastest-varying), run one OCIO apply
    // over the whole grid, then expand to RGBA with alpha = 0.
    const size_t nodes = size_t(n) * size_t(n) * size_t(n);
    std::vector<float> rgb(nodes * 3);
    size_t w = 0;
    for (int iR = 0; iR < n; ++iR) {
        const float lr = tToLin(double(iR) / double(n - 1));
        for (int iG = 0; iG < n; ++iG) {
            const float lg = tToLin(double(iG) / double(n - 1));
            for (int iB = 0; iB < n; ++iB) {
                const float lb = tToLin(double(iB) / double(n - 1));
                rgb[w++] = lr;
                rgb[w++] = lg;
                rgb[w++] = lb;
            }
        }
    }

    OCIO::PackedImageDesc img(rgb.data(), static_cast<long>(nodes), 1, 3);
    cpu->apply(img);

    std::vector<float> rgba(nodes * 4);
    for (size_t i = 0; i < nodes; ++i) {
        rgba[i * 4 + 0] = rgb[i * 3 + 0];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 0.0f;
    }
    return rgba;
}

void OcioTransform::Apply(float* rgb, uint32_t width, uint32_t height) const {
    OCIO::PackedImageDesc img(
        rgb,
        static_cast<long>(width),
        static_cast<long>(height),
        3);
    p->cpu->apply(img);
}

}
}
