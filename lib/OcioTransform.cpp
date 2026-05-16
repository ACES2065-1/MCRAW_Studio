#include <motioncam/OcioTransform.hpp>

#include <OpenColorIO/OpenColorIO.h>

#include <stdexcept>
#include <string>

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
