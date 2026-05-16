#include <motioncam/ExrWriter.hpp>

#include <ImfOutputFile.h>
#include <ImfChannelList.h>
#include <ImfHeader.h>
#include <ImfFrameBuffer.h>
#include <ImfCompression.h>
#include <ImfStringAttribute.h>
#include <ImfChromaticities.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImathVec.h>
#include <half.h>

#include <algorithm>
#include <cctype>
#include <vector>

namespace motioncam {
namespace color {

static Imf::Compression ToImfCompression(ExrCompression c) {
    switch (c) {
        case ExrCompression::None:  return Imf::NO_COMPRESSION;
        case ExrCompression::RLE:   return Imf::RLE_COMPRESSION;
        case ExrCompression::ZIPS:  return Imf::ZIPS_COMPRESSION;
        case ExrCompression::ZIP:   return Imf::ZIP_COMPRESSION;
        case ExrCompression::PIZ:   return Imf::PIZ_COMPRESSION;
        case ExrCompression::PXR24: return Imf::PXR24_COMPRESSION;
        case ExrCompression::B44:   return Imf::B44_COMPRESSION;
        case ExrCompression::B44A:  return Imf::B44A_COMPRESSION;
        case ExrCompression::DWAA:  return Imf::DWAA_COMPRESSION;
        case ExrCompression::DWAB:  return Imf::DWAB_COMPRESSION;
    }
    return Imf::ZIP_COMPRESSION;
}

const char* ExrCompressionName(ExrCompression c) {
    switch (c) {
        case ExrCompression::None:  return "none";
        case ExrCompression::RLE:   return "rle";
        case ExrCompression::ZIPS:  return "zips";
        case ExrCompression::ZIP:   return "zip";
        case ExrCompression::PIZ:   return "piz";
        case ExrCompression::PXR24: return "pxr24";
        case ExrCompression::B44:   return "b44";
        case ExrCompression::B44A:  return "b44a";
        case ExrCompression::DWAA:  return "dwaa";
        case ExrCompression::DWAB:  return "dwab";
    }
    return "zip";
}

bool ParseExrCompression(const std::string& s, ExrCompression& out) {
    std::string v = s;
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c){ return char(std::tolower(c)); });
    if (v == "none" || v == "uncompressed") { out = ExrCompression::None; return true; }
    if (v == "rle")   { out = ExrCompression::RLE;   return true; }
    if (v == "zips")  { out = ExrCompression::ZIPS;  return true; }
    if (v == "zip")   { out = ExrCompression::ZIP;   return true; }
    if (v == "piz")   { out = ExrCompression::PIZ;   return true; }
    if (v == "pxr24") { out = ExrCompression::PXR24; return true; }
    if (v == "b44")   { out = ExrCompression::B44;   return true; }
    if (v == "b44a")  { out = ExrCompression::B44A;  return true; }
    if (v == "dwaa")  { out = ExrCompression::DWAA;  return true; }
    if (v == "dwab")  { out = ExrCompression::DWAB;  return true; }
    return false;
}

void WriteExrHalfRgb(
    const std::string& path,
    const float* rgb,
    uint32_t width,
    uint32_t height,
    const std::string& ocioColorSpace,
    const float* chroma,
    ExrCompression compression)
{
    using namespace Imf;
    using namespace Imath;

    const size_t numPixels = size_t(width) * size_t(height);
    std::vector<half> hR(numPixels), hG(numPixels), hB(numPixels);
    for (size_t i = 0; i < numPixels; ++i) {
        hR[i] = rgb[i*3 + 0];
        hG[i] = rgb[i*3 + 1];
        hB[i] = rgb[i*3 + 2];
    }

    const int w = static_cast<int>(width);
    const int h = static_cast<int>(height);
    Header header(w, h);
    header.channels().insert("R", Channel(HALF));
    header.channels().insert("G", Channel(HALF));
    header.channels().insert("B", Channel(HALF));
    header.compression() = ToImfCompression(compression);

    if (!ocioColorSpace.empty()) {
        header.insert("ocio:colorSpace", StringAttribute(ocioColorSpace));
    }

    if (chroma != nullptr) {
        Chromaticities c;
        c.red   = V2f(chroma[0], chroma[1]);
        c.green = V2f(chroma[2], chroma[3]);
        c.blue  = V2f(chroma[4], chroma[5]);
        c.white = V2f(chroma[6], chroma[7]);
        header.insert("chromaticities", ChromaticitiesAttribute(c));
    }

    const size_t rowStride = sizeof(half) * width;

    FrameBuffer fb;
    fb.insert("R", Slice(HALF, reinterpret_cast<char*>(hR.data()), sizeof(half), rowStride));
    fb.insert("G", Slice(HALF, reinterpret_cast<char*>(hG.data()), sizeof(half), rowStride));
    fb.insert("B", Slice(HALF, reinterpret_cast<char*>(hB.data()), sizeof(half), rowStride));

    OutputFile file(path.c_str(), header);
    file.setFrameBuffer(fb);
    file.writePixels(h);
}

}
}
