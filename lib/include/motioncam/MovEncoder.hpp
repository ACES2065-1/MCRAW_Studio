#ifndef MovEncoder_hpp
#define MovEncoder_hpp

#include <cstdint>
#include <memory>
#include <string>

namespace motioncam {
namespace video {

enum class Codec {
    ProRes422,
    ProRes422HQ,
    ProRes4444,
    ProRes4444XQ,
    H264,
    H265,
    H264NVENC,    // GPU-accelerated H.264 via NVIDIA NVENC; falls back to libx264.
    H265NVENC,    // GPU-accelerated H.265 via NVIDIA NVENC; falls back to libx265.
    AV1NVENC,     // GPU-accelerated AV1 via NVENC (RTX 40-series and newer).
    DNxHR_HQX,    // Avid DNxHR HQX, 10-bit YUV422 — pro intermediate.
    DNxHR_444,    // Avid DNxHR 444, 12-bit YUV444 — pro intermediate, alpha-capable.
    CineForm,     // GoPro CineForm, 10-bit YUV422 — visually-lossless intermediate.
};

// True if the codec is a GPU-accelerated NVENC variant.
bool IsNvenc(Codec c);
// True if the codec carries pix_fmt with native depth >= 10-bit.
bool IsTenBitNative(Codec c);

bool ParseCodec(const std::string& s, Codec& out);
const char* CodecName(Codec c);

struct EncodeSettings {
    std::string outputPath;
    Codec codec;
    int width;
    int height;
    int fpsNum;
    int fpsDen;
    int bitrateMbps;
    int audioSampleRate;
    int audioChannels;
    // FFmpeg muxer name: "mov" (default) or "mp4".
    // MP4 accepts H.264 / H.265 / AV1 only; ProRes / DNxHR / CineForm get rejected.
    std::string containerFormat = "mov";
    // For H.264 / H.265 paths only: when true, encode 10-bit (Main10 profile).
    // Other codecs ignore this — they have their own native bit depths.
    // libx265 + tenBit needs vcpkg's x265 multilib feature; if unavailable,
    // MovEncoder falls back to 8-bit with a stderr warning.
    bool tenBit = false;
    // QuickTime 'nclc' / MP4 color tag (FFmpeg AVCOL_* enum values).
    // Use 2 (UNSPECIFIED) for spaces with no clean mapping (ACEScg, S-Log3, etc.).
    int colorPrimaries = 2;
    int colorTrc = 2;
    int colorMatrix = 2;
};

class MovEncoder {
public:
    explicit MovEncoder(const EncodeSettings& settings);
    ~MovEncoder();

    MovEncoder(const MovEncoder&) = delete;
    MovEncoder& operator=(const MovEncoder&) = delete;

    void WriteVideoFrame(const float* rgbInterleaved);
    void WriteAudio(const int16_t* samples, int numSamplesTotal);
    void Finalize();

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

}
}

#endif
