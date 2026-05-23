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

    // ----- Tier 2.1 Phase C.2: GPU bayer pipeline ------------------------
    //
    // Optional fast path that does the whole bayer -> NV12 chain on the
    // GPU and feeds the result straight into NVENC. Eliminates the CPU
    // NormalizeBayer + LSM + Debayer + Matrix + sws_scale steps.
    //
    // To use:
    //   1. Construct MovEncoder as usual (chooses NVENC, opens the
    //      CUDA hwframes context per Phase A/B if MCRAW_GPU_YUV=1).
    //   2. Call EnableGpuBayerPipeline() exactly once after the
    //      constructor returns. Returns true only if:
    //        - The codec is an NVENC variant (h264_nvenc / h265_nvenc /
    //          av1_nvenc) AND the Phase B kernel is active.
    //        - The output target has a BakedTransform (sRGB / Rec.709
    //          family / ACEScg / ACES2065-1). OCIO-only targets fall
    //          back to the CPU pipeline.
    //        - This binary was compiled with CUDA support.
    //      Returns false otherwise; the caller should continue using
    //      WriteVideoFrame(rgb) and run the CPU pipeline itself.
    //   3. For each frame, call WriteVideoFrameFromBayer() instead of
    //      WriteVideoFrame(). The encoder owns the entire processing
    //      chain from raw bayer to NVENC-encoded packet.
    struct GpuBayerSetup {
        // OutputColorSpace value cast to int. A BakedTransform target
        // (ACEScg / LinearRec709 / ACES2065-1 / Rec709Gamma22 /
        // Rec709Display / SRGB) runs entirely as matrix + curve. An
        // OCIO-only target (ACEScct / S-Log3 / DaVinci / Rec.2020 PQ-HLG /
        // ...) instead bakes a 3D LUT — see ocioColorSpace below.
        int      targetColorSpace;

        // From container metadata. ForwardMatrix2 is DNG's camera->XYZ_D50
        // matrix (row-major). blackPerPosition / whiteLevel set the
        // sensor's dynamic range. cfaPattern is CfaPattern as int (0=RGGB,
        // 1=BGGR, 2=GRBG, 3=GBRG).
        float    forwardMatrix2[9];
        uint16_t blackPerPosition[4];
        double   whiteLevel;
        int      cfaPattern;

        // Phase D: OCIO config colour-space name (e.g. "ACEScct"). Only
        // consulted when targetColorSpace has no BakedTransform — then the
        // encoder bakes an ACEScg->target 3D LUT and runs it on the GPU
        // after the cam->ACEScg matrix. Leave empty for baked targets; an
        // empty name on a non-baked target makes EnableGpuBayerPipeline
        // fall back to the CPU pipeline.
        std::string ocioColorSpace;

        // Phase E.3: highlight handling on the GPU (matches ColorPipeline).
        // highlightRecovery enables the pre-matrix NeutraliseClippedHighlights
        // pass; displayEncoded (= IsDisplayEncoded(target)) additionally
        // enables the post-transform 1.0->1.4 rolloff. With these set the GPU
        // path no longer has to be skipped for highlight-recovery renders.
        bool highlightRecovery = false;
        bool displayEncoded    = false;
    };

    bool EnableGpuBayerPipeline(const GpuBayerSetup& setup);
    bool HasGpuBayerPipeline() const;

    // Per-frame call when the bayer pipeline is active. `bayer` points at
    // width*height u16 raw bayer (the bytes the decoder hands you).
    // `wb` is asShotNeutral (3 floats). `lsm` is an optional channel-first
    // 4 * lsmW * lsmH float lens-shading-map (pass nullptr to skip).
    //
    // Throws std::runtime_error if EnableGpuBayerPipeline() didn't succeed
    // — the caller is expected to check HasGpuBayerPipeline() first.
    void WriteVideoFrameFromBayer(
        const uint16_t* bayer,
        const float     wb[3],
        const float*    lsm,
        int             lsmWidth,
        int             lsmHeight);

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

}
}

#endif
