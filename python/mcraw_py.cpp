#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <motioncam/Decoder.hpp>
#include <motioncam/ColorPipeline.hpp>
#include <motioncam/BakedTransform.hpp>
#include <motioncam/Denoise.hpp>
#include <motioncam/ExrWriter.hpp>
#include <motioncam/OcioTransform.hpp>
#include <motioncam/MovEncoder.hpp>
#include <motioncam/FrameTiming.hpp>
#include <motioncam/Trimmer.hpp>

#if MCRAW_HAVE_CUDA
#include <motioncam/CudaHwHandoff.hpp>
#endif

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
namespace mc = motioncam;
namespace mcc = motioncam::color;
namespace mcv = motioncam::video;

namespace {

py::object JsonToPy(const nlohmann::json& j) {
    if (j.is_null()) return py::none();
    if (j.is_boolean()) return py::bool_(j.get<bool>());
    if (j.is_number_integer()) return py::int_(j.get<int64_t>());
    if (j.is_number_unsigned()) return py::int_(j.get<uint64_t>());
    if (j.is_number_float()) return py::float_(j.get<double>());
    if (j.is_string()) return py::str(j.get<std::string>());
    if (j.is_array()) {
        py::list lst;
        for (auto& el : j) lst.append(JsonToPy(el));
        return lst;
    }
    if (j.is_object()) {
        py::dict d;
        for (auto it = j.begin(); it != j.end(); ++it) {
            d[py::str(it.key())] = JsonToPy(it.value());
        }
        return d;
    }
    return py::none();
}

// Cell-wise Bayer subsample for the real-time preview proxy. Copies every
// `bin`-th 2x2 CFA cell intact, so the result is a smaller mosaic with the
// SAME CFA pattern, black levels and white balance — the existing debayer /
// colour / lens-shading code (CPU and GPU) runs unchanged, just on far fewer
// pixels (a 1/N proxy costs ~1/N^2 of the per-pixel work). This is nearest-cell
// decimation: fast and a little aliased, which is exactly what a preview wants.
// `bin` >= 2; `outW/outH` return the reduced dimensions. If the frame is too
// small to yield even one output cell, `dst` is left empty and the caller keeps
// full resolution.
void SubsampleBayerCells(const uint16_t* src, uint32_t W, uint32_t H, int bin,
                         std::vector<uint16_t>& dst,
                         uint32_t& outW, uint32_t& outH) {
    const uint32_t oCellsW = (W / 2u) / uint32_t(bin);
    const uint32_t oCellsH = (H / 2u) / uint32_t(bin);
    if (oCellsW == 0u || oCellsH == 0u) { dst.clear(); outW = W; outH = H; return; }
    outW = oCellsW * 2u;
    outH = oCellsH * 2u;
    dst.resize(size_t(outW) * size_t(outH));
    const uint32_t step = uint32_t(bin) * 2u;   // source pixels spanned per cell
    for (uint32_t cy = 0; cy < oCellsH; ++cy) {
        const uint32_t sy = cy * step;
        const uint16_t* s0 = src + size_t(sy)        * W;
        const uint16_t* s1 = src + size_t(sy + 1u)   * W;
        uint16_t* d0 = dst.data() + size_t(cy * 2u)      * outW;
        uint16_t* d1 = dst.data() + size_t(cy * 2u + 1u) * outW;
        for (uint32_t cx = 0; cx < oCellsW; ++cx) {
            const uint32_t sx = cx * step;
            const uint32_t dx = cx * 2u;
            d0[dx]      = s0[sx];        // CFA (0,0)
            d0[dx + 1u] = s0[sx + 1u];  // CFA (1,0)
            d1[dx]      = s1[sx];        // CFA (0,1)
            d1[dx + 1u] = s1[sx + 1u];  // CFA (1,1)
        }
    }
}

#if MCRAW_HAVE_CUDA
// Build the GPU bayer-pipeline constants for a BakedTransform target
// (cam->output matrix + curve). Returns false when `cs` is OCIO-only (no
// baked transform) — the caller should then use the CPU path. Mirrors the
// math in ColorPipeline.cpp / BakedTransform.cpp (same chain the CPU uses).
bool BuildBakedBayerConstants(const mcc::FrameParams& params,
                              mcc::OutputColorSpace cs,
                              motioncam::cuda::BayerPipelineConstants& C) {
    if (!mcc::HasBakedTransform(cs)) return false;

    constexpr float Bradford_D50_to_D60[9] = {
         0.96766f, -0.01686f,  0.04424f,
        -0.02099f,  1.00778f,  0.01477f,
         0.00853f, -0.01415f,  1.22963f,
    };
    constexpr float XYZ_D60_to_AP1[9] = {
         1.6410233797f, -0.3248032942f, -0.2364246952f,
        -0.6636628587f,  1.6153315917f,  0.0167563477f,
         0.0117218943f, -0.0082844420f,  0.9883948585f,
    };
    auto matMul = [](const float A[9], const float B[9], float out[9]) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                out[i*3+j] = A[i*3+0]*B[j] + A[i*3+1]*B[3+j] + A[i*3+2]*B[6+j];
    };
    float tmp[9], camToAcescg[9];
    matMul(Bradford_D50_to_D60, params.forwardMatrix2, tmp);
    matMul(XYZ_D60_to_AP1, tmp, camToAcescg);

    static const float M_Rec709[9] = {
         1.7050514f, -0.6217908f, -0.0832606f,
        -0.1302561f,  1.1408047f, -0.0105486f,
        -0.0240083f, -0.1289693f,  1.1529777f,
    };
    static const float M_AP0[9] = {
         0.6954522f,  0.1406787f,  0.1638691f,
         0.0447946f,  0.8596711f,  0.0955343f,
        -0.0055258f,  0.0040252f,  1.0015007f,
    };
    const float* bakedM = nullptr;
    int curveCode = 0;  // 0 None, 1 Gamma22, 2 Gamma24, 3 SRGB
    switch (cs) {
        case mcc::OutputColorSpace::ACEScg:        bakedM = nullptr;   curveCode = 0; break;
        case mcc::OutputColorSpace::LinearRec709:  bakedM = M_Rec709;  curveCode = 0; break;
        case mcc::OutputColorSpace::ACES2065_1:    bakedM = M_AP0;     curveCode = 0; break;
        case mcc::OutputColorSpace::Rec709Gamma22: bakedM = M_Rec709;  curveCode = 1; break;
        case mcc::OutputColorSpace::Rec709Display: bakedM = M_Rec709;  curveCode = 2; break;
        case mcc::OutputColorSpace::SRGB:          bakedM = M_Rec709;  curveCode = 3; break;
        default: return false;
    }
    float fullMatrix[9];
    if (bakedM) matMul(bakedM, camToAcescg, fullMatrix);
    else        std::memcpy(fullMatrix, camToAcescg, sizeof(fullMatrix));

    int chMap[4], cfaToLsm[4];
    switch (params.cfa) {
        case mcc::CfaPattern::RGGB:
            chMap[0]=0; chMap[1]=1; chMap[2]=1; chMap[3]=2;
            cfaToLsm[0]=0; cfaToLsm[1]=1; cfaToLsm[2]=2; cfaToLsm[3]=3; break;
        case mcc::CfaPattern::BGGR:
            chMap[0]=2; chMap[1]=1; chMap[2]=1; chMap[3]=0;
            cfaToLsm[0]=3; cfaToLsm[1]=2; cfaToLsm[2]=1; cfaToLsm[3]=0; break;
        case mcc::CfaPattern::GRBG:
            chMap[0]=1; chMap[1]=0; chMap[2]=2; chMap[3]=1;
            cfaToLsm[0]=1; cfaToLsm[1]=0; cfaToLsm[2]=3; cfaToLsm[3]=2; break;
        case mcc::CfaPattern::GBRG:
            chMap[0]=1; chMap[1]=2; chMap[2]=0; chMap[3]=1;
            cfaToLsm[0]=2; cfaToLsm[1]=3; cfaToLsm[2]=0; cfaToLsm[3]=1; break;
    }

    std::memcpy(C.cam_to_output, fullMatrix, sizeof(fullMatrix));
    for (int i = 0; i < 4; ++i) {
        C.black[i] = params.blackPerPosition[i];
        const double denom = params.whiteLevel - double(params.blackPerPosition[i]);
        C.inv_range[i] = denom > 0.0 ? float(1.0 / denom) : 0.0f;
        C.cfa_channel[i] = chMap[i];
        C.cfa_to_lsm[i]  = cfaToLsm[i];
    }
    C.curve  = curveCode;
    C.width  = int(params.width);
    C.height = int(params.height);
    if (!params.lensShadingMap.empty() &&
        params.lsmWidth >= 2 && params.lsmHeight >= 2) {
        C.lsm_w    = int(params.lsmWidth);
        C.lsm_h    = int(params.lsmHeight);
        C.lsm_host = params.lensShadingMap.data();
    } else {
        C.lsm_w = 0; C.lsm_h = 0; C.lsm_host = nullptr;
    }
    return true;
}
#endif  // MCRAW_HAVE_CUDA

bool EndsWithExt(const std::string& s, const std::string& ext) {
    if (s.size() < ext.size()) return false;
    auto tail = s.substr(s.size() - ext.size());
    for (auto& c : tail) c = char(std::tolower(c));
    return tail == ext;
}
bool EndsWithMov(const std::string& s) { return EndsWithExt(s, ".mov"); }
bool EndsWithMp4(const std::string& s) { return EndsWithExt(s, ".mp4"); }

}

class PyDecoder {
public:
    explicit PyDecoder(const std::string& path)
        : d_(std::make_unique<mc::Decoder>(path))
    {
        const auto& f = d_->getFrames();
        timestamps_.assign(f.begin(), f.end());
    }

    int frame_count() const { return int(timestamps_.size()); }
    const std::vector<int64_t>& frames() const { return timestamps_; }
    int audio_sample_rate() const { return d_->audioSampleRateHz(); }
    int audio_channels() const { return d_->numAudioChannels(); }
    py::object container_metadata() const { return JsonToPy(d_->getContainerMetadata()); }

    py::tuple load_bayer(int64_t timestamp) {
        std::vector<uint8_t> rawBuf;
        nlohmann::json frameMeta;
        {
            py::gil_scoped_release release;
            d_->loadFrame(timestamp, rawBuf, frameMeta);
        }
        const int width  = frameMeta["width"].get<int>();
        const int height = frameMeta["height"].get<int>();

        py::array_t<uint16_t> arr({ height, width });
        std::memcpy(arr.mutable_data(), rawBuf.data(),
                    size_t(width) * size_t(height) * sizeof(uint16_t));
        return py::make_tuple(arr, JsonToPy(frameMeta));
    }

    py::array_t<float> process_frame(int64_t timestamp, const std::string& colorspace,
                                     bool highlight_recovery, bool bake_vignette) {
        mcc::OutputColorSpace cs;
        if (!mcc::ParseOutputColorSpace(colorspace, cs)) {
            throw std::runtime_error("unknown colorspace: " + colorspace);
        }

        std::vector<uint8_t> rawBuf;
        nlohmann::json frameMeta;
        std::vector<float> rgbOut;
        uint32_t width = 0, height = 0;
        {
            py::gil_scoped_release release;
            d_->loadFrame(timestamp, rawBuf, frameMeta);
            auto params = mcc::BuildFrameParams(frameMeta, d_->getContainerMetadata());
            width = params.width;
            height = params.height;
            const uint16_t* raw = reinterpret_cast<const uint16_t*>(rawBuf.data());
            mcc::ProcessFrame(raw, params, cs, rgbOut, highlight_recovery, bake_vignette);

            EnsureXform(cs);
            cachedXform_.Apply(rgbOut.data(), width, height);
            if (highlight_recovery && mcc::IsDisplayEncoded(cs)) {
                mcc::HighlightRolloff(rgbOut.data(), width, height);
            }
        }

        py::array_t<float> arr({ int(height), int(width), 3 });
        std::memcpy(arr.mutable_data(), rgbOut.data(), rgbOut.size() * sizeof(float));
        return arr;
    }

    // numpy-free counterpart of process_frame: returns ([0,255]-clamped RGB888
    // packed bytes, height, width). Lets callers (e.g. PyInstaller bundles that
    // don't ship numpy) build a QImage directly without importing numpy.
    py::tuple process_frame_rgb24(int64_t timestamp, const std::string& colorspace,
                                  bool highlight_recovery, bool bake_vignette,
                                  bool prefer_gpu, int preview_bin) {
        return Rgb24Impl(timestamp, colorspace, highlight_recovery,
                         bake_vignette, prefer_gpu, preview_bin, false);
    }

    // Scoped sibling for the player's scopes panel: returns
    // (rgb_bytes, height, width, hist_bytes, clip_lo, clip_hi) where
    // hist_bytes is 3*256 little-endian uint32 (R,G,B bins over the
    // pre-quantize float output) and clip_lo/clip_hi are the fraction of
    // pixels with any channel <= 0.0 / >= 1.0 BEFORE the u8 clamp.
    py::tuple process_frame_rgb24_scopes(int64_t timestamp, const std::string& colorspace,
                                         bool highlight_recovery, bool bake_vignette,
                                         bool prefer_gpu, int preview_bin) {
        return Rgb24Impl(timestamp, colorspace, highlight_recovery,
                         bake_vignette, prefer_gpu, preview_bin, true);
    }

private:
    py::tuple Rgb24Impl(int64_t timestamp, const std::string& colorspace,
                        bool highlight_recovery, bool bake_vignette,
                        bool prefer_gpu, int preview_bin, bool want_scopes) {
        mcc::OutputColorSpace cs;
        if (!mcc::ParseOutputColorSpace(colorspace, cs))
            throw std::runtime_error("unknown color space: " + colorspace);

        constexpr int kSlots = 3 * 256 + 2;
        std::vector<uint32_t> scopes;
        if (want_scopes) scopes.assign(kSlots, 0u);

        std::vector<uint8_t> rawBuf;
        nlohmann::json frameMeta;
        std::vector<float> rgbOut;
        std::vector<uint16_t> binnedBayer;
        uint32_t width = 0, height = 0;
        std::string outBytes;
        {
            py::gil_scoped_release release;
            d_->loadFrame(timestamp, rawBuf, frameMeta);
            auto params = mcc::BuildFrameParams(frameMeta, d_->getContainerMetadata());
            const uint16_t* raw = reinterpret_cast<const uint16_t*>(rawBuf.data());

            // Preview proxy: decimate the Bayer to 1/bin resolution BEFORE the
            // debayer/colour chain so a weak (CPU) machine can keep up in the
            // real-time player. bin == 1 keeps full resolution; both the CPU
            // and GPU paths below read params.width/height, so this transparently
            // shrinks the per-pixel work on whichever backend runs.
            if (preview_bin > 1) {
                uint32_t bw = 0, bh = 0;
                SubsampleBayerCells(raw, params.width, params.height, preview_bin,
                                    binnedBayer, bw, bh);
                if (!binnedBayer.empty()) {
                    raw = binnedBayer.data();
                    params.width = bw;
                    params.height = bh;
                }
            }
            width = params.width;
            height = params.height;
            outBytes.resize(size_t(width) * size_t(height) * 3);

            bool didGpu = false;
#if MCRAW_HAVE_CUDA
            // Real-time preview fast path: full bayer->output chain on the GPU,
            // 8-bit readback. Only for baked targets (OCIO falls to CPU below).
            if (prefer_gpu && motioncam::cuda::IsCudaAvailable()) {
                motioncam::cuda::BayerPipelineConstants C{};
                if (BuildBakedBayerConstants(params, cs, C)) {
                    C.highlight_recovery = highlight_recovery ? 1 : 0;
                    C.highlight_rolloff  =
                        (highlight_recovery && mcc::IsDisplayEncoded(cs)) ? 1 : 0;
                    if (!bake_vignette) { C.lsm_w = 0; C.lsm_h = 0; C.lsm_host = nullptr; }
                    didGpu = motioncam::cuda::ProcessBayerToRgb8(
                        raw, params.asShotNeutral, C,
                        reinterpret_cast<uint8_t*>(outBytes.data()),
                        want_scopes ? scopes.data() : nullptr);
                }
            }
#else
            (void)prefer_gpu;
#endif
            if (!didGpu) {
                mcc::ProcessFrame(raw, params, cs, rgbOut, highlight_recovery, bake_vignette);
                EnsureXform(cs);
                cachedXform_.Apply(rgbOut.data(), width, height);
                if (highlight_recovery && mcc::IsDisplayEncoded(cs)) {
                    mcc::HighlightRolloff(rgbOut.data(), width, height);
                }
                // Float [0,1] -> uint8 [0,255], packed RGB888. When scopes are
                // requested, accumulate them in the same pass — bin formula
                // matches the GPU ScopeHistogramKernel exactly.
                uint8_t* dst = reinterpret_cast<uint8_t*>(outBytes.data());
                const float* src = rgbOut.data();
                const size_t n = rgbOut.size();
                if (want_scopes) {
                    for (size_t i = 0; i < n; i += 3) {
                        bool lo = false, hi = false;
                        for (int c = 0; c < 3; ++c) {
                            float v = src[i + c];
                            if (v <= 0.0f) lo = true;
                            if (v >= 1.0f) hi = true;
                            float t = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                            const int b = int(t * 255.0f + 0.5f);
                            scopes[size_t(c) * 256 + size_t(b)] += 1u;
                            dst[i + c] = uint8_t(b);
                        }
                        if (lo) scopes[768] += 1u;
                        if (hi) scopes[769] += 1u;
                    }
                } else {
                    for (size_t i = 0; i < n; ++i) {
                        float v = src[i];
                        if (v < 0.0f) v = 0.0f;
                        else if (v > 1.0f) v = 1.0f;
                        dst[i] = uint8_t(v * 255.0f + 0.5f);
                    }
                }
            }
        }

        if (!want_scopes)
            return py::make_tuple(py::bytes(outBytes), int(height), int(width));

        const double pixels = double(width) * double(height);
        const double clipLo = pixels > 0 ? scopes[768] / pixels : 0.0;
        const double clipHi = pixels > 0 ? scopes[769] / pixels : 0.0;
        std::string histBytes(reinterpret_cast<const char*>(scopes.data()),
                              size_t(3 * 256) * sizeof(uint32_t));
        return py::make_tuple(py::bytes(outBytes), int(height), int(width),
                              py::bytes(histBytes), clipLo, clipHi);
    }

public:

    py::array_t<int16_t> load_audio() {
        std::vector<mc::AudioChunk> chunks;
        {
            py::gil_scoped_release release;
            d_->loadAudio(chunks);
        }
        const int channels = std::max(1, d_->numAudioChannels());
        size_t total = 0;
        for (auto& c : chunks) total += c.second.size();
        const int rows = int(total / channels);
        py::array_t<int16_t> arr({ rows, channels });
        int16_t* dst = arr.mutable_data();
        size_t off = 0;
        for (auto& c : chunks) {
            std::memcpy(dst + off, c.second.data(), c.second.size() * sizeof(int16_t));
            off += c.second.size();
        }
        return arr;
    }

    // numpy-free sibling of load_audio: interleaved int16 PCM as raw bytes, all
    // chunks concatenated in order (audio starts at frame 0). The bundled GUI
    // excludes numpy, so it feeds these bytes straight into a QAudioSink. Pair
    // with audio_sample_rate / audio_channels. Empty bytes if the clip is silent.
    py::bytes load_audio_bytes() {
        std::vector<mc::AudioChunk> chunks;
        {
            py::gil_scoped_release release;
            d_->loadAudio(chunks);
        }
        size_t total = 0;
        for (auto& c : chunks) total += c.second.size();
        std::string out;
        out.resize(total * sizeof(int16_t));
        size_t off = 0;
        for (auto& c : chunks) {
            const size_t nbytes = c.second.size() * sizeof(int16_t);
            std::memcpy(&out[off], c.second.data(), nbytes);
            off += nbytes;
        }
        return py::bytes(out);
    }

    // Internal access for free-function bindings that need to call the
    // raw mc::Decoder (e.g. the CUDA Phase C correctness test). Not
    // exposed to Python — it's just here because pybind11 only knows
    // about PyDecoder, so other bindings reach the real decoder through
    // this.
    mc::Decoder* underlying() { return d_.get(); }

private:
    void EnsureXform(mcc::OutputColorSpace cs) {
        // Re-Init only when the requested space changes — avoids reallocating
        // OCIO context (or rebuilding LUTs) on repeated identical calls.
        if (cachedXformCs_ != cs || !cachedXformInit_) {
            cachedXform_.Init(cs);
            cachedXformCs_ = cs;
            cachedXformInit_ = true;
        }
    }

    std::unique_ptr<mc::Decoder> d_;
    std::vector<int64_t> timestamps_;
    mcc::OutputTransform cachedXform_;
    mcc::OutputColorSpace cachedXformCs_ = mcc::OutputColorSpace::ACEScg;
    bool cachedXformInit_ = false;
};

class PyOcio {
public:
    PyOcio(const std::string& src, const std::string& dst)
        : t_(std::make_unique<mcc::OcioTransform>(src, dst)) {}

    py::array_t<float> apply(py::array_t<float, py::array::c_style | py::array::forcecast> rgb) {
        py::buffer_info buf = rgb.request();
        if (buf.ndim != 3 || buf.shape[2] != 3) {
            throw std::runtime_error("expected float32 array of shape (H, W, 3)");
        }
        const int height = int(buf.shape[0]);
        const int width  = int(buf.shape[1]);
        float* data = static_cast<float*>(buf.ptr);
        {
            py::gil_scoped_release release;
            t_->Apply(data, uint32_t(width), uint32_t(height));
        }
        return rgb;
    }

private:
    std::unique_ptr<mcc::OcioTransform> t_;
};

static void DoRender(
    const std::string& input,
    const std::string& output,
    const std::string& colorspace,
    py::object codec_obj,
    py::object start_obj,
    py::object end_obj,
    py::object fps_obj,
    int bitrate,
    py::object progress_obj,
    const std::string& exr_compression,
    int denoise_chroma,
    int denoise_luma,
    bool ten_bit,
    py::object cancel_obj,
    bool highlight_recovery,
    double frame_rate_conversion,
    bool bake_vignette)
{
    // Convert all py::object args to native C++ types while we still hold the GIL.
    int start_arg = start_obj.is_none() ? 0 : start_obj.cast<int>();
    int end_arg = end_obj.is_none() ? -1 : end_obj.cast<int>();
    double fps_arg = fps_obj.is_none() ? 0.0 : fps_obj.cast<double>();
    std::string codec_str = codec_obj.is_none() ? std::string() : codec_obj.cast<std::string>();
    const bool has_progress = !progress_obj.is_none();

    mcc::OutputColorSpace cs;
    if (!mcc::ParseOutputColorSpace(colorspace, cs)) {
        throw std::runtime_error("unknown colorspace: " + colorspace);
    }

    mcc::ExrCompression exr_comp = mcc::ExrCompression::ZIP;
    if (!exr_compression.empty() && !mcc::ParseExrCompression(exr_compression, exr_comp)) {
        throw std::runtime_error("unknown exr_compression: " + exr_compression);
    }

    // Cancel callback — invoked once per frame from the encode loop. The
    // GUI's pause/cancel button fronts this. We capture by value into a
    // C++ lambda so the inner threads (producer / parallel EXR workers) can
    // call it without holding GIL state across the render loop.
    const bool has_cancel = !cancel_obj.is_none();
    auto cancel_check = [&]() -> bool {
        if (!has_cancel) return false;
        py::gil_scoped_acquire gil;
        try {
            return py::cast<bool>(cancel_obj());
        } catch (...) {
            return false;
        }
    };

    mcv::Codec vcodec = mcv::Codec::ProRes4444;
    if (!codec_str.empty()) {
        if (!mcv::ParseCodec(codec_str, vcodec))
            throw std::runtime_error("unknown codec: " + codec_str);
    }

    // Release the GIL for the duration of the render — the rest of the function
    // is pure C++ and doesn't touch Python objects.
    py::gil_scoped_release release;

    mc::Decoder decoder(input);
    const auto& frames = decoder.getFrames();
    const auto& containerMeta = decoder.getContainerMetadata();
    const auto& csInfo = mcc::GetInfo(cs);

    mcc::OutputTransform xform;
    xform.Init(cs);

    const int totalFrames = int(frames.size());
    int s = start_arg;
    int e = (end_arg < 0) ? totalFrames : std::min(totalFrames, end_arg);
    s = std::max(0, std::min(e, s));

    if (EndsWithMov(output) || EndsWithMp4(output)) {
        std::vector<uint8_t> rawBuf;
        nlohmann::json frameMeta;
        decoder.loadFrame(frames[s], rawBuf, frameMeta);
        auto params0 = mcc::BuildFrameParams(frameMeta, containerMeta);

        // Frame-rate plan: detect the source rate (median of inter-frame
        // intervals) and, if a conversion target was requested, resample
        // (duplicate/drop frames) to a constant output rate while staying
        // time-aligned with the audio. With no target it's an identity 1:1
        // map at the detected rate. frame_rate_conversion takes priority over
        // the legacy fps override.
        const double frcTarget = (frame_rate_conversion > 0.0) ? frame_rate_conversion : fps_arg;
        std::vector<int64_t> ts(frames.begin(), frames.end());
        auto plan = mcv::BuildFramePlan(ts, s, e, frcTarget);

        mcv::EncodeSettings es{};
        es.outputPath = output;
        es.codec = vcodec;
        es.width = int(params0.width);
        es.height = int(params0.height);
        es.fpsNum = plan.outRate.num;
        es.fpsDen = plan.outRate.den;
        es.bitrateMbps = bitrate;
        es.audioSampleRate = decoder.audioSampleRateHz();
        es.audioChannels = decoder.numAudioChannels();
        es.containerFormat = EndsWithMp4(output) ? "mp4" : "mov";
        es.tenBit = ten_bit;
        es.colorPrimaries = csInfo.qtPrimaries;
        es.colorTrc       = csInfo.qtTransfer;
        es.colorMatrix    = csInfo.qtMatrix;

        mcv::MovEncoder enc(es);
        const int total_to_render = int(plan.srcIndex.size());

        // Denoise: MP4 only, and only when at least one strength is > 0.
        const bool wantDenoise = EndsWithMp4(output)
            && (denoise_chroma > 0 || denoise_luma > 0);

        // Tier 2.1 Phase C.2: GPU bayer pipeline opt-in. Enabled when the
        // encoder is on the Phase B CUDA hwframe path (MCRAW_GPU_YUV=1 +
        // NVENC + compatible codec), the user didn't request highlight
        // recovery / denoise (no GPU kernel yet), and the target colour
        // space has a BakedTransform. Falls back silently to the CPU
        // producer-consumer below otherwise.
        bool gpuBayerActive = false;
        {
            mcv::MovEncoder::GpuBayerSetup setup{};
            setup.targetColorSpace = static_cast<int>(cs);
            std::memcpy(setup.forwardMatrix2, params0.forwardMatrix2,
                        sizeof(setup.forwardMatrix2));
            for (int i = 0; i < 4; ++i)
                setup.blackPerPosition[i] = params0.blackPerPosition[i];
            setup.whiteLevel = params0.whiteLevel;
            setup.cfaPattern = static_cast<int>(params0.cfa);
            // OCIO targets (no BakedTransform) take the Phase D GPU 3D-LUT path.
            setup.ocioColorSpace = csInfo.ocioName;
            setup.highlightRecovery = highlight_recovery;
            setup.displayEncoded    = mcc::IsDisplayEncoded(cs);
            // Denoise is MP4-only; wantDenoise already encodes that.
            setup.denoiseChroma = wantDenoise ? denoise_chroma : 0;
            setup.denoiseLuma   = wantDenoise ? denoise_luma   : 0;
            gpuBayerActive = enc.EnableGpuBayerPipeline(setup);
        }

        if (gpuBayerActive) {
            // Sequential GPU bayer pipeline: decode -> upload bayer ->
            // normalise + LSM + debayer + matrix + RGB->NV12 -> NVENC.
            // CPU has almost nothing to do per frame so the producer-
            // consumer overlap below doesn't pay off. We iterate the frame
            // plan (identity unless converting); a 1-frame decode cache means
            // duplicated source frames aren't re-decoded.
            int written = 0;
            std::vector<uint8_t> rb = std::move(rawBuf);   // seed cache w/ probe frame
            nlohmann::json fm;
            mcc::FrameParams cp = params0;
            int cachedIdx = s;
            for (int srcIdx : plan.srcIndex) {
                if (cancel_check()) break;
                if (srcIdx != cachedIdx) {
                    decoder.loadFrame(frames[srcIdx], rb, fm);
                    cp = mcc::BuildFrameParams(fm, containerMeta);
                    cachedIdx = srcIdx;
                }
                const uint16_t* raw = reinterpret_cast<const uint16_t*>(rb.data());
                const bool useLsm = bake_vignette && !cp.lensShadingMap.empty();
                enc.WriteVideoFrameFromBayer(
                    raw, cp.asShotNeutral,
                    useLsm ? cp.lensShadingMap.data() : nullptr,
                    useLsm ? int(cp.lsmWidth)  : 0,
                    useLsm ? int(cp.lsmHeight) : 0);
                ++written;
                if (has_progress) {
                    py::gil_scoped_acquire gil;
                    try { progress_obj(written, total_to_render); } catch (...) {}
                }
            }
        } else {
        // Producer-consumer: producer thread runs decode + color pipeline;
        // main thread runs the encoder. Decode of frame N+1 overlaps with
        // encode of frame N — ~30-50% throughput win on multi-core.
        struct ProcessedFrame {
            std::vector<float> rgb;
            uint32_t width;
            uint32_t height;
        };
        constexpr size_t kQueueLimit = 2;

        std::deque<ProcessedFrame> queue;
        std::mutex mtx;
        std::condition_variable not_full, not_empty;
        std::atomic<bool> cancel{false};
        bool producer_done = false;
        std::exception_ptr producer_err;

        std::thread producer([&]() {
            try {
                std::vector<uint8_t> rb = rawBuf;        // seed cache w/ probe frame
                nlohmann::json fm;
                mcc::FrameParams cp = params0;
                int cachedIdx = s;
                for (int srcIdx : plan.srcIndex) {
                    if (cancel.load()) break;
                    if (cancel_check()) { cancel.store(true); not_empty.notify_all(); break; }
                    if (srcIdx != cachedIdx) {
                        decoder.loadFrame(frames[srcIdx], rb, fm);
                        cp = mcc::BuildFrameParams(fm, containerMeta);
                        cachedIdx = srcIdx;
                    }
                    ProcessedFrame buf;
                    buf.width = cp.width;
                    buf.height = cp.height;
                    const uint16_t* raw = reinterpret_cast<const uint16_t*>(rb.data());
                    mcc::ProcessFrame(raw, cp, cs, buf.rgb, highlight_recovery, bake_vignette);
                    xform.Apply(buf.rgb.data(), buf.width, buf.height);
                    if (highlight_recovery && mcc::IsDisplayEncoded(cs)) {
                        mcc::HighlightRolloff(buf.rgb.data(), buf.width, buf.height);
                    }
                    if (wantDenoise) {
                        mcc::DenoiseRgb(buf.rgb.data(), buf.width, buf.height,
                                        denoise_chroma, denoise_luma);
                    }

                    std::unique_lock<std::mutex> lk(mtx);
                    not_full.wait(lk, [&]{ return queue.size() < kQueueLimit || cancel.load(); });
                    if (cancel.load()) break;
                    queue.push_back(std::move(buf));
                    lk.unlock();
                    not_empty.notify_one();
                }
            } catch (...) {
                producer_err = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                producer_done = true;
            }
            not_empty.notify_all();
        });

        int written = 0;
        try {
            while (true) {
                std::unique_lock<std::mutex> lk(mtx);
                not_empty.wait(lk, [&]{ return !queue.empty() || producer_done; });
                if (queue.empty() && producer_done) break;
                ProcessedFrame buf = std::move(queue.front());
                queue.pop_front();
                lk.unlock();
                not_full.notify_one();

                enc.WriteVideoFrame(buf.rgb.data());
                ++written;
                if (has_progress) {
                    py::gil_scoped_acquire gil;
                    try {
                        progress_obj(written, total_to_render);
                    } catch (...) {}
                }
                // Cancel check on the consumer thread too — short-circuits
                // the rest of the queue if the user cancelled mid-batch.
                if (cancel_check()) {
                    cancel.store(true);
                    not_full.notify_all();
                    break;
                }
            }
        } catch (...) {
            cancel.store(true);
            not_full.notify_all();
            not_empty.notify_all();
            if (producer.joinable()) producer.join();
            throw;
        }
        producer.join();
        if (producer_err) std::rethrow_exception(producer_err);
        }  // end CPU producer-consumer branch

        if (es.audioSampleRate > 0 && es.audioChannels > 0) {
            std::vector<mc::AudioChunk> chunks;
            decoder.loadAudio(chunks);
            std::vector<int16_t> all;
            for (auto& c : chunks) all.insert(all.end(), c.second.begin(), c.second.end());
            // Trim audio to the rendered range using the source TIMESTAMPS
            // (audio assumed to start with frame 0). Timestamp-based so it
            // stays correct under frame-rate conversion, where output frame
            // count != source frame count.
            if (!all.empty() && (s > 0 || e < totalFrames) && e > s) {
                const double skipSec = double(frames[s]   - frames[0]) / 1.0e9;
                const double keepSec = double(frames[e-1] - frames[s]) / 1.0e9;
                const size_t skip = size_t(skipSec * es.audioSampleRate) * es.audioChannels;
                const size_t keep = size_t(keepSec * es.audioSampleRate) * es.audioChannels;
                if (skip >= all.size()) {
                    all.clear();
                } else {
                    all.erase(all.begin(), all.begin() + skip);
                    if (all.size() > keep) all.resize(keep);
                }
            }
            if (!all.empty()) enc.WriteAudio(all.data(), int(all.size()));
        }
        enc.Finalize();
    } else {
        std::filesystem::path outDir(output);
        std::filesystem::create_directories(outDir);
        const int total_to_render = e - s;

        // Parallel EXR: N worker threads, each opens its own Decoder + OcioTransform,
        // pulls frame indices from an atomic counter. EXR frames are independent files
        // so this scales linearly with cores and disk I/O.
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const unsigned nWorkers = std::min(hw, 8u);

        std::atomic<int> nextFrame{s};
        std::atomic<int> completedCount{0};
        std::atomic<bool> failed{false};
        std::mutex errMtx;
        std::exception_ptr workerErr;

        auto workerFn = [&]() {
            try {
                mc::Decoder dec(input);
                const auto& localFrames = dec.getFrames();
                const auto& cmeta = dec.getContainerMetadata();
                mcc::OutputTransform xform;
                xform.Init(cs);
                std::vector<uint8_t> rb;
                nlohmann::json fm;
                std::vector<float> rgb;

                while (!failed.load()) {
                    if (cancel_check()) { failed.store(true); break; }

                    const int i = nextFrame.fetch_add(1);
                    if (i >= e) break;
                    dec.loadFrame(localFrames[i], rb, fm);
                    auto p = mcc::BuildFrameParams(fm, cmeta);
                    const uint16_t* raw = reinterpret_cast<const uint16_t*>(rb.data());
                    mcc::ProcessFrame(raw, p, cs, rgb, highlight_recovery);
                    xform.Apply(rgb.data(), p.width, p.height);
                    if (highlight_recovery && mcc::IsDisplayEncoded(cs)) {
                        mcc::HighlightRolloff(rgb.data(), p.width, p.height);
                    }

                    char name[64];
                    std::snprintf(name, sizeof(name), "frame_%06d.exr", i);
                    auto outPath = outDir / name;
                    mcc::WriteExrHalfRgb(
                        outPath.string(), rgb.data(), p.width, p.height,
                        csInfo.ocioName, csInfo.chromaticities,
                        exr_comp);

                    const int done = completedCount.fetch_add(1) + 1;
                    if (has_progress) {
                        py::gil_scoped_acquire gil;
                        try { progress_obj(done, total_to_render); } catch (...) {}
                    }
                }
            } catch (...) {
                std::lock_guard<std::mutex> lk(errMtx);
                if (!workerErr) workerErr = std::current_exception();
                failed.store(true);
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(nWorkers);
        for (unsigned t = 0; t < nWorkers; ++t) workers.emplace_back(workerFn);
        for (auto& w : workers) w.join();

        if (workerErr) std::rethrow_exception(workerErr);
    }
}

PYBIND11_MODULE(mcraw, m) {
    m.doc() = "MotionCam MCRAW decoder + ACES color pipeline + OCIO + ProRes/H.26x transcoder";

    py::class_<PyDecoder>(m, "Decoder")
        .def(py::init<const std::string&>(), py::arg("path"))
        .def_property_readonly("frame_count", &PyDecoder::frame_count)
        .def_property_readonly("frames", &PyDecoder::frames)
        .def_property_readonly("audio_sample_rate", &PyDecoder::audio_sample_rate)
        .def_property_readonly("audio_channels", &PyDecoder::audio_channels)
        .def_property_readonly("container_metadata", &PyDecoder::container_metadata)
        .def("load_bayer", &PyDecoder::load_bayer, py::arg("timestamp"),
             "Returns (uint16 numpy (H,W) bayer, frame metadata dict).")
        .def("load_audio", &PyDecoder::load_audio,
             "Returns int16 numpy (samples, channels), interleaved.")
        .def("load_audio_bytes", &PyDecoder::load_audio_bytes,
             "Interleaved int16 PCM as raw bytes (numpy-free), all chunks "
             "concatenated. Pair with audio_sample_rate / audio_channels.")
        .def("process_frame", &PyDecoder::process_frame,
             py::arg("timestamp"), py::arg("colorspace") = "acescg",
             py::arg("highlight_recovery") = false, py::arg("bake_vignette") = true,
             "Returns float32 numpy (H, W, 3) RGB in the requested color space.")
        .def("process_frame_rgb24", &PyDecoder::process_frame_rgb24,
             py::arg("timestamp"), py::arg("colorspace") = "srgb",
             py::arg("highlight_recovery") = false, py::arg("bake_vignette") = true,
             py::arg("prefer_gpu") = false, py::arg("preview_bin") = 1,
             "Returns (RGB888 packed bytes, height, width). No numpy required. "
             "prefer_gpu uses the CUDA bayer pipeline for baked targets (real-time preview). "
             "preview_bin>1 decimates the Bayer to 1/bin resolution first (preview proxy, "
             "~1/bin^2 the per-pixel cost) for low-end machines.")
        .def("process_frame_rgb24_scopes", &PyDecoder::process_frame_rgb24_scopes,
             py::arg("timestamp"), py::arg("colorspace") = "srgb",
             py::arg("highlight_recovery") = false, py::arg("bake_vignette") = true,
             py::arg("prefer_gpu") = false, py::arg("preview_bin") = 1,
             "Like process_frame_rgb24 but also returns scope data: "
             "(rgb_bytes, height, width, hist_bytes, clip_lo, clip_hi). "
             "hist_bytes is 3*256 little-endian uint32 (R,G,B histogram of the "
             "pre-quantize float output); clip_lo/clip_hi are the fractions of "
             "pixels with any channel <= 0.0 / >= 1.0 before the 8-bit clamp.");

    py::class_<PyOcio>(m, "OcioTransform")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("src"), py::arg("dst"),
             "Build a CPU OCIO transform between two color spaces by name (matching the\n"
             "loaded studio-config-v4.0.0_aces-v2.0_ocio-v2.5).")
        .def("apply", &PyOcio::apply, py::arg("rgb"),
             "Apply the transform in-place to a float32 numpy (H, W, 3) array.");

    m.def("render", &DoRender,
        py::arg("input"),
        py::arg("output"),
        py::arg("colorspace") = "acescg",
        py::arg("codec") = py::none(),
        py::arg("start") = py::none(),
        py::arg("end") = py::none(),
        py::arg("fps") = py::none(),
        py::arg("bitrate") = 80,
        py::arg("progress") = py::none(),
        py::arg("exr_compression") = "zip",
        py::arg("denoise_chroma") = 0,
        py::arg("denoise_luma") = 0,
        py::arg("ten_bit") = false,
        py::arg("cancel") = py::none(),
        py::arg("highlight_recovery") = false,
        py::arg("frame_rate_conversion") = 0.0,
        py::arg("bake_vignette") = true,
        R"doc(Render an MCRAW file end-to-end.

If `output` ends in '.mov', encodes a QuickTime file (codec defaults to prores4444).
Otherwise treats `output` as a directory and writes an EXR sequence.

colorspace: acescg (default), rec709, aces2065-1, acescct,
            slog3-sgamut3cine, srgb, rec709-display
codec (mov only): prores422, prores422hq, prores4444, prores4444xq, h264, h265
frame_rate_conversion: target fps to convert to (duplicate/drop to keep A/V
            sync); 0 = keep the source rate. bake_vignette: apply the lens
            shading gainmap (default True); False keeps the natural vignette.
)doc");

    // Frame-rate plan preview for the GUI clip header: returns the detected
    // source rate, the resulting output rate, output frame count, and the
    // duplicated/dropped counts — WITHOUT rendering. start/end default to the
    // whole clip; frame_rate_conversion=0 means no conversion (identity).
    m.def("frame_plan_info",
        [](PyDecoder& dec, py::object start_obj, py::object end_obj,
           double frame_rate_conversion) -> py::dict {
            const auto& fr = dec.frames();
            const int total = int(fr.size());
            const int s = start_obj.is_none() ? 0 : std::max(0, start_obj.cast<int>());
            const int e = end_obj.is_none() ? total
                                            : std::min(total, end_obj.cast<int>());
            std::vector<int64_t> ts(fr.begin(), fr.end());
            auto plan = mcv::BuildFramePlan(ts, s, std::max(s, e), frame_rate_conversion);
            py::dict d;
            d["src_fps"]    = plan.srcRate.fps();
            d["out_fps"]    = plan.outRate.fps();
            d["out_num"]    = plan.outRate.num;
            d["out_den"]    = plan.outRate.den;
            d["out_frames"] = int(plan.srcIndex.size());
            d["duplicated"] = plan.duplicated;
            d["dropped"]    = plan.dropped;
            return d;
        },
        py::arg("decoder"), py::arg("start") = py::none(),
        py::arg("end") = py::none(), py::arg("frame_rate_conversion") = 0.0,
        "Frame-rate plan summary (src/out fps, frame count, dropped/duplicated) "
        "without rendering. For the GUI clip header.");

    m.def("color_spaces", []() {
        py::list out;
        const mcc::OutputColorSpace all[] = {
            mcc::OutputColorSpace::ACEScg,
            mcc::OutputColorSpace::LinearRec709,
            mcc::OutputColorSpace::ACES2065_1,
            mcc::OutputColorSpace::ACEScct,
            mcc::OutputColorSpace::SLog3SGamut3Cine,
            mcc::OutputColorSpace::SRGB,
            mcc::OutputColorSpace::Rec709Display,
        };
        for (auto cs : all) {
            const auto& info = mcc::GetInfo(cs);
            py::dict d;
            d["short_name"] = info.shortName;
            d["ocio_name"] = info.ocioName;
            d["requires_ocio"] = info.requiresOcio;
            out.append(d);
        }
        return out;
    }, "List supported output color spaces.");

    m.def("codecs", []() {
        return std::vector<std::string>{
            "prores422", "prores422hq", "prores4444", "prores4444xq",
            "h264", "h265",
            "h264_nvenc", "h265_nvenc", "av1_nvenc",
            "dnxhr_hqx", "dnxhr_444", "cineform",
        };
    }, "List supported video codecs (h264_nvenc / h265_nvenc / av1_nvenc require an NVIDIA GPU).");

    // CPU reference denoise (the same DenoiseRgb the CPU render path uses).
    // In-place on a float32 (H, W, 3) array. Exposed for the Phase F GPU-vs-CPU
    // correctness test.
    m.def("denoise_rgb",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> rgb,
           int chroma, int luma) -> py::array_t<float> {
            py::buffer_info buf = rgb.request();
            if (buf.ndim != 3 || buf.shape[2] != 3)
                throw std::runtime_error("expected float32 array of shape (H, W, 3)");
            const int height = int(buf.shape[0]);
            const int width  = int(buf.shape[1]);
            float* data = static_cast<float*>(buf.ptr);
            {
                py::gil_scoped_release release;
                mcc::DenoiseRgb(data, uint32_t(width), uint32_t(height), chroma, luma);
            }
            return rgb;
        },
        py::arg("rgb"), py::arg("chroma"), py::arg("luma"),
        "Apply the CPU output-space denoise in-place to a float32 (H,W,3) array.");

    // ---- CUDA probes (Tier 2.1 GPU pipeline) ----
    m.def("cuda_built", []() -> bool {
#if MCRAW_HAVE_CUDA
        return true;
#else
        return false;
#endif
    }, "Whether this build was compiled with CUDA support. False = CPU-only.");

    m.def("cuda_available", []() -> bool {
#if MCRAW_HAVE_CUDA
        return motioncam::cuda::IsCudaAvailable();
#else
        return false;
#endif
    }, "Whether a CUDA-capable GPU is visible at runtime (requires NVIDIA driver).");

    m.def("cuda_phase_a_probe", []() -> bool {
#if MCRAW_HAVE_CUDA
        return motioncam::cuda::RunPhaseAProbe();
#else
        return false;
#endif
    }, "Tier 2.1 Phase A health check: launches a no-op CUDA kernel and "
       "synchronises. Returns true if the kernel completed without error.");

#if MCRAW_HAVE_CUDA
    // Phase C.1 correctness test. Runs the GPU bayer pipeline (normalise +
    // debayer + matrix + optional curve) on a single decoded frame and
    // returns the result as a (H, W, 3) float32 numpy array — same shape
    // as Decoder.process_frame, so the test script can subtract the two
    // and verify they're equal within float epsilon.
    m.def("cuda_process_frame_phase_c",
        [](PyDecoder& pyDec, int64_t timestamp,
           const std::string& target_colorspace,
           bool highlight_recovery,
           int denoise_chroma, int denoise_luma) -> py::array_t<float> {
            mc::Decoder& dec = *pyDec.underlying();

            // Map the target colorspace to a (matrix, curve) plan.
            // Matches BakedTransform.cpp's LookupPlan(). Throws if the
            // target isn't bake-compatible (caller should fall back to CPU).
            mcc::OutputColorSpace cs;
            if (!mcc::ParseOutputColorSpace(target_colorspace, cs)) {
                throw std::runtime_error("unknown colorspace: " + target_colorspace);
            }
            if (!mcc::HasBakedTransform(cs)) {
                throw std::runtime_error(
                    "GPU pipeline doesn't support " + target_colorspace +
                    " yet (OCIO-only output). Use a baked-transform target "
                    "(acescg, rec709, aces2065-1, srgb, rec709-2.2, rec709-display).");
            }

            // Pull the bayer + per-frame + per-clip metadata.
            std::vector<uint8_t> rawBuf;
            nlohmann::json frameMeta;
            uint32_t width = 0, height = 0;
            mcc::FrameParams params;
            {
                py::gil_scoped_release release;
                dec.loadFrame(timestamp, rawBuf, frameMeta);
                params = mcc::BuildFrameParams(frameMeta, dec.getContainerMetadata());
                width = params.width;
                height = params.height;
            }

            // Build the combined cam->output matrix on CPU. Same chain the
            // CPU pipeline uses: ForwardMatrix2 -> Bradford D50/D60 ->
            // AP1 (ACEScg) -> BakedTransform matrix (AP1 -> output).
            //
            // We don't have a public helper exposing the cam->ACEScg /
            // cam->Rec709 matrix builders from ColorPipeline.cpp, so this
            // duplicates them. Kept short - they're 3x3 matrix muls.
            // TODO(phase-c.2): refactor BuildCamera*() helpers into the
            //                  public header and call them here.

            // From ColorPipeline.cpp (Bradford D50->D60).
            constexpr float Bradford_D50_to_D60[9] = {
                 0.96766f, -0.01686f,  0.04424f,
                -0.02099f,  1.00778f,  0.01477f,
                 0.00853f, -0.01415f,  1.22963f,
            };
            // XYZ D60 -> AP1 (ACEScg) - same constants as ColorPipeline.cpp.
            constexpr float XYZ_D60_to_AP1[9] = {
                 1.6410233797f, -0.3248032942f, -0.2364246952f,
                -0.6636628587f,  1.6153315917f,  0.0167563477f,
                 0.0117218943f, -0.0082844420f,  0.9883948585f,
            };

            auto matMul = [](const float A[9], const float B[9], float out[9]) {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        out[i*3+j] = A[i*3+0]*B[j] + A[i*3+1]*B[3+j] + A[i*3+2]*B[6+j];
            };

            // cam -> XYZ_D50 -> XYZ_D60 -> AP1 (ACEScg)
            float tmp[9];
            float camToAcescg[9];
            matMul(Bradford_D50_to_D60, params.forwardMatrix2, tmp);
            matMul(XYZ_D60_to_AP1, tmp, camToAcescg);

            // ACEScg -> output (BakedTransform matrix).
            const float* bakedM = nullptr;
            int curveCode = 0;  // matches BakedTransform.cpp Curve enum
            switch (cs) {
                case mcc::OutputColorSpace::ACEScg:
                    bakedM = nullptr; curveCode = 0; break;
                case mcc::OutputColorSpace::LinearRec709: {
                    static const float M[9] = {
                         1.7050514f, -0.6217908f, -0.0832606f,
                        -0.1302561f,  1.1408047f, -0.0105486f,
                        -0.0240083f, -0.1289693f,  1.1529777f,
                    };
                    bakedM = M; curveCode = 0; break;
                }
                case mcc::OutputColorSpace::ACES2065_1: {
                    static const float M[9] = {
                         0.6954522f,  0.1406787f,  0.1638691f,
                         0.0447946f,  0.8596711f,  0.0955343f,
                        -0.0055258f,  0.0040252f,  1.0015007f,
                    };
                    bakedM = M; curveCode = 0; break;
                }
                case mcc::OutputColorSpace::Rec709Gamma22: {
                    static const float M[9] = {
                         1.7050514f, -0.6217908f, -0.0832606f,
                        -0.1302561f,  1.1408047f, -0.0105486f,
                        -0.0240083f, -0.1289693f,  1.1529777f,
                    };
                    bakedM = M; curveCode = 1; break;
                }
                case mcc::OutputColorSpace::Rec709Display: {
                    static const float M[9] = {
                         1.7050514f, -0.6217908f, -0.0832606f,
                        -0.1302561f,  1.1408047f, -0.0105486f,
                        -0.0240083f, -0.1289693f,  1.1529777f,
                    };
                    bakedM = M; curveCode = 2; break;
                }
                case mcc::OutputColorSpace::SRGB: {
                    static const float M[9] = {
                         1.7050514f, -0.6217908f, -0.0832606f,
                        -0.1302561f,  1.1408047f, -0.0105486f,
                        -0.0240083f, -0.1289693f,  1.1529777f,
                    };
                    bakedM = M; curveCode = 3; break;
                }
                default:
                    throw std::runtime_error("unreachable (non-baked target)");
            }

            float fullMatrix[9];
            if (bakedM) {
                matMul(bakedM, camToAcescg, fullMatrix);
            } else {
                std::memcpy(fullMatrix, camToAcescg, sizeof(fullMatrix));
            }

            // Build per-CFA-position constants. The CPU NormalizeBayer
            // splits the channel mapping via CfaChannelMap; we replicate
            // that here so the kernel gets {R,G,G,B} (or BGGR etc.).
            int chMap[4];
            int cfaToLsm[4];
            switch (params.cfa) {
                case mcc::CfaPattern::RGGB:
                    chMap[0]=0; chMap[1]=1; chMap[2]=1; chMap[3]=2;
                    cfaToLsm[0]=0; cfaToLsm[1]=1; cfaToLsm[2]=2; cfaToLsm[3]=3;
                    break;
                case mcc::CfaPattern::BGGR:
                    chMap[0]=2; chMap[1]=1; chMap[2]=1; chMap[3]=0;
                    cfaToLsm[0]=3; cfaToLsm[1]=2; cfaToLsm[2]=1; cfaToLsm[3]=0;
                    break;
                case mcc::CfaPattern::GRBG:
                    chMap[0]=1; chMap[1]=0; chMap[2]=2; chMap[3]=1;
                    cfaToLsm[0]=1; cfaToLsm[1]=0; cfaToLsm[2]=3; cfaToLsm[3]=2;
                    break;
                case mcc::CfaPattern::GBRG:
                    chMap[0]=1; chMap[1]=2; chMap[2]=0; chMap[3]=1;
                    cfaToLsm[0]=2; cfaToLsm[1]=3; cfaToLsm[2]=0; cfaToLsm[3]=1;
                    break;
            }

            motioncam::cuda::BayerPipelineConstants C{};
            std::memcpy(C.cam_to_output, fullMatrix, sizeof(fullMatrix));
            for (int i = 0; i < 4; ++i) {
                C.black[i] = params.blackPerPosition[i];
                double denom = params.whiteLevel - double(params.blackPerPosition[i]);
                C.inv_range[i] = denom > 0.0 ? float(1.0 / denom) : 0.0f;
                C.cfa_channel[i] = chMap[i];
                C.cfa_to_lsm[i]  = cfaToLsm[i];
            }
            C.curve  = curveCode;
            C.width  = int(width);
            C.height = int(height);
            C.highlight_recovery = highlight_recovery ? 1 : 0;
            C.highlight_rolloff  = (highlight_recovery && mcc::IsDisplayEncoded(cs)) ? 1 : 0;
            C.denoise_chroma = denoise_chroma;
            C.denoise_luma   = denoise_luma;
            // Pass-through optional LSM. params.lensShadingMap is already
            // flattened as 4 * lsmW * lsmH channel-first floats by
            // BuildFrameParams — exactly what the kernel expects.
            if (!params.lensShadingMap.empty() &&
                params.lsmWidth >= 2 && params.lsmHeight >= 2) {
                C.lsm_w    = int(params.lsmWidth);
                C.lsm_h    = int(params.lsmHeight);
                C.lsm_host = params.lensShadingMap.data();
            } else {
                C.lsm_w    = 0;
                C.lsm_h    = 0;
                C.lsm_host = nullptr;
            }

            // Allocate output array, run pipeline.
            py::array_t<float> arr({ int(height), int(width), 3 });
            {
                py::gil_scoped_release release;
                const uint16_t* raw = reinterpret_cast<const uint16_t*>(rawBuf.data());
                bool ok = motioncam::cuda::ProcessBayerToRgb(
                    raw, params.asShotNeutral, C, arr.mutable_data());
                if (!ok) {
                    throw std::runtime_error("ProcessBayerToRgb kernel failed");
                }
            }
            return arr;
        },
        py::arg("decoder"), py::arg("timestamp"), py::arg("target_colorspace"),
        py::arg("highlight_recovery") = false,
        py::arg("denoise_chroma") = 0, py::arg("denoise_luma") = 0,
        "Phase C.1 GPU bayer pipeline (normalise + debayer + cam-to-output "
        "matrix + optional curve, highlight recovery/rolloff, denoise). "
        "Returns float32 (H, W, 3) RGB in the target colour space. For "
        "correctness testing against process_frame.");

    // Phase D correctness test. Runs the GPU bayer pipeline with the OCIO
    // 3D-LUT step (normalise + debayer + cam->ACEScg matrix + asinh-shaped
    // 3D LUT) on a single frame and returns (H, W, 3) float32 in the target
    // OCIO space — same shape as process_frame, so the test script can
    // subtract the two and confirm the GPU LUT matches the OCIO CPU path.
    m.def("cuda_process_frame_phase_d",
        [](PyDecoder& pyDec, int64_t timestamp,
           const std::string& target_colorspace) -> py::array_t<float> {
            mc::Decoder& dec = *pyDec.underlying();

            mcc::OutputColorSpace cs;
            if (!mcc::ParseOutputColorSpace(target_colorspace, cs)) {
                throw std::runtime_error("unknown colorspace: " + target_colorspace);
            }
            const auto& info = mcc::GetInfo(cs);
            if (!info.requiresOcio) {
                throw std::runtime_error(
                    target_colorspace + " is not an OCIO target (use "
                    "cuda_process_frame_phase_c for baked targets).");
            }

            std::vector<uint8_t> rawBuf;
            nlohmann::json frameMeta;
            uint32_t width = 0, height = 0;
            mcc::FrameParams params;
            {
                py::gil_scoped_release release;
                dec.loadFrame(timestamp, rawBuf, frameMeta);
                params = mcc::BuildFrameParams(frameMeta, dec.getContainerMetadata());
                width = params.width;
                height = params.height;
            }

            // cam -> ACEScg (AP1). Same chain as the baked binding above, but
            // we stop at ACEScg and let the 3D LUT carry ACEScg -> target.
            constexpr float Bradford_D50_to_D60[9] = {
                 0.96766f, -0.01686f,  0.04424f,
                -0.02099f,  1.00778f,  0.01477f,
                 0.00853f, -0.01415f,  1.22963f,
            };
            constexpr float XYZ_D60_to_AP1[9] = {
                 1.6410233797f, -0.3248032942f, -0.2364246952f,
                -0.6636628587f,  1.6153315917f,  0.0167563477f,
                 0.0117218943f, -0.0082844420f,  0.9883948585f,
            };
            auto matMul = [](const float A[9], const float B[9], float out[9]) {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        out[i*3+j] = A[i*3+0]*B[j] + A[i*3+1]*B[3+j] + A[i*3+2]*B[6+j];
            };
            float tmp[9], camToAcescg[9];
            matMul(Bradford_D50_to_D60, params.forwardMatrix2, tmp);
            matMul(XYZ_D60_to_AP1, tmp, camToAcescg);

            // CFA channel + LSM maps (same as the baked binding).
            int chMap[4], cfaToLsm[4];
            switch (params.cfa) {
                case mcc::CfaPattern::RGGB:
                    chMap[0]=0; chMap[1]=1; chMap[2]=1; chMap[3]=2;
                    cfaToLsm[0]=0; cfaToLsm[1]=1; cfaToLsm[2]=2; cfaToLsm[3]=3; break;
                case mcc::CfaPattern::BGGR:
                    chMap[0]=2; chMap[1]=1; chMap[2]=1; chMap[3]=0;
                    cfaToLsm[0]=3; cfaToLsm[1]=2; cfaToLsm[2]=1; cfaToLsm[3]=0; break;
                case mcc::CfaPattern::GRBG:
                    chMap[0]=1; chMap[1]=0; chMap[2]=2; chMap[3]=1;
                    cfaToLsm[0]=1; cfaToLsm[1]=0; cfaToLsm[2]=3; cfaToLsm[3]=2; break;
                case mcc::CfaPattern::GBRG:
                    chMap[0]=1; chMap[1]=2; chMap[2]=0; chMap[3]=1;
                    cfaToLsm[0]=2; cfaToLsm[1]=3; cfaToLsm[2]=0; cfaToLsm[3]=1; break;
            }

            // Bake the OCIO ACEScg->target cube and upload it.
            {
                py::gil_scoped_release release;
                std::vector<float> lut = mcc::BakeAcesCgToTargetLut3D(
                    info.ocioName, motioncam::cuda::kLutSize,
                    motioncam::cuda::kLutShaperK,
                    motioncam::cuda::kLutShaperLo,
                    motioncam::cuda::kLutShaperHi);
                if (!motioncam::cuda::SetupLut3D(lut.data(), motioncam::cuda::kLutSize)) {
                    throw std::runtime_error("SetupLut3D failed");
                }
            }

            motioncam::cuda::BayerPipelineConstants C{};
            std::memcpy(C.cam_to_output, camToAcescg, sizeof(camToAcescg));
            for (int i = 0; i < 4; ++i) {
                C.black[i] = params.blackPerPosition[i];
                double denom = params.whiteLevel - double(params.blackPerPosition[i]);
                C.inv_range[i] = denom > 0.0 ? float(1.0 / denom) : 0.0f;
                C.cfa_channel[i] = chMap[i];
                C.cfa_to_lsm[i]  = cfaToLsm[i];
            }
            C.curve     = 0;          // ACEScg is linear; LUT does the rest
            C.width     = int(width);
            C.height    = int(height);
            C.use_lut3d = 1;
            if (!params.lensShadingMap.empty() &&
                params.lsmWidth >= 2 && params.lsmHeight >= 2) {
                C.lsm_w    = int(params.lsmWidth);
                C.lsm_h    = int(params.lsmHeight);
                C.lsm_host = params.lensShadingMap.data();
            } else {
                C.lsm_w = 0; C.lsm_h = 0; C.lsm_host = nullptr;
            }

            py::array_t<float> arr({ int(height), int(width), 3 });
            {
                py::gil_scoped_release release;
                const uint16_t* raw = reinterpret_cast<const uint16_t*>(rawBuf.data());
                bool ok = motioncam::cuda::ProcessBayerToRgb(
                    raw, params.asShotNeutral, C, arr.mutable_data());
                motioncam::cuda::ReleaseLut3D();
                if (!ok) throw std::runtime_error("ProcessBayerToRgb (LUT) kernel failed");
            }
            return arr;
        },
        py::arg("decoder"), py::arg("timestamp"), py::arg("target_colorspace"),
        "Phase D GPU bayer pipeline with OCIO 3D LUT. Returns float32 "
        "(H, W, 3) RGB in the target OCIO space. For correctness testing "
        "against process_frame.");
#endif

    m.def("encoder_available", [](const std::string& name) -> bool {
        // Two-step check: codec compiled into FFmpeg, AND it can actually open
        // (NVENC will pass step 1 on every bundle but fail step 2 on machines
        // without an NVIDIA driver / GPU). We use a tiny dummy frame size so the
        // probe is cheap and identical for all encoders.
        const AVCodec* codec = avcodec_find_encoder_by_name(name.c_str());
        if (!codec) return false;

        AVCodecContext* ctx = avcodec_alloc_context3(codec);
        if (!ctx) return false;

        ctx->width = 320;
        ctx->height = 240;
        ctx->time_base = {1, 30};
        ctx->framerate = {30, 1};
        ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        ctx->bit_rate = 1000000;

        int ret = avcodec_open2(ctx, codec, nullptr);
        avcodec_free_context(&ctx);
        return ret >= 0;
    }, py::arg("name"),
       "Probe whether an FFmpeg encoder is compiled in AND can be opened on this\n"
       "machine (e.g. NVENC fails on machines without an NVIDIA GPU/driver). Used\n"
       "by the GUI to gate GPU codec entries to machines that can actually run them.");

    m.def("trim_mcraw",
        [](const std::string& input,
           const std::string& output,
           int start, int end,
           py::object progress_obj,
           py::object cancel_obj)
        {
            const bool has_progress = !progress_obj.is_none();
            const bool has_cancel   = !cancel_obj.is_none();
            std::function<void(int, int)> prog_cb;
            std::function<bool()> cancel_cb;
            if (has_progress) {
                prog_cb = [&progress_obj](int cur, int tot) {
                    py::gil_scoped_acquire gil;
                    try { progress_obj(cur, tot); } catch (...) {}
                };
            }
            if (has_cancel) {
                cancel_cb = [&cancel_obj]() -> bool {
                    py::gil_scoped_acquire gil;
                    try { return py::cast<bool>(cancel_obj()); }
                    catch (...) { return false; }
                };
            }
            py::gil_scoped_release release;
            mc::TrimMcraw(input, output, start, end, prog_cb, cancel_cb);
        },
        py::arg("input"),
        py::arg("output"),
        py::arg("start"),
        py::arg("end"),
        py::arg("progress") = py::none(),
        py::arg("cancel") = py::none(),
        "Trim an MCRAW file to frames [start, end). Output is a fully-valid MCRAW\n"
        "with bit-perfect copies of the compressed bayer + frame metadata. Audio\n"
        "chunks overlapping the trim range are also copied. Frame timestamps are\n"
        "preserved (not rebased). The optional `progress(current, total)` callback\n"
        "fires every ~10 frames during the copy.");

    m.def("exr_compressions", []() {
        // Order matters: presented to the GUI in this order. Lossless first,
        // lossy after, "none" last as the rare-special-case option.
        return std::vector<std::string>{
            "piz", "zip", "zips", "rle",       // lossless
            "dwab", "dwaa", "b44a", "b44", "pxr24",  // lossy
            "none",
        };
    }, "List supported EXR compression methods (lossless: piz/zip/zips/rle/none, lossy: dwab/dwaa/b44a/b44/pxr24).");
}
