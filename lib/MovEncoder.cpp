#include <motioncam/MovEncoder.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

#if MCRAW_HAVE_CUDA
#include <motioncam/CudaHwHandoff.hpp>
#include <motioncam/OcioTransform.hpp>   // Phase D: bake OCIO ACEScg->target LUT
#include <vector>
#endif

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace motioncam {
namespace video {

namespace {

const AVCodec* FindEncoder(Codec c) {
    switch (c) {
        case Codec::ProRes422:
        case Codec::ProRes422HQ:
        case Codec::ProRes4444:
        case Codec::ProRes4444XQ:
            return avcodec_find_encoder_by_name("prores_ks");
        case Codec::H264:
            return avcodec_find_encoder_by_name("libx264");
        case Codec::H265:
            return avcodec_find_encoder_by_name("libx265");
        case Codec::H264NVENC: {
            // Prefer NVENC; transparently fall back to libx264 if FFmpeg wasn't
            // built with NVENC or no NVIDIA GPU is present.
            if (auto* e = avcodec_find_encoder_by_name("h264_nvenc")) return e;
            return avcodec_find_encoder_by_name("libx264");
        }
        case Codec::H265NVENC: {
            if (auto* e = avcodec_find_encoder_by_name("hevc_nvenc")) return e;
            return avcodec_find_encoder_by_name("libx265");
        }
        case Codec::AV1NVENC: {
            // No CPU AV1 fallback — libsvtav1/libaom-av1 aren't in our build,
            // and even if they were they're so slow they're not a sane substitute.
            // GUI gates AV1 NVENC to machines that have it; if it slips through,
            // we'll hit an avcodec_open2 error and surface it cleanly.
            return avcodec_find_encoder_by_name("av1_nvenc");
        }
        case Codec::DNxHR_HQX:
        case Codec::DNxHR_444:
            return avcodec_find_encoder_by_name("dnxhd");
        case Codec::CineForm:
            return avcodec_find_encoder_by_name("cfhd");
    }
    return nullptr;
}

int ProResProfile(Codec c) {
    switch (c) {
        case Codec::ProRes422:    return 2;
        case Codec::ProRes422HQ:  return 3;
        case Codec::ProRes4444:   return 4;
        case Codec::ProRes4444XQ: return 5;
        default: return -1;
    }
}

AVPixelFormat PreferredPixFmt(Codec c) {
    switch (c) {
        case Codec::ProRes422:
        case Codec::ProRes422HQ:    return AV_PIX_FMT_YUV422P10LE;
        case Codec::ProRes4444:
        case Codec::ProRes4444XQ:   return AV_PIX_FMT_YUV444P10LE;
        case Codec::H264:           return AV_PIX_FMT_YUV420P;
        case Codec::H265:           return AV_PIX_FMT_YUV420P;
        case Codec::H264NVENC:      return AV_PIX_FMT_YUV420P;
        case Codec::H265NVENC:      return AV_PIX_FMT_YUV420P;
        case Codec::AV1NVENC:       return AV_PIX_FMT_YUV420P;       // 8-bit AV1 main; tenBit flag overrides to YUV420P10LE
        case Codec::DNxHR_HQX:      return AV_PIX_FMT_YUV422P10LE;   // DNxHR HQX is 10-bit native
        case Codec::DNxHR_444:      return AV_PIX_FMT_YUV444P10LE;   // 10-bit YUV444 (12-bit needs explicit bit_rate config)
        case Codec::CineForm:       return AV_PIX_FMT_YUV422P10LE;
    }
    return AV_PIX_FMT_YUV420P;
}

[[noreturn]] void ThrowAv(const char* msg, int err) {
    char buf[256] = {0};
    av_strerror(err, buf, sizeof(buf));
    std::string s = "MovEncoder: ";
    s += msg;
    s += ": ";
    s += buf;
    throw std::runtime_error(s);
}

[[noreturn]] void Throw(const char* msg) {
    std::string s = "MovEncoder: ";
    s += msg;
    throw std::runtime_error(s);
}

}

struct MovEncoder::Impl {
    EncodeSettings settings;

    AVFormatContext* fmt = nullptr;

    AVStream* videoStream = nullptr;
    AVCodecContext* videoCtx = nullptr;
    SwsContext* sws = nullptr;
    AVFrame* yuvFrame = nullptr;
    AVFrame* rgbStaging = nullptr;
    int64_t videoPts = 0;

    AVStream* audioStream = nullptr;
    AVCodecContext* audioCtx = nullptr;
    int64_t audioPts = 0;

    AVPacket* pkt = nullptr;
    bool finalized = false;

#if MCRAW_HAVE_CUDA
    // CUDA upload path (Tier 2.1 Phase A.2 / B). Opt-in via MCRAW_GPU_YUV=1
    // env var. WriteVideoFrame replaces sws_scale + CPU->GPU upload with a
    // single GPU kernel that converts RGB float directly to NV12 in the
    // NVENC hwframe.
    AVBufferRef* hwDeviceCtx = nullptr;
    AVBufferRef* hwFramesCtx = nullptr;
    bool useCudaUpload = false;
    // Phase B sub-path. True iff the kernel supports the current output
    // (BT.709 limited, NV12 or P010). False -> CUDA path is disabled and
    // we use the regular CPU YUV pipeline (the kernels don't cover the
    // BT.2020 NCL matrix yet — that lands in Phase E.2).
    bool useCudaDirectKernel = false;
    // Phase E.1: true when the hwframe is 10-bit P010 (vs 8-bit NV12). The
    // RGB->YUV step then uses the P010 kernel.
    bool useCudaP010 = false;

    // Phase C.2 state. When useGpuBayerPipeline is true the caller passes
    // bayer + WB + LSM via WriteVideoFrameFromBayer and the encoder runs
    // the full bayer->NV12 chain on GPU. Constants here are clip-constant
    // (set once by EnableGpuBayerPipeline); per-frame data (asShotNeutral
    // and LSM) is passed each WriteVideoFrameFromBayer call.
    bool useGpuBayerPipeline = false;
    motioncam::cuda::BayerPipelineConstants bayerConsts{};
#endif
};

MovEncoder::MovEncoder(const EncodeSettings& s) : p(std::make_unique<Impl>()) {
    p->settings = s;

    const std::string container = s.containerFormat.empty() ? std::string("mov") : s.containerFormat;
    if (container == "mp4") {
        // MP4 muxer accepts H.264 / H.265 / AV1. Everything else lives in MOV.
        if (s.codec == Codec::ProRes422 || s.codec == Codec::ProRes422HQ ||
            s.codec == Codec::ProRes4444 || s.codec == Codec::ProRes4444XQ) {
            Throw("ProRes cannot be muxed into .mp4 — use .mov, or pick H.264 / H.265 / AV1 for .mp4");
        }
        if (s.codec == Codec::DNxHR_HQX || s.codec == Codec::DNxHR_444) {
            Throw("DNxHR is delivered as .mov — change format or pick H.264 / H.265 / AV1");
        }
        if (s.codec == Codec::CineForm) {
            Throw("CineForm is delivered as .mov — change format or pick H.264 / H.265 / AV1");
        }
    }
    avformat_alloc_output_context2(&p->fmt, nullptr, container.c_str(), s.outputPath.c_str());
    if (!p->fmt) Throw(("failed to allocate output context for ." + container).c_str());

    const AVCodec* venc = FindEncoder(s.codec);
    if (!venc) Throw("video encoder not found (did vcpkg build ffmpeg with x264/x265?)");

    // NVENC fallback probe: FFmpeg may have h264_nvenc/hevc_nvenc compiled in,
    // but on a machine without an NVIDIA driver / GPU the encoder fails to open.
    // Probe with a tiny throwaway context; if it can't open, swap to libx264/libx265
    // transparently so the user gets a render instead of an error.
    if (IsNvenc(s.codec)) {
        AVCodecContext* probe = avcodec_alloc_context3(venc);
        if (probe) {
            probe->width = 320;
            probe->height = 240;
            probe->time_base = {1, 30};
            probe->framerate = {30, 1};
            probe->pix_fmt = AV_PIX_FMT_YUV420P;
            probe->bit_rate = 1000000;
            int probeErr = avcodec_open2(probe, venc, nullptr);
            avcodec_free_context(&probe);
            if (probeErr < 0) {
                const char* fallbackName =
                    (s.codec == Codec::H264NVENC) ? "libx264" : "libx265";
                const AVCodec* fb = avcodec_find_encoder_by_name(fallbackName);
                if (fb) {
                    char buf[256] = {0};
                    av_strerror(probeErr, buf, sizeof(buf));
                    fprintf(stderr,
                        "[MovEncoder] %s unavailable (%s); falling back to %s\n",
                        venc->name, buf, fallbackName);
                    venc = fb;
                }
            }
        }
    }

    // 10-bit probe for libx264 / libx265 — vcpkg's default build is 8-bit only.
    // We detect by trying to open the encoder with a small 10-bit context;
    // if that fails, the user-facing 10-bit toggle is silently downgraded.
    bool effectiveTenBit = s.tenBit;
    if (s.tenBit && (s.codec == Codec::H264 || s.codec == Codec::H265)) {
        AVCodecContext* probe = avcodec_alloc_context3(venc);
        if (probe) {
            probe->width = 320;
            probe->height = 240;
            probe->time_base = {1, 30};
            probe->framerate = {30, 1};
            probe->pix_fmt = AV_PIX_FMT_YUV420P10LE;
            probe->bit_rate = 1000000;
            if (s.codec == Codec::H265) {
                av_opt_set(probe->priv_data, "profile", "main10", 0);
            }
            int probeErr = avcodec_open2(probe, venc, nullptr);
            avcodec_free_context(&probe);
            if (probeErr < 0) {
                char buf[256] = {0};
                av_strerror(probeErr, buf, sizeof(buf));
                fprintf(stderr,
                    "[MovEncoder] %s 10-bit unavailable (%s); falling back to 8-bit. "
                    "Rebuild vcpkg ffmpeg with x264/x265 multilib for 10-bit CPU encode.\n",
                    venc->name, buf);
                effectiveTenBit = false;
            }
        }
    }

    p->videoStream = avformat_new_stream(p->fmt, nullptr);
    if (!p->videoStream) Throw("avformat_new_stream(video) failed");

    p->videoCtx = avcodec_alloc_context3(venc);
    if (!p->videoCtx) Throw("avcodec_alloc_context3(video) failed");

    p->videoCtx->width = s.width;
    p->videoCtx->height = s.height;
    p->videoCtx->time_base = {s.fpsDen, s.fpsNum};
    p->videoCtx->framerate = {s.fpsNum, s.fpsDen};
    p->videoStream->time_base = p->videoCtx->time_base;

    {
        AVPixelFormat pix = PreferredPixFmt(s.codec);
        // 10-bit override. libx265 takes YUV420P10LE; NVENC encoders take P010LE
        // (NV12-style 16-bit packed). Different layouts — passing the wrong one
        // makes avcodec_open2 fail with EINVAL.
        if (effectiveTenBit) {
            if (s.codec == Codec::H265) {
                pix = AV_PIX_FMT_YUV420P10LE;
            } else if (s.codec == Codec::H265NVENC || s.codec == Codec::AV1NVENC) {
                pix = AV_PIX_FMT_P010LE;
            }
        }
        p->videoCtx->pix_fmt = pix;
    }

    const int proResProf = ProResProfile(s.codec);
    if (proResProf >= 0) {
        p->videoCtx->profile = proResProf;
        av_opt_set(p->videoCtx->priv_data, "mbs_per_slice", "4", 0);
        av_opt_set(p->videoCtx->priv_data, "vendor", "apl0", 0);
    }
    // DNxHR profile selection — the dnxhd encoder takes its profile via priv_data.
    if (s.codec == Codec::DNxHR_HQX) {
        av_opt_set(p->videoCtx->priv_data, "profile", "dnxhr_hqx", 0);
    } else if (s.codec == Codec::DNxHR_444) {
        av_opt_set(p->videoCtx->priv_data, "profile", "dnxhr_444", 0);
    }
    // CineForm: pick a high-quality preset. "film3" is the highest tier;
    // produces ~50-100 Mbps at 4K, visually lossless on natural content.
    if (s.codec == Codec::CineForm) {
        av_opt_set(p->videoCtx->priv_data, "quality", "film3", 0);
    }
    // Codecs with explicit user-controlled bitrate.
    if (s.codec == Codec::H264 || s.codec == Codec::H265 ||
        s.codec == Codec::H264NVENC || s.codec == Codec::H265NVENC ||
        s.codec == Codec::AV1NVENC) {
        p->videoCtx->bit_rate = static_cast<int64_t>(s.bitrateMbps) * 1000000;
    }
    // VBV ceiling for libx264 / libx265. Without this, x264's ABR mode
    // overshoots the target on high-detail 4K content (we observed ~7×
    // overshoot at default 80 Mbps), pushing the stream past H.264 Level 5.1
    // = 240 Mbps which is the cap for Microsoft Media Foundation's H.264
    // decoder. Files become unplayable in Windows Media Player / Movies & TV
    // ("0xC00D36C4 — file format unsupported"). Setting maxrate=2× target
    // and bufsize=1s caps the peak while still allowing some VBR flex on
    // hard-to-compress moments. NVENC path already does this above.
    {
        const bool isCpuX = venc->name && (
            std::strstr(venc->name, "libx264") != nullptr ||
            std::strstr(venc->name, "libx265") != nullptr);
        if (isCpuX && (s.codec == Codec::H264 || s.codec == Codec::H265 ||
                       s.codec == Codec::H264NVENC || s.codec == Codec::H265NVENC)) {
            p->videoCtx->rc_max_rate = static_cast<int64_t>(s.bitrateMbps) * 2 * 1000000;
            p->videoCtx->rc_buffer_size = static_cast<int>(s.bitrateMbps * 1000000);
        }
    }

    // NVENC tuning — applies to h264_nvenc, hevc_nvenc, av1_nvenc. We detect by
    // encoder name so the codec-fallback case (NVENC unavailable → libx264/x265)
    // gets the standard CPU defaults instead.
    const bool isNvenc = (venc->name && std::strstr(venc->name, "nvenc") != nullptr);
    if (isNvenc) {
        // p1 = fastest, p7 = slowest+best quality. p4 = balanced.
        // tune=hq for high-quality offline encoding (vs ll/ull for streaming).
        // rc=vbr lets the encoder hit a target avg bitrate while allowing peaks.
        av_opt_set(p->videoCtx->priv_data, "preset", "p4", 0);
        av_opt_set(p->videoCtx->priv_data, "tune",   "hq", 0);
        av_opt_set(p->videoCtx->priv_data, "rc",     "vbr", 0);
        p->videoCtx->rc_max_rate = static_cast<int64_t>(s.bitrateMbps) * 2 * 1000000;
        p->videoCtx->rc_buffer_size = static_cast<int>(s.bitrateMbps * 1000000);
        // Profile selection for 10-bit paths.
        if (effectiveTenBit) {
            if (s.codec == Codec::H265NVENC) {
                av_opt_set(p->videoCtx->priv_data, "profile", "main10", 0);
            } else if (s.codec == Codec::AV1NVENC) {
                av_opt_set(p->videoCtx->priv_data, "profile", "main", 0);
            }
        }
    }
    // libx265 10-bit Main10 — only set if the probe succeeded.
    if (effectiveTenBit && s.codec == Codec::H265 && !isNvenc) {
        av_opt_set(p->videoCtx->priv_data, "profile", "main10", 0);
    }

    // QuickTime 'nclc' color tag — caller supplies primaries / transfer / matrix
    // matching the encoded values. Resolve and friends use this to pick the input
    // transform automatically on import.
    p->videoCtx->color_primaries = static_cast<AVColorPrimaries>(s.colorPrimaries);
    p->videoCtx->color_trc       = static_cast<AVColorTransferCharacteristic>(s.colorTrc);
    p->videoCtx->colorspace      = static_cast<AVColorSpace>(s.colorMatrix);
    p->videoCtx->color_range     = AVCOL_RANGE_MPEG;

    if (p->fmt->oformat->flags & AVFMT_GLOBALHEADER) {
        p->videoCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

#if MCRAW_HAVE_CUDA
    // Tier 2.1 Phase A.2 — CUDA hwframe upload path. Opt-in via env var so
    // we can A/B test against the CPU path during v0.2.0. We require:
    //   1. The encoder is an NVENC variant (it actually consumes GPU surfaces)
    //   2. CUDA is built into this binary AND a CUDA device is visible
    //      (cuda::IsCudaAvailable does the device-count probe)
    //   3. MCRAW_GPU_YUV=1 in the environment
    // If any check fails we silently take the existing CPU YUV path.
    {
        const bool isNvencCodec =
            venc->name && std::strstr(venc->name, "nvenc") != nullptr;
        const char* envOpt = std::getenv("MCRAW_GPU_YUV");
        const bool envOptIn = envOpt && std::strcmp(envOpt, "1") == 0;

        if (isNvencCodec && envOptIn && motioncam::cuda::IsCudaAvailable()) {
            // Kernel-supported check. The RGB->YUV kernels handle 8-bit NV12
            // and 10-bit P010, both BT.709 limited. The BT.2020 NCL matrix
            // (Rec.2020 PQ/HLG delivery) still keeps the CPU YUV pipeline
            // until Phase E.2.
            const bool is8bit  = (p->videoCtx->pix_fmt == AV_PIX_FMT_YUV420P);
            const bool is10bit = (p->videoCtx->pix_fmt == AV_PIX_FMT_P010LE);
            const bool bt709Compat = (s.colorMatrix == 1 || s.colorMatrix == 2);
            if ((!is8bit && !is10bit) || !bt709Compat) {
                fprintf(stderr,
                    "[MovEncoder] CUDA path requested but output is "
                    "%s%s — using CPU YUV.\n",
                    (!is8bit && !is10bit) ? "an unsupported pixel format" : "",
                    !bt709Compat ? " / BT.2020 matrix" : "");
            } else {
                int hwErr = av_hwdevice_ctx_create(&p->hwDeviceCtx,
                    AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
                if (hwErr < 0) {
                    char buf[128]{};
                    av_strerror(hwErr, buf, sizeof(buf));
                    fprintf(stderr,
                        "[MovEncoder] CUDA hwdevice init failed (%s); "
                        "falling back to CPU YUV path.\n", buf);
                } else {
                    AVBufferRef* framesRef = av_hwframe_ctx_alloc(p->hwDeviceCtx);
                    if (framesRef) {
                        auto* fc = reinterpret_cast<AVHWFramesContext*>(framesRef->data);
                        fc->format = AV_PIX_FMT_CUDA;
                        // The kernel writes NV12 (8-bit) or P010 (10-bit)
                        // directly, so request that hwframe sw_format and let
                        // libavcodec skip its own format conversion.
                        fc->sw_format = is10bit ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
                        fc->width = s.width;
                        fc->height = s.height;
                        fc->initial_pool_size = 8;
                        int initErr = av_hwframe_ctx_init(framesRef);
                        if (initErr >= 0) {
                            p->hwFramesCtx = framesRef;
                            p->videoCtx->hw_frames_ctx = av_buffer_ref(p->hwFramesCtx);
                            p->videoCtx->pix_fmt = AV_PIX_FMT_CUDA;
                            p->useCudaUpload = true;
                            p->useCudaDirectKernel = true;
                            p->useCudaP010 = is10bit;
                            fprintf(stderr,
                                "[MovEncoder] CUDA RGB->%s kernel "
                                "(Phase B%s) enabled.\n",
                                is10bit ? "P010" : "NV12",
                                is10bit ? "/E.1" : "");
                        } else {
                            av_buffer_unref(&framesRef);
                            char buf[128]{};
                            av_strerror(initErr, buf, sizeof(buf));
                            fprintf(stderr,
                                "[MovEncoder] hwframes init failed (%s); "
                                "falling back to CPU YUV path.\n", buf);
                        }
                    }
                    if (!p->useCudaUpload && p->hwDeviceCtx) {
                        av_buffer_unref(&p->hwDeviceCtx);
                    }
                }
            }
        }
    }
#endif

    int err = avcodec_open2(p->videoCtx, venc, nullptr);
    if (err < 0) ThrowAv("avcodec_open2(video)", err);

    err = avcodec_parameters_from_context(p->videoStream->codecpar, p->videoCtx);
    if (err < 0) ThrowAv("avcodec_parameters_from_context(video)", err);

    p->yuvFrame = av_frame_alloc();
    if (!p->yuvFrame) Throw("av_frame_alloc(yuv) failed");
    // yuvFrame is the *CPU* staging buffer for sws_scale output. When the
    // CUDA upload path is active, videoCtx->pix_fmt is AV_PIX_FMT_CUDA —
    // we need to allocate yuvFrame with the underlying software format
    // (recorded in hwFramesCtx->sw_format) instead.
    AVPixelFormat yuvFormat = p->videoCtx->pix_fmt;
#if MCRAW_HAVE_CUDA
    if (p->useCudaUpload && p->hwFramesCtx) {
        yuvFormat = reinterpret_cast<AVHWFramesContext*>(p->hwFramesCtx->data)->sw_format;
    }
#endif
    p->yuvFrame->format = yuvFormat;
    p->yuvFrame->width = s.width;
    p->yuvFrame->height = s.height;
    err = av_frame_get_buffer(p->yuvFrame, 0);
    if (err < 0) ThrowAv("av_frame_get_buffer(yuv)", err);

    if (s.audioSampleRate > 0 && s.audioChannels > 0) {
        // MP4: AAC (universally supported by media players).
        // MOV: PCM_S16LE (lossless, what NLEs expect).
        const bool useAac = (container == "mp4");
        const AVCodecID audio_codec_id = useAac ? AV_CODEC_ID_AAC : AV_CODEC_ID_PCM_S16LE;
        const AVSampleFormat audio_sample_fmt = useAac ? AV_SAMPLE_FMT_FLTP : AV_SAMPLE_FMT_S16;

        const AVCodec* aenc = avcodec_find_encoder(audio_codec_id);
        if (!aenc) Throw(useAac ? "AAC encoder not found" : "PCM_S16LE encoder not found");

        // Configure and OPEN the audio context BEFORE attaching it as a
        // stream — that way if the encoder rejects the sample rate / channel
        // layout (AAC is picky about both), we can fall back to video-only
        // without leaving a half-configured stream in the muxer.
        AVCodecContext* audCtx = avcodec_alloc_context3(aenc);
        if (!audCtx) Throw("avcodec_alloc_context3(audio) failed");

        audCtx->sample_rate = s.audioSampleRate;
        audCtx->sample_fmt = audio_sample_fmt;
        if (useAac) {
            // 192 kbps stereo @ 48 kHz is transparent for typical content.
            audCtx->bit_rate = 192000;
        }
        av_channel_layout_default(&audCtx->ch_layout, s.audioChannels);
        audCtx->time_base = {1, s.audioSampleRate};

        if (p->fmt->oformat->flags & AVFMT_GLOBALHEADER) {
            audCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        err = avcodec_open2(audCtx, aenc, nullptr);
        if (err < 0) {
            // Audio open failed — typically AAC rejecting an unusual sample
            // rate, or a channel layout it can't represent. Don't fail the
            // whole render; surface the cause on stderr (captured in the GUI
            // log) and proceed with video-only.
            char buf[256] = {0};
            av_strerror(err, buf, sizeof(buf));
            fprintf(stderr,
                "[MovEncoder] audio encoder failed to open "
                "(sr=%d ch=%d: %s); rendering video-only.\n",
                s.audioSampleRate, s.audioChannels, buf);
            avcodec_free_context(&audCtx);
        } else {
            p->audioStream = avformat_new_stream(p->fmt, nullptr);
            if (!p->audioStream) {
                avcodec_free_context(&audCtx);
                Throw("avformat_new_stream(audio) failed");
            }
            p->audioStream->time_base = audCtx->time_base;
            p->audioCtx = audCtx;
            err = avcodec_parameters_from_context(p->audioStream->codecpar, p->audioCtx);
            if (err < 0) ThrowAv("avcodec_parameters_from_context(audio)", err);
        }
    }

    err = avio_open(&p->fmt->pb, s.outputPath.c_str(), AVIO_FLAG_WRITE);
    if (err < 0) ThrowAv("avio_open", err);

    err = avformat_write_header(p->fmt, nullptr);
    if (err < 0) ThrowAv("avformat_write_header", err);

    // sws_scale runs on CPU and outputs the *software* YUV format; the
    // CUDA path then uploads that to the GPU separately. So we use the
    // same yuvFormat the yuvFrame was allocated with, not pix_fmt (which
    // may be AV_PIX_FMT_CUDA on the GPU path).
    p->sws = sws_getContext(
        s.width, s.height, AV_PIX_FMT_GBRPF32LE,
        s.width, s.height, yuvFormat,
        SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!p->sws) Throw("sws_getContext failed");

    p->rgbStaging = av_frame_alloc();
    if (!p->rgbStaging) Throw("av_frame_alloc(rgb staging) failed");
    p->rgbStaging->format = AV_PIX_FMT_GBRPF32LE;
    p->rgbStaging->width = s.width;
    p->rgbStaging->height = s.height;
    err = av_frame_get_buffer(p->rgbStaging, 0);
    if (err < 0) ThrowAv("av_frame_get_buffer(rgb staging)", err);

    p->pkt = av_packet_alloc();
    if (!p->pkt) Throw("av_packet_alloc failed");
}

MovEncoder::~MovEncoder() {
    if (!p->finalized && p->fmt) {
        try { Finalize(); } catch (...) {}
    }
    if (p->yuvFrame) av_frame_free(&p->yuvFrame);
    if (p->rgbStaging) av_frame_free(&p->rgbStaging);
    if (p->sws) sws_freeContext(p->sws);
    // Destroy encoder first — it holds refs on hw_frames_ctx / hw_device_ctx
    // and must be torn down before we unref those.
    if (p->videoCtx) avcodec_free_context(&p->videoCtx);
    if (p->audioCtx) avcodec_free_context(&p->audioCtx);
#if MCRAW_HAVE_CUDA
    if (p->hwFramesCtx) av_buffer_unref(&p->hwFramesCtx);
    if (p->hwDeviceCtx) av_buffer_unref(&p->hwDeviceCtx);
    // Release the persistent RGB upload buffer. Cheap to re-allocate on
    // the next render, and avoids holding VRAM after the encoder is gone.
    if (p->useCudaDirectKernel) {
        motioncam::cuda::ReleaseRgbScratch();
    }
    // Phase D: free the clip-specific OCIO 3D LUT (no-op if none loaded).
    if (p->useGpuBayerPipeline) {
        motioncam::cuda::ReleaseLut3D();
    }
#endif
    if (p->pkt) av_packet_free(&p->pkt);
    if (p->fmt) {
        if (p->fmt->pb) avio_closep(&p->fmt->pb);
        avformat_free_context(p->fmt);
    }
}

void MovEncoder::WriteVideoFrame(const float* rgb) {
    const int w = p->settings.width;
    const int h = p->settings.height;

    AVFrame* frameToSend = p->yuvFrame;
    int err = 0;

#if MCRAW_HAVE_CUDA
    AVFrame* hwFrame = nullptr;
    if (p->useCudaDirectKernel) {
        // Phase B fast path. Skips sws_scale entirely: kernel takes the
        // interleaved RGB float buffer and writes NV12 (8-bit) or P010
        // (10-bit) straight into the NVENC hwframe. The H->D copy of the
        // RGB happens inside the kernel wrapper (one cudaMemcpyAsync), so we
        // don't touch p->yuvFrame / p->rgbStaging / p->sws at all.
        hwFrame = av_frame_alloc();
        if (!hwFrame) Throw("av_frame_alloc(hw) failed");
        int hwErr = av_hwframe_get_buffer(p->hwFramesCtx, hwFrame, 0);
        if (hwErr < 0) {
            av_frame_free(&hwFrame);
            ThrowAv("av_hwframe_get_buffer", hwErr);
        }
        const bool ok = p->useCudaP010
            ? motioncam::cuda::RgbFloatToP010(
                rgb, hwFrame->data[0], hwFrame->data[1],
                w, h, hwFrame->linesize[0], hwFrame->linesize[1])
            : motioncam::cuda::RgbFloatToNv12(
                rgb, hwFrame->data[0], hwFrame->data[1],
                w, h, hwFrame->linesize[0], hwFrame->linesize[1]);
        if (!ok) {
            av_frame_free(&hwFrame);
            Throw(p->useCudaP010 ? "RgbFloatToP010 kernel failed"
                                 : "RgbFloatToNv12 kernel failed");
        }
        frameToSend = hwFrame;
        goto send_frame;
    }
#endif

    err = av_frame_make_writable(p->rgbStaging);
    if (err < 0) ThrowAv("av_frame_make_writable(rgb staging)", err);
    err = av_frame_make_writable(p->yuvFrame);
    if (err < 0) ThrowAv("av_frame_make_writable(yuv)", err);

    // GBRPF32LE: planar G/B/R order. Deinterleave RGB float → GBR planes, clamp to [0,1].
    {
        const int gStride = p->rgbStaging->linesize[0] / static_cast<int>(sizeof(float));
        const int bStride = p->rgbStaging->linesize[1] / static_cast<int>(sizeof(float));
        const int rStride = p->rgbStaging->linesize[2] / static_cast<int>(sizeof(float));
        float* gPlane = reinterpret_cast<float*>(p->rgbStaging->data[0]);
        float* bPlane = reinterpret_cast<float*>(p->rgbStaging->data[1]);
        float* rPlane = reinterpret_cast<float*>(p->rgbStaging->data[2]);

        for (int y = 0; y < h; ++y) {
            const float* src = rgb + static_cast<size_t>(y) * static_cast<size_t>(w) * 3;
            float* gRow = gPlane + static_cast<size_t>(y) * static_cast<size_t>(gStride);
            float* bRow = bPlane + static_cast<size_t>(y) * static_cast<size_t>(bStride);
            float* rRow = rPlane + static_cast<size_t>(y) * static_cast<size_t>(rStride);
            for (int x = 0; x < w; ++x) {
                float r = src[x*3 + 0];
                float g = src[x*3 + 1];
                float b = src[x*3 + 2];
                if (r < 0.0f) r = 0.0f; else if (r > 1.0f) r = 1.0f;
                if (g < 0.0f) g = 0.0f; else if (g > 1.0f) g = 1.0f;
                if (b < 0.0f) b = 0.0f; else if (b > 1.0f) b = 1.0f;
                gRow[x] = g;
                bRow[x] = b;
                rRow[x] = r;
            }
        }
    }

    sws_scale(p->sws,
        p->rgbStaging->data, p->rgbStaging->linesize, 0, h,
        p->yuvFrame->data, p->yuvFrame->linesize);

#if MCRAW_HAVE_CUDA
    if (p->useCudaUpload && !p->useCudaDirectKernel) {
        // Phase A.2 path. Used when MCRAW_GPU_YUV=1 was requested but the
        // direct kernel doesn't support this output config (10-bit / BT.2020):
        // uploads CPU YUV to GPU and lets NVENC consume that. In v0.2.0 this
        // path is unreachable because we disable useCudaUpload when the
        // kernel can't run — kept as scaffolding for when we add the 10-bit
        // / BT.2020 kernels.
        hwFrame = av_frame_alloc();
        if (!hwFrame) Throw("av_frame_alloc(hw) failed");
        int hwErr = av_hwframe_get_buffer(p->hwFramesCtx, hwFrame, 0);
        if (hwErr < 0) {
            av_frame_free(&hwFrame);
            ThrowAv("av_hwframe_get_buffer", hwErr);
        }
        hwErr = av_hwframe_transfer_data(hwFrame, p->yuvFrame, 0);
        if (hwErr < 0) {
            av_frame_free(&hwFrame);
            ThrowAv("av_hwframe_transfer_data", hwErr);
        }
        frameToSend = hwFrame;
    }
#endif

#if MCRAW_HAVE_CUDA
send_frame:
#endif
    frameToSend->pts = p->videoPts++;
    err = avcodec_send_frame(p->videoCtx, frameToSend);

#if MCRAW_HAVE_CUDA
    if (hwFrame) av_frame_free(&hwFrame);
#endif

    if (err < 0) ThrowAv("avcodec_send_frame(video)", err);

    while (true) {
        int rec = avcodec_receive_packet(p->videoCtx, p->pkt);
        if (rec == AVERROR(EAGAIN) || rec == AVERROR_EOF) break;
        if (rec < 0) ThrowAv("avcodec_receive_packet(video)", rec);

        av_packet_rescale_ts(p->pkt, p->videoCtx->time_base, p->videoStream->time_base);
        p->pkt->stream_index = p->videoStream->index;
        err = av_interleaved_write_frame(p->fmt, p->pkt);
        if (err < 0) ThrowAv("av_interleaved_write_frame(video)", err);
        av_packet_unref(p->pkt);
    }
}

void MovEncoder::WriteAudio(const int16_t* samples, int numSamplesTotal) {
    if (!p->audioCtx || numSamplesTotal <= 0) return;

    const int channels = p->settings.audioChannels;
    const int totalPerCh = numSamplesTotal / channels;
    if (totalPerCh <= 0) return;

    auto drainPackets = [&]() {
        while (true) {
            int rec = avcodec_receive_packet(p->audioCtx, p->pkt);
            if (rec == AVERROR(EAGAIN) || rec == AVERROR_EOF) break;
            if (rec < 0) ThrowAv("avcodec_receive_packet(audio)", rec);
            av_packet_rescale_ts(p->pkt, p->audioCtx->time_base, p->audioStream->time_base);
            p->pkt->stream_index = p->audioStream->index;
            int err = av_interleaved_write_frame(p->fmt, p->pkt);
            if (err < 0) ThrowAv("av_interleaved_write_frame(audio)", err);
            av_packet_unref(p->pkt);
        }
    };

    if (p->audioCtx->sample_fmt == AV_SAMPLE_FMT_S16) {
        // PCM path — one big AVFrame, S16 interleaved as-is.
        AVFrame* af = av_frame_alloc();
        af->format = AV_SAMPLE_FMT_S16;
        af->nb_samples = totalPerCh;
        av_channel_layout_copy(&af->ch_layout, &p->audioCtx->ch_layout);
        af->sample_rate = p->settings.audioSampleRate;
        int err = av_frame_get_buffer(af, 0);
        if (err < 0) { av_frame_free(&af); ThrowAv("av_frame_get_buffer(audio s16)", err); }
        std::memcpy(af->data[0], samples,
                    static_cast<size_t>(numSamplesTotal) * sizeof(int16_t));
        af->pts = p->audioPts;
        p->audioPts += totalPerCh;
        err = avcodec_send_frame(p->audioCtx, af);
        av_frame_free(&af);
        if (err < 0) ThrowAv("avcodec_send_frame(audio s16)", err);
        drainPackets();
        return;
    }

    // AAC path — convert S16 interleaved to FLTP planar float, slice into
    // encoder-frame-sized chunks (typically 1024 samples per channel for AAC),
    // send one frame at a time.
    const int frameSize = p->audioCtx->frame_size > 0 ? p->audioCtx->frame_size : 1024;
    const float invScale = 1.0f / 32768.0f;

    int written = 0;
    while (written < totalPerCh) {
        const int n = std::min(frameSize, totalPerCh - written);

        AVFrame* af = av_frame_alloc();
        af->format = AV_SAMPLE_FMT_FLTP;
        af->nb_samples = n;
        av_channel_layout_copy(&af->ch_layout, &p->audioCtx->ch_layout);
        af->sample_rate = p->settings.audioSampleRate;
        int err = av_frame_get_buffer(af, 0);
        if (err < 0) { av_frame_free(&af); ThrowAv("av_frame_get_buffer(audio fltp)", err); }

        // Deinterleave + convert int16 -> float per channel
        for (int ch = 0; ch < channels; ++ch) {
            float* dst = reinterpret_cast<float*>(af->data[ch]);
            const int16_t* src = samples + written * channels + ch;
            for (int i = 0; i < n; ++i) {
                dst[i] = static_cast<float>(src[i * channels]) * invScale;
            }
        }

        af->pts = p->audioPts;
        p->audioPts += n;
        written += n;

        err = avcodec_send_frame(p->audioCtx, af);
        av_frame_free(&af);
        if (err < 0) ThrowAv("avcodec_send_frame(audio aac)", err);
        drainPackets();
    }
}

void MovEncoder::Finalize() {
    if (p->finalized) return;

    if (p->videoCtx) {
        avcodec_send_frame(p->videoCtx, nullptr);
        while (true) {
            int rec = avcodec_receive_packet(p->videoCtx, p->pkt);
            if (rec == AVERROR(EAGAIN) || rec == AVERROR_EOF) break;
            if (rec < 0) break;
            av_packet_rescale_ts(p->pkt, p->videoCtx->time_base, p->videoStream->time_base);
            p->pkt->stream_index = p->videoStream->index;
            av_interleaved_write_frame(p->fmt, p->pkt);
            av_packet_unref(p->pkt);
        }
    }

    if (p->audioCtx) {
        avcodec_send_frame(p->audioCtx, nullptr);
        while (true) {
            int rec = avcodec_receive_packet(p->audioCtx, p->pkt);
            if (rec == AVERROR(EAGAIN) || rec == AVERROR_EOF) break;
            if (rec < 0) break;
            av_packet_rescale_ts(p->pkt, p->audioCtx->time_base, p->audioStream->time_base);
            p->pkt->stream_index = p->audioStream->index;
            av_interleaved_write_frame(p->fmt, p->pkt);
            av_packet_unref(p->pkt);
        }
    }

    av_write_trailer(p->fmt);
    p->finalized = true;
}

bool ParseCodec(const std::string& s, Codec& out) {
    if (s == "prores422" || s == "prores-422")          { out = Codec::ProRes422;    return true; }
    if (s == "prores422hq" || s == "prores-422hq")      { out = Codec::ProRes422HQ;  return true; }
    if (s == "prores4444" || s == "prores-4444")        { out = Codec::ProRes4444;   return true; }
    if (s == "prores4444xq" || s == "prores-4444xq")    { out = Codec::ProRes4444XQ; return true; }
    if (s == "h264" || s == "x264")                     { out = Codec::H264;         return true; }
    if (s == "h265" || s == "hevc" || s == "x265")      { out = Codec::H265;         return true; }
    if (s == "h264_nvenc" || s == "h264-nvenc" || s == "h264_gpu") {
        out = Codec::H264NVENC; return true;
    }
    if (s == "h265_nvenc" || s == "hevc_nvenc" || s == "h265-nvenc" || s == "h265_gpu") {
        out = Codec::H265NVENC; return true;
    }
    if (s == "av1_nvenc" || s == "av1-nvenc" || s == "av1_gpu" || s == "av1") {
        out = Codec::AV1NVENC; return true;
    }
    if (s == "dnxhr_hqx" || s == "dnxhr-hqx" || s == "dnxhr_10")     { out = Codec::DNxHR_HQX; return true; }
    if (s == "dnxhr_444" || s == "dnxhr-444" || s == "dnxhr_12")     { out = Codec::DNxHR_444; return true; }
    if (s == "cineform" || s == "cfhd")                              { out = Codec::CineForm;  return true; }
    return false;
}

const char* CodecName(Codec c) {
    switch (c) {
        case Codec::ProRes422:    return "prores422";
        case Codec::ProRes422HQ:  return "prores422hq";
        case Codec::ProRes4444:   return "prores4444";
        case Codec::ProRes4444XQ: return "prores4444xq";
        case Codec::H264:         return "h264";
        case Codec::H265:         return "h265";
        case Codec::H264NVENC:    return "h264_nvenc";
        case Codec::H265NVENC:    return "h265_nvenc";
        case Codec::AV1NVENC:     return "av1_nvenc";
        case Codec::DNxHR_HQX:    return "dnxhr_hqx";
        case Codec::DNxHR_444:    return "dnxhr_444";
        case Codec::CineForm:     return "cineform";
    }
    return "unknown";
}

bool IsNvenc(Codec c) {
    return c == Codec::H264NVENC || c == Codec::H265NVENC || c == Codec::AV1NVENC;
}

#if MCRAW_HAVE_CUDA
// ---------------------------------------------------------------------------
//  Tier 2.1 Phase C.2 — GPU bayer pipeline integration
// ---------------------------------------------------------------------------

namespace {

// All the constants we need to reproduce ColorPipeline.cpp's per-target
// matrix on the encoder side. Kept here so we don't have to drag the
// color pipeline into MovEncoder.hpp; refactored into a shared helper if
// we end up needing this in a third place.

constexpr float kBradfordD50toD60[9] = {
     0.96766f, -0.01686f,  0.04424f,
    -0.02099f,  1.00778f,  0.01477f,
     0.00853f, -0.01415f,  1.22963f,
};
constexpr float kXyzD60toAP1[9] = {
     1.6410233797f, -0.3248032942f, -0.2364246952f,
    -0.6636628587f,  1.6153315917f,  0.0167563477f,
     0.0117218943f, -0.0082844420f,  0.9883948585f,
};
// Same AP1->Rec.709 matrix as BakedTransform.cpp.
constexpr float kAP1toRec709[9] = {
     1.7050514f, -0.6217908f, -0.0832606f,
    -0.1302561f,  1.1408047f, -0.0105486f,
    -0.0240083f, -0.1289693f,  1.1529777f,
};
// AP1 -> AP0 from the ACES spec.
constexpr float kAP1toAP0[9] = {
     0.6954522f,  0.1406787f,  0.1638691f,
     0.0447946f,  0.8596711f,  0.0955343f,
    -0.0055258f,  0.0040252f,  1.0015007f,
};

void Mat3Mul(const float A[9], const float B[9], float out[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out[i*3+j] = A[i*3+0]*B[j] + A[i*3+1]*B[3+j] + A[i*3+2]*B[6+j];
}

// Mirrors ColorPipeline.cpp BuildCameraToAcescg / BakedTransform's plan
// table. Output: combined cam -> target matrix + curve code matching the
// CUDA kernel's enum (0=None, 1=Gamma22, 2=Gamma24, 3=SRGB).
//
// Returns false if `target` isn't bake-compatible — the caller should
// skip Phase C in that case.
//
// `target` is OutputColorSpace cast to int; values come from
// motioncam::color::OutputColorSpace which we don't include here to keep
// the encoder loosely coupled. The supported integer values must match
// the enum's declaration order in ColorPipeline.hpp:
//   0 ACEScg, 1 LinearRec709, 2 ACES2065_1, 6 Rec709Gamma22,
//   7 Rec709Display, 5 SRGB
// (See OutputColorSpace in ColorPipeline.hpp for the canonical list.)
bool BuildBakedCamToOutput(
    int target, const float forwardMatrix2[9],
    float outMatrix[9], int& outCurve)
{
    // Build cam -> ACEScg (D50 -> D60 -> AP1) as the common base.
    float tmp[9], camToAcescg[9];
    Mat3Mul(kBradfordD50toD60, forwardMatrix2, tmp);
    Mat3Mul(kXyzD60toAP1, tmp, camToAcescg);

    const float* baked = nullptr;
    outCurve = 0;
    switch (target) {
        case 0:  baked = nullptr;        outCurve = 0; break;  // ACEScg
        case 1:  baked = kAP1toRec709;   outCurve = 0; break;  // LinearRec709
        case 2:  baked = kAP1toAP0;      outCurve = 0; break;  // ACES2065_1
        case 5:  baked = kAP1toRec709;   outCurve = 3; break;  // SRGB
        case 6:  baked = kAP1toRec709;   outCurve = 1; break;  // Rec709Gamma22
        case 7:  baked = kAP1toRec709;   outCurve = 2; break;  // Rec709Display
        default: return false;
    }

    if (baked) {
        Mat3Mul(baked, camToAcescg, outMatrix);
    } else {
        std::memcpy(outMatrix, camToAcescg, sizeof(float) * 9);
    }
    return true;
}

// Same CFA -> channel mapping the CPU NormalizeBayer uses (per the
// CfaChannelMap helper in Debayer.cpp). Caller passes CfaPattern as int.
void CfaMaps(int cfaPattern, int chMap[4], int cfaToLsm[4]) {
    switch (cfaPattern) {
        case 0:  // RGGB
            chMap[0]=0; chMap[1]=1; chMap[2]=1; chMap[3]=2;
            cfaToLsm[0]=0; cfaToLsm[1]=1; cfaToLsm[2]=2; cfaToLsm[3]=3; return;
        case 1:  // BGGR
            chMap[0]=2; chMap[1]=1; chMap[2]=1; chMap[3]=0;
            cfaToLsm[0]=3; cfaToLsm[1]=2; cfaToLsm[2]=1; cfaToLsm[3]=0; return;
        case 2:  // GRBG
            chMap[0]=1; chMap[1]=0; chMap[2]=2; chMap[3]=1;
            cfaToLsm[0]=1; cfaToLsm[1]=0; cfaToLsm[2]=3; cfaToLsm[3]=2; return;
        case 3:  // GBRG
            chMap[0]=1; chMap[1]=2; chMap[2]=0; chMap[3]=1;
            cfaToLsm[0]=2; cfaToLsm[1]=3; cfaToLsm[2]=0; cfaToLsm[3]=1; return;
        default:
            // Fall back to RGGB so we never get uninitialised garbage.
            chMap[0]=0; chMap[1]=1; chMap[2]=1; chMap[3]=2;
            cfaToLsm[0]=0; cfaToLsm[1]=1; cfaToLsm[2]=2; cfaToLsm[3]=3; return;
    }
}

}  // namespace
#endif  // MCRAW_HAVE_CUDA

bool MovEncoder::EnableGpuBayerPipeline(const GpuBayerSetup& setup) {
#if MCRAW_HAVE_CUDA
    // Phase C builds on Phase B's NV12 hwframe pipeline. If Phase B isn't
    // active for this encoder (e.g. user didn't set MCRAW_GPU_YUV, or the
    // codec / output combo isn't supported), Phase C can't run either.
    if (!p->useCudaDirectKernel) return false;

    // Build the combined cam -> target matrix and the curve code. If the
    // target has no BakedTransform, fall back to the Phase D path: cam ->
    // ACEScg matrix followed by an OCIO-baked 3D LUT.
    float fullMatrix[9];
    int   curve = 0;
    bool  useLut = false;
    if (!BuildBakedCamToOutput(setup.targetColorSpace,
                                setup.forwardMatrix2,
                                fullMatrix, curve)) {
        // Not bake-compatible. Try Phase D (OCIO 3D LUT) if the caller
        // supplied an OCIO colour-space name.
        if (setup.ocioColorSpace.empty()) {
            fprintf(stderr,
                "[MovEncoder] Phase C/D: target colourspace %d has no "
                "BakedTransform and no OCIO name supplied; staying on the "
                "CPU bayer pipeline (Phase B RGB->NV12 still on GPU).\n",
                setup.targetColorSpace);
            return false;
        }
        // cam -> ACEScg base matrix (case 0 = ACEScg, curve None).
        if (!BuildBakedCamToOutput(0, setup.forwardMatrix2, fullMatrix, curve)) {
            return false;  // unreachable: ACEScg is always bake-compatible
        }
        curve = 0;
        try {
            std::vector<float> lut = motioncam::color::BakeAcesCgToTargetLut3D(
                setup.ocioColorSpace,
                motioncam::cuda::kLutSize,
                motioncam::cuda::kLutShaperK,
                motioncam::cuda::kLutShaperLo,
                motioncam::cuda::kLutShaperHi);
            if (!motioncam::cuda::SetupLut3D(lut.data(), motioncam::cuda::kLutSize)) {
                fprintf(stderr,
                    "[MovEncoder] Phase D: SetupLut3D failed for '%s'; "
                    "CPU bayer pipeline.\n", setup.ocioColorSpace.c_str());
                return false;
            }
        } catch (const std::exception& e) {
            fprintf(stderr,
                "[MovEncoder] Phase D: LUT bake failed for '%s' (%s); "
                "CPU bayer pipeline.\n", setup.ocioColorSpace.c_str(), e.what());
            return false;
        }
        useLut = true;
    }

    int chMap[4], cfaToLsm[4];
    CfaMaps(setup.cfaPattern, chMap, cfaToLsm);

    auto& C = p->bayerConsts;
    std::memcpy(C.cam_to_output, fullMatrix, sizeof(fullMatrix));
    for (int i = 0; i < 4; ++i) {
        C.black[i]        = setup.blackPerPosition[i];
        const double dr   = setup.whiteLevel - double(setup.blackPerPosition[i]);
        C.inv_range[i]    = dr > 0.0 ? float(1.0 / dr) : 0.0f;
        C.cfa_channel[i]  = chMap[i];
        C.cfa_to_lsm[i]   = cfaToLsm[i];
    }
    C.curve     = curve;
    C.width     = p->settings.width;
    C.height    = p->settings.height;
    C.use_lut3d = useLut ? 1 : 0;
    C.lsm_w  = 0;
    C.lsm_h  = 0;
    C.lsm_host = nullptr;

    p->useGpuBayerPipeline = true;
    if (useLut) {
        fprintf(stderr,
            "[MovEncoder] Phase D bayer+LUT pipeline enabled "
            "(target=%d, OCIO '%s', %d^3 LUT).\n",
            setup.targetColorSpace, setup.ocioColorSpace.c_str(),
            motioncam::cuda::kLutSize);
    } else {
        fprintf(stderr,
            "[MovEncoder] Phase C bayer pipeline enabled (target=%d, curve=%d).\n",
            setup.targetColorSpace, curve);
    }
    return true;
#else
    (void)setup;
    return false;
#endif
}

bool MovEncoder::HasGpuBayerPipeline() const {
#if MCRAW_HAVE_CUDA
    return p->useGpuBayerPipeline;
#else
    return false;
#endif
}

void MovEncoder::WriteVideoFrameFromBayer(
    const uint16_t* bayer,
    const float wb[3],
    const float* lsm,
    int lsmWidth,
    int lsmHeight)
{
#if MCRAW_HAVE_CUDA
    if (!p->useGpuBayerPipeline) {
        Throw("WriteVideoFrameFromBayer called but Phase C bayer pipeline "
              "isn't active; caller must check HasGpuBayerPipeline() first.");
    }

    // Per-frame LSM (matches what the CPU pipeline does in Debayer.cpp).
    auto& C = p->bayerConsts;
    if (lsm && lsmWidth >= 2 && lsmHeight >= 2) {
        C.lsm_w    = lsmWidth;
        C.lsm_h    = lsmHeight;
        C.lsm_host = lsm;
    } else {
        C.lsm_w    = 0;
        C.lsm_h    = 0;
        C.lsm_host = nullptr;
    }

    // Borrow an NV12 hwframe from the pool (same pool as Phase B uses).
    AVFrame* hwFrame = av_frame_alloc();
    if (!hwFrame) Throw("av_frame_alloc(hw) failed");
    int hwErr = av_hwframe_get_buffer(p->hwFramesCtx, hwFrame, 0);
    if (hwErr < 0) {
        av_frame_free(&hwFrame);
        ThrowAv("av_hwframe_get_buffer", hwErr);
    }

    // Full chain on GPU: bayer -> normalise -> LSM -> debayer -> matrix
    // + curve (+ optional 3D LUT) -> NV12 (8-bit) or P010 (10-bit). Writes
    // directly into the hwframe's Y/UV planes.
    const bool ok = p->useCudaP010
        ? motioncam::cuda::ProcessBayerToP010(
            bayer, wb, C,
            hwFrame->data[0], hwFrame->data[1],
            hwFrame->linesize[0], hwFrame->linesize[1])
        : motioncam::cuda::ProcessBayerToNv12(
            bayer, wb, C,
            hwFrame->data[0], hwFrame->data[1],
            hwFrame->linesize[0], hwFrame->linesize[1]);
    if (!ok) {
        av_frame_free(&hwFrame);
        Throw(p->useCudaP010 ? "ProcessBayerToP010 kernel failed"
                             : "ProcessBayerToNv12 kernel failed");
    }

    hwFrame->pts = p->videoPts++;
    int err = avcodec_send_frame(p->videoCtx, hwFrame);
    av_frame_free(&hwFrame);
    if (err < 0) ThrowAv("avcodec_send_frame(video, hw bayer)", err);

    // Drain encoded packets — same code as the regular WriteVideoFrame.
    while (true) {
        int rec = avcodec_receive_packet(p->videoCtx, p->pkt);
        if (rec == AVERROR(EAGAIN) || rec == AVERROR_EOF) break;
        if (rec < 0) ThrowAv("avcodec_receive_packet(video)", rec);
        av_packet_rescale_ts(p->pkt, p->videoCtx->time_base, p->videoStream->time_base);
        p->pkt->stream_index = p->videoStream->index;
        err = av_interleaved_write_frame(p->fmt, p->pkt);
        if (err < 0) ThrowAv("av_interleaved_write_frame(video)", err);
        av_packet_unref(p->pkt);
    }
#else
    (void)bayer; (void)wb; (void)lsm; (void)lsmWidth; (void)lsmHeight;
    Throw("WriteVideoFrameFromBayer: this build was compiled without CUDA "
          "support; rebuild with MCRAW_ENABLE_CUDA=ON.");
#endif
}

bool IsTenBitNative(Codec c) {
    // ProRes / DNxHR / CineForm always carry >= 10-bit pix_fmts.
    switch (c) {
        case Codec::ProRes422:
        case Codec::ProRes422HQ:
        case Codec::ProRes4444:
        case Codec::ProRes4444XQ:
        case Codec::DNxHR_HQX:
        case Codec::DNxHR_444:
        case Codec::CineForm:
            return true;
        default:
            return false;
    }
}

}
}
