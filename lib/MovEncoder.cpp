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
    // (8-bit NV12 / BT.709 limited). False -> CUDA path is disabled and
    // we use the regular CPU YUV pipeline (Phase B doesn't cover Rec.2020
    // or 10-bit yet — those land in a follow-up patch).
    bool useCudaDirectKernel = false;
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
            // Phase B kernel-supported check. The direct RGB->NV12 kernel
            // only handles 8-bit BT.709 / unspecified at the moment.
            // Other cases (10-bit P010, BT.2020 matrix for Rec.2020 PQ/HLG
            // delivery) silently keep the CPU YUV pipeline — correctness
            // first, more kernels in a follow-up patch.
            const bool is8bit = (p->videoCtx->pix_fmt == AV_PIX_FMT_YUV420P);
            const bool bt709Compat = (s.colorMatrix == 1 || s.colorMatrix == 2);
            if (!is8bit || !bt709Compat) {
                fprintf(stderr,
                    "[MovEncoder] CUDA path requested but output is "
                    "%s%s — using CPU YUV.\n",
                    !is8bit ? "10-bit" : "",
                    !bt709Compat ? " / BT.2020" : "");
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
                        // Phase B writes NV12 directly from the kernel — request
                        // an NV12 hwframe instead of YUV420P so libavcodec
                        // doesn't insert an extra format-conversion step.
                        fc->sw_format = AV_PIX_FMT_NV12;
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
                            fprintf(stderr,
                                "[MovEncoder] CUDA RGB->NV12 kernel "
                                "(Phase B) enabled.\n");
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
        // interleaved RGB float buffer and writes NV12 straight into the
        // NVENC hwframe. The H->D copy of the RGB happens inside the
        // kernel wrapper (one cudaMemcpyAsync), so we don't touch
        // p->yuvFrame / p->rgbStaging / p->sws at all.
        hwFrame = av_frame_alloc();
        if (!hwFrame) Throw("av_frame_alloc(hw) failed");
        int hwErr = av_hwframe_get_buffer(p->hwFramesCtx, hwFrame, 0);
        if (hwErr < 0) {
            av_frame_free(&hwFrame);
            ThrowAv("av_hwframe_get_buffer", hwErr);
        }
        const bool ok = motioncam::cuda::RgbFloatToNv12(
            rgb,
            hwFrame->data[0], hwFrame->data[1],
            w, h,
            hwFrame->linesize[0], hwFrame->linesize[1]);
        if (!ok) {
            av_frame_free(&hwFrame);
            Throw("RgbFloatToNv12 kernel failed");
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
