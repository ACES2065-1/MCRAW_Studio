#include <motioncam/Decoder.hpp>
#include <motioncam/ColorPipeline.hpp>
#include <motioncam/BakedTransform.hpp>
#include <motioncam/Denoise.hpp>
#include <motioncam/ExrWriter.hpp>
#include <motioncam/OcioTransform.hpp>
#include <motioncam/MovEncoder.hpp>
#include <motioncam/Trimmer.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

enum class OutputFormat { Exr, Mov, Mp4, Mcraw };

bool IsVideoFormat(OutputFormat f) { return f == OutputFormat::Mov || f == OutputFormat::Mp4; }
const char* ContainerName(OutputFormat f) { return f == OutputFormat::Mp4 ? "mp4" : "mov"; }

struct Args {
    std::string input;
    std::string output;
    int startFrame = 0;
    int endFrame = -1;
    motioncam::color::OutputColorSpace colorSpace = motioncam::color::OutputColorSpace::ACEScg;
    OutputFormat format = OutputFormat::Exr;
    bool formatExplicit = false;
    motioncam::video::Codec codec = motioncam::video::Codec::ProRes4444;
    int bitrateMbps = 80;
    double fpsOverride = 0.0;
    motioncam::color::ExrCompression exrCompression = motioncam::color::ExrCompression::ZIP;
    int denoiseChroma = 0;   // 0..100, MP4-only (mov/exr ignore)
    int denoiseLuma   = 0;   // 0..100, MP4-only
    bool tenBit = false;     // H.265 / AV1: encode 10-bit. ProRes/DNxHR/CineForm are already 10/12-bit.
    bool highlightRecovery = false;  // Pre-WB saturation neutralisation + (display-encoded only) soft knee
};

void PrintUsage() {
    std::cerr <<
        "Usage: mcraw_render <input.mcraw> [options]\n"
        "  -o, --output <path>     Output file (.mov) or directory (EXR sequence). Default: frames\n"
        "  -f, --format <fmt>      Output format: exr | mov. Inferred from -o extension if omitted.\n"
        "  -c, --colorspace <cs>   Output color space (default: acescg). Options:\n"
        "                            acescg, rec709, aces2065-1, acescct,\n"
        "                            slog3-sgamut3cine, srgb, rec709-display\n"
        "  --codec <name>          Video codec (default: prores4444).\n"
        "                            prores422, prores422hq, prores4444, prores4444xq,\n"
        "                            h264, h265, h264_nvenc, h265_nvenc, av1_nvenc,\n"
        "                            dnxhr_hqx, dnxhr_444, cineform\n"
        "  --ten-bit               Encode H.265 / AV1 at 10-bit (Main10).\n"
        "                            Other codecs already use their native bit depth.\n"
        "  --bitrate <Mbps>        H.264/H.265 target bitrate (default: 80)\n"
        "  --fps <num>             Override frame rate (default: estimated from timestamps)\n"
        "  --start <n>             First frame index (default: 0)\n"
        "  --end <n>               End frame index, exclusive (default: all)\n"
        "  --exr-compression <c>   EXR compression (default: zip). Lossless: none, rle, zips,\n"
        "                            zip, piz. Lossy: pxr24, b44, b44a, dwaa, dwab.\n"
        "  --denoise-chroma <n>    Chroma noise reduction strength 0..100 (default: 0).\n"
        "                            MP4 only. Kills color noise without softening detail.\n"
        "  --denoise-luma <n>      Luma noise reduction strength 0..100 (default: 0).\n"
        "                            MP4 only. Edge-preserving; high values can soften detail.\n"
        "  --highlight-recovery    Neutralise clipped highlights (kills magenta-sun artifact).\n"
        "                            Adds a soft shoulder rolloff for display-encoded outputs\n"
        "                            (sRGB / Rec.709 / PQ / HLG); scene-referred outputs only\n"
        "                            get the saturation neutralisation.\n"
        "\n"
        "EXR output: HALF-float, ZIP-compressed, tagged with chromaticities + ocio:colorSpace.\n"
        "MOV output: ProRes via prores_ks (full-range 10-bit), or H.264/H.265 via libx264/libx265.\n"
        "Pipeline is internal ACEScg; non-native targets are routed through OCIO\n"
        "(built-in studio-config-v4.0.0_aces-v2.0_ocio-v2.5).\n";
}

bool EndsWith(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(),
        [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

bool ParseArgs(int argc, const char* argv[], Args& out) {
    if (argc < 2) return false;
    out.input = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto needValue = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return std::string(argv[++i]);
        };
        if (a == "-o" || a == "--output") out.output = needValue();
        else if (a == "-f" || a == "--format") {
            std::string v = needValue();
            if (v == "exr")        out.format = OutputFormat::Exr;
            else if (v == "mov")   out.format = OutputFormat::Mov;
            else if (v == "mp4")   out.format = OutputFormat::Mp4;
            else if (v == "mcraw") out.format = OutputFormat::Mcraw;
            else throw std::runtime_error("unknown format: " + v);
            out.formatExplicit = true;
        }
        else if (a == "-c" || a == "--colorspace") {
            std::string v = needValue();
            if (!motioncam::color::ParseOutputColorSpace(v, out.colorSpace))
                throw std::runtime_error("unknown colorspace: " + v);
        }
        else if (a == "--codec") {
            std::string v = needValue();
            if (!motioncam::video::ParseCodec(v, out.codec))
                throw std::runtime_error("unknown codec: " + v);
        }
        else if (a == "--bitrate") out.bitrateMbps = std::stoi(needValue());
        else if (a == "--fps") out.fpsOverride = std::stod(needValue());
        else if (a == "--start") out.startFrame = std::stoi(needValue());
        else if (a == "--end") out.endFrame = std::stoi(needValue());
        else if (a == "--exr-compression") {
            std::string v = needValue();
            if (!motioncam::color::ParseExrCompression(v, out.exrCompression))
                throw std::runtime_error("unknown exr compression: " + v);
        }
        else if (a == "--denoise-chroma") out.denoiseChroma = std::stoi(needValue());
        else if (a == "--denoise-luma")   out.denoiseLuma   = std::stoi(needValue());
        else if (a == "--ten-bit" || a == "--10-bit") out.tenBit = true;
        else if (a == "--highlight-recovery" || a == "--recover-highlights") out.highlightRecovery = true;
        else if (a == "-h" || a == "--help") return false;
        else throw std::runtime_error("unknown flag: " + a);
    }

    if (out.output.empty()) {
        out.output = "frames";
    }
    if (!out.formatExplicit) {
        if (EndsWith(out.output, ".mov"))        out.format = OutputFormat::Mov;
        else if (EndsWith(out.output, ".mp4"))   out.format = OutputFormat::Mp4;
        else if (EndsWith(out.output, ".mcraw")) out.format = OutputFormat::Mcraw;
        else                                      out.format = OutputFormat::Exr;
    }
    return true;
}

void EstimateFps(const std::vector<motioncam::Timestamp>& frames, int& fpsNum, int& fpsDen) {
    fpsNum = 30; fpsDen = 1;
    if (frames.size() < 2) return;
    const int64_t totalNs = frames.back() - frames.front();
    if (totalNs <= 0) return;
    const double avgPeriodNs = double(totalNs) / double(frames.size() - 1);
    const double fps = 1.0e9 / avgPeriodNs;
    // Snap to common fractional rates
    auto near = [&](double target) { return std::abs(fps - target) < 0.05; };
    if      (near(23.976)) { fpsNum = 24000; fpsDen = 1001; }
    else if (near(29.97))  { fpsNum = 30000; fpsDen = 1001; }
    else if (near(59.94))  { fpsNum = 60000; fpsDen = 1001; }
    else                    { fpsNum = int(fps + 0.5); fpsDen = 1; }
}

int RunExr(motioncam::Decoder& decoder, const Args& args, int start, int end) {
    (void)decoder;  // workers open their own Decoder instances for thread-safety
    const auto& csInfo = motioncam::color::GetInfo(args.colorSpace);

    const int total = end - start;
    std::cout << "EXR sequence: " << total << " frames as "
              << csInfo.shortName << " (" << csInfo.ocioName << ")"
              << (csInfo.requiresOcio
                    ? (motioncam::color::HasBakedTransform(args.colorSpace)
                        ? " [baked fast-path]"
                        : " [via OCIO]")
                    : "")
              << "\n";

    std::filesystem::path outDir(args.output);
    std::filesystem::create_directories(outDir);

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned nWorkers = std::min(hw, 8u);

    std::atomic<int> nextFrame{start};
    std::atomic<int> completed{0};
    std::atomic<bool> failed{false};
    std::mutex errMtx;
    std::exception_ptr workerErr;

    auto workerFn = [&]() {
        try {
            motioncam::Decoder dec(args.input);
            const auto& localFrames = dec.getFrames();
            const auto& cmeta = dec.getContainerMetadata();
            motioncam::color::OutputTransform xform;
            xform.Init(args.colorSpace);
            std::vector<uint8_t> rb;
            nlohmann::json fm;
            std::vector<float> rgb;

            while (!failed.load()) {
                const int i = nextFrame.fetch_add(1);
                if (i >= end) break;
                dec.loadFrame(localFrames[i], rb, fm);
                auto p = motioncam::color::BuildFrameParams(fm, cmeta);
                const uint16_t* raw = reinterpret_cast<const uint16_t*>(rb.data());
                motioncam::color::ProcessFrame(raw, p, args.colorSpace, rgb, args.highlightRecovery);
                xform.Apply(rgb.data(), p.width, p.height);
                if (args.highlightRecovery && motioncam::color::IsDisplayEncoded(args.colorSpace)) {
                    motioncam::color::HighlightRolloff(rgb.data(), p.width, p.height);
                }

                char name[64];
                std::snprintf(name, sizeof(name), "frame_%06d.exr", i);
                auto outPath = outDir / name;
                motioncam::color::WriteExrHalfRgb(
                    outPath.string(), rgb.data(), p.width, p.height,
                    csInfo.ocioName, csInfo.chromaticities,
                    args.exrCompression);

                const int done = completed.fetch_add(1) + 1;
                if (done % 10 == 0 || done == total) {
                    std::cout << "Wrote " << done << "/" << total << " frames\r" << std::flush;
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

    std::cout << "\n";
    if (workerErr) std::rethrow_exception(workerErr);
    return 0;
}

int RunMov(motioncam::Decoder& decoder, const Args& args, int start, int end) {
    const auto& containerMeta = decoder.getContainerMetadata();
    const auto& frames = decoder.getFrames();
    const auto& csInfo = motioncam::color::GetInfo(args.colorSpace);

    motioncam::color::OutputTransform xform;
    xform.Init(args.colorSpace);

    // Probe first frame for dimensions
    std::vector<uint8_t> rawBuf;
    nlohmann::json frameMeta;
    decoder.loadFrame(frames[start], rawBuf, frameMeta);
    auto params0 = motioncam::color::BuildFrameParams(frameMeta, containerMeta);
    const int width = static_cast<int>(params0.width);
    const int height = static_cast<int>(params0.height);

    motioncam::video::EncodeSettings es{};
    es.outputPath = args.output;
    es.codec = args.codec;
    es.width = width;
    es.height = height;
    if (args.fpsOverride > 0.0) {
        es.fpsNum = int(args.fpsOverride + 0.5); es.fpsDen = 1;
    } else {
        EstimateFps(frames, es.fpsNum, es.fpsDen);
    }
    es.bitrateMbps = args.bitrateMbps;
    es.audioSampleRate = decoder.audioSampleRateHz();
    es.audioChannels = decoder.numAudioChannels();
    es.containerFormat = ContainerName(args.format);
    es.tenBit = args.tenBit;
    es.colorPrimaries = csInfo.qtPrimaries;
    es.colorTrc       = csInfo.qtTransfer;
    es.colorMatrix    = csInfo.qtMatrix;

    std::cout << "MOV: " << motioncam::video::CodecName(args.codec) << " "
              << width << "x" << height << " @ " << es.fpsNum << "/" << es.fpsDen << "fps, "
              << csInfo.shortName << " (" << csInfo.ocioName << ")"
              << (csInfo.requiresOcio
                    ? (xform.isBaked() ? " [baked fast-path]" : " [via OCIO]")
                    : "")
              << "\n";

    motioncam::video::MovEncoder enc(es);

    // Producer-consumer pipeline: producer thread runs decode + color
    // (loadFrame, ProcessFrame, OCIO) so it overlaps with the encoder
    // running on the main thread. ~30-50% throughput win on multi-core.
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

    // Denoise applies to MP4 deliverables only — MOV stays clean for NLE
    // workflows. Skip the call entirely when both strengths are 0.
    const bool wantDenoise = (args.format == OutputFormat::Mp4) &&
                             (args.denoiseChroma > 0 || args.denoiseLuma > 0);

    // Process the already-probed first frame and queue it.
    {
        ProcessedFrame first;
        first.width = params0.width;
        first.height = params0.height;
        const uint16_t* rawU16 = reinterpret_cast<const uint16_t*>(rawBuf.data());
        motioncam::color::ProcessFrame(rawU16, params0, args.colorSpace, first.rgb, args.highlightRecovery);
        xform.Apply(first.rgb.data(), first.width, first.height);
        if (args.highlightRecovery && motioncam::color::IsDisplayEncoded(args.colorSpace)) {
            motioncam::color::HighlightRolloff(first.rgb.data(), first.width, first.height);
        }
        if (wantDenoise) {
            motioncam::color::DenoiseRgb(
                first.rgb.data(), first.width, first.height,
                args.denoiseChroma, args.denoiseLuma);
        }
        queue.push_back(std::move(first));
    }

    std::thread producer([&]() {
        try {
            std::vector<uint8_t> rb;
            nlohmann::json fm;
            for (int i = start + 1; i < end; ++i) {
                if (cancel.load()) break;
                decoder.loadFrame(frames[i], rb, fm);
                auto p = motioncam::color::BuildFrameParams(fm, containerMeta);
                ProcessedFrame buf;
                buf.width = p.width;
                buf.height = p.height;
                const uint16_t* raw = reinterpret_cast<const uint16_t*>(rb.data());
                motioncam::color::ProcessFrame(raw, p, args.colorSpace, buf.rgb, args.highlightRecovery);
                xform.Apply(buf.rgb.data(), buf.width, buf.height);
                if (args.highlightRecovery && motioncam::color::IsDisplayEncoded(args.colorSpace)) {
                    motioncam::color::HighlightRolloff(buf.rgb.data(), buf.width, buf.height);
                }
                if (wantDenoise) {
                    motioncam::color::DenoiseRgb(
                        buf.rgb.data(), buf.width, buf.height,
                        args.denoiseChroma, args.denoiseLuma);
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
    const int totalToRender = end - start;
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
            if (written % 24 == 0) {
                std::cout << "Encoded frame " << written << "/" << totalToRender << "\r" << std::flush;
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
    std::cout << "\n";

    if (es.audioSampleRate > 0 && es.audioChannels > 0) {
        std::vector<motioncam::AudioChunk> audioChunks;
        decoder.loadAudio(audioChunks);
        std::vector<int16_t> allSamples;
        for (auto& chunk : audioChunks) {
            allSamples.insert(allSamples.end(), chunk.second.begin(), chunk.second.end());
        }
        // Trim audio to match the rendered video range. Assumes audio starts at the
        // same wall-clock time as frame 0 (true for typical MotionCam recordings).
        if (!allSamples.empty() && (start > 0 || end < int(frames.size())) && es.fpsNum > 0) {
            const double framePeriodSec = double(es.fpsDen) / double(es.fpsNum);
            const size_t skip = size_t(start * framePeriodSec * es.audioSampleRate) * es.audioChannels;
            const size_t keep = size_t((end - start) * framePeriodSec * es.audioSampleRate) * es.audioChannels;
            if (skip >= allSamples.size()) {
                allSamples.clear();
            } else {
                allSamples.erase(allSamples.begin(), allSamples.begin() + skip);
                if (allSamples.size() > keep) allSamples.resize(keep);
            }
        }
        if (!allSamples.empty()) {
            std::cout << "Muxing " << allSamples.size() / es.audioChannels
                      << " audio samples (" << es.audioChannels << " ch)\n";
            enc.WriteAudio(allSamples.data(), static_cast<int>(allSamples.size()));
        }
    }

    enc.Finalize();
    std::cout << "Wrote " << args.output << "\n";
    return 0;
}

}

int main(int argc, const char* argv[]) {
    Args args;
    try {
        if (!ParseArgs(argc, argv, args)) { PrintUsage(); return 1; }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        PrintUsage();
        return 1;
    }

    try {
        // MCRAW trim — bypasses the color pipeline entirely; just byte-copies
        // frames + audio in the requested range to a new MCRAW.
        if (args.format == OutputFormat::Mcraw) {
            // Probe just to find totalFrames — Decoder's lightweight init is fine.
            int totalFrames;
            {
                motioncam::Decoder probe(args.input);
                totalFrames = static_cast<int>(probe.getFrames().size());
            }
            const int end = args.endFrame < 0 ? totalFrames : std::min(totalFrames, args.endFrame);
            const int start = std::max(0, std::min(end, args.startFrame));
            std::cout << "MCRAW trim: " << totalFrames << " frames -> ["
                      << start << " .. " << end << ")\n";
            motioncam::TrimMcraw(args.input, args.output, start, end,
                [&](int cur, int tot) {
                    std::cout << "Trimmed " << cur << "/" << tot << " frames\r" << std::flush;
                });
            std::cout << "\nWrote " << args.output << "\n";
            return 0;
        }

        motioncam::Decoder decoder(args.input);
        const auto& frames = decoder.getFrames();
        const int totalFrames = static_cast<int>(frames.size());
        const int end = args.endFrame < 0 ? totalFrames : std::min(totalFrames, args.endFrame);
        const int start = std::max(0, std::min(end, args.startFrame));

        std::cout << "Found " << totalFrames << " frames; rendering ["
                  << start << " .. " << end << ")\n";

        if (IsVideoFormat(args.format)) {
            return RunMov(decoder, args, start, end);
        } else {
            return RunExr(decoder, args, start, end);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
