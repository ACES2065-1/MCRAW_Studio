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

bool EndsWithExt(const std::string& s, const std::string& ext) {
    if (s.size() < ext.size()) return false;
    auto tail = s.substr(s.size() - ext.size());
    for (auto& c : tail) c = char(std::tolower(c));
    return tail == ext;
}
bool EndsWithMov(const std::string& s) { return EndsWithExt(s, ".mov"); }
bool EndsWithMp4(const std::string& s) { return EndsWithExt(s, ".mp4"); }

void EstimateFps(const std::vector<mc::Timestamp>& frames, int& fpsNum, int& fpsDen) {
    fpsNum = 30; fpsDen = 1;
    if (frames.size() < 2) return;
    const int64_t totalNs = frames.back() - frames.front();
    if (totalNs <= 0) return;
    const double avgPeriodNs = double(totalNs) / double(frames.size() - 1);
    const double fps = 1.0e9 / avgPeriodNs;
    auto near = [&](double t) { return std::abs(fps - t) < 0.05; };
    if      (near(23.976)) { fpsNum = 24000; fpsDen = 1001; }
    else if (near(29.97))  { fpsNum = 30000; fpsDen = 1001; }
    else if (near(59.94))  { fpsNum = 60000; fpsDen = 1001; }
    else                    { fpsNum = int(fps + 0.5); fpsDen = 1; }
}

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
                                     bool highlight_recovery) {
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
            mcc::ProcessFrame(raw, params, cs, rgbOut, highlight_recovery);

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
                                  bool highlight_recovery) {
        mcc::OutputColorSpace cs;
        if (!mcc::ParseOutputColorSpace(colorspace, cs))
            throw std::runtime_error("unknown color space: " + colorspace);

        std::vector<uint8_t> rawBuf;
        nlohmann::json frameMeta;
        std::vector<float> rgbOut;
        uint32_t width = 0, height = 0;
        std::string outBytes;
        {
            py::gil_scoped_release release;
            d_->loadFrame(timestamp, rawBuf, frameMeta);
            auto params = mcc::BuildFrameParams(frameMeta, d_->getContainerMetadata());
            width = params.width;
            height = params.height;
            const uint16_t* raw = reinterpret_cast<const uint16_t*>(rawBuf.data());
            mcc::ProcessFrame(raw, params, cs, rgbOut, highlight_recovery);

            EnsureXform(cs);
            cachedXform_.Apply(rgbOut.data(), width, height);
            if (highlight_recovery && mcc::IsDisplayEncoded(cs)) {
                mcc::HighlightRolloff(rgbOut.data(), width, height);
            }

            // Float [0,1] -> uint8 [0,255], packed RGB888.
            outBytes.resize(size_t(width) * size_t(height) * 3);
            uint8_t* dst = reinterpret_cast<uint8_t*>(outBytes.data());
            const float* src = rgbOut.data();
            const size_t n = rgbOut.size();
            for (size_t i = 0; i < n; ++i) {
                float v = src[i];
                if (v < 0.0f) v = 0.0f;
                else if (v > 1.0f) v = 1.0f;
                dst[i] = uint8_t(v * 255.0f + 0.5f);
            }
        }

        return py::make_tuple(py::bytes(outBytes), int(height), int(width));
    }

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
    bool highlight_recovery)
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

        mcv::EncodeSettings es{};
        es.outputPath = output;
        es.codec = vcodec;
        es.width = int(params0.width);
        es.height = int(params0.height);
        if (fps_arg > 0.0) {
            es.fpsNum = int(fps_arg + 0.5); es.fpsDen = 1;
        } else {
            EstimateFps(frames, es.fpsNum, es.fpsDen);
        }
        es.bitrateMbps = bitrate;
        es.audioSampleRate = decoder.audioSampleRateHz();
        es.audioChannels = decoder.numAudioChannels();
        es.containerFormat = EndsWithMp4(output) ? "mp4" : "mov";
        es.tenBit = ten_bit;
        es.colorPrimaries = csInfo.qtPrimaries;
        es.colorTrc       = csInfo.qtTransfer;
        es.colorMatrix    = csInfo.qtMatrix;

        mcv::MovEncoder enc(es);
        const int total_to_render = e - s;

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

        // Denoise: MP4 only, and only when at least one strength is > 0.
        const bool wantDenoise = EndsWithMp4(output)
            && (denoise_chroma > 0 || denoise_luma > 0);

        // Process the already-probed first frame and queue it.
        {
            ProcessedFrame first;
            first.width = params0.width;
            first.height = params0.height;
            const uint16_t* raw = reinterpret_cast<const uint16_t*>(rawBuf.data());
            mcc::ProcessFrame(raw, params0, cs, first.rgb, highlight_recovery);
            xform.Apply(first.rgb.data(), first.width, first.height);
            if (highlight_recovery && mcc::IsDisplayEncoded(cs)) {
                mcc::HighlightRolloff(first.rgb.data(), first.width, first.height);
            }
            if (wantDenoise) {
                mcc::DenoiseRgb(first.rgb.data(), first.width, first.height,
                                denoise_chroma, denoise_luma);
            }
            queue.push_back(std::move(first));
        }

        std::thread producer([&]() {
            try {
                std::vector<uint8_t> rb;
                nlohmann::json fm;
                for (int i = s + 1; i < e; ++i) {
                    if (cancel.load()) break;
                    if (cancel_check()) { cancel.store(true); not_empty.notify_all(); break; }
                    decoder.loadFrame(frames[i], rb, fm);
                    auto p = mcc::BuildFrameParams(fm, containerMeta);
                    ProcessedFrame buf;
                    buf.width = p.width;
                    buf.height = p.height;
                    const uint16_t* raw = reinterpret_cast<const uint16_t*>(rb.data());
                    mcc::ProcessFrame(raw, p, cs, buf.rgb, highlight_recovery);
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

        if (es.audioSampleRate > 0 && es.audioChannels > 0) {
            std::vector<mc::AudioChunk> chunks;
            decoder.loadAudio(chunks);
            std::vector<int16_t> all;
            for (auto& c : chunks) all.insert(all.end(), c.second.begin(), c.second.end());
            // Trim audio to match the rendered video range (assumes audio starts with frame 0).
            if (!all.empty() && (s > 0 || e < totalFrames) && es.fpsNum > 0) {
                const double framePeriodSec = double(es.fpsDen) / double(es.fpsNum);
                const size_t skip = size_t(s * framePeriodSec * es.audioSampleRate) * es.audioChannels;
                const size_t keep = size_t((e - s) * framePeriodSec * es.audioSampleRate) * es.audioChannels;
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
        .def("process_frame", &PyDecoder::process_frame,
             py::arg("timestamp"), py::arg("colorspace") = "acescg",
             py::arg("highlight_recovery") = false,
             "Returns float32 numpy (H, W, 3) RGB in the requested color space.")
        .def("process_frame_rgb24", &PyDecoder::process_frame_rgb24,
             py::arg("timestamp"), py::arg("colorspace") = "srgb",
             py::arg("highlight_recovery") = false,
             "Returns (RGB888 packed bytes, height, width). No numpy required.");

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
        R"doc(Render an MCRAW file end-to-end.

If `output` ends in '.mov', encodes a QuickTime file (codec defaults to prores4444).
Otherwise treats `output` as a directory and writes an EXR sequence.

colorspace: acescg (default), rec709, aces2065-1, acescct,
            slog3-sgamut3cine, srgb, rec709-display
codec (mov only): prores422, prores422hq, prores4444, prores4444xq, h264, h265
)doc");

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
           const std::string& target_colorspace) -> py::array_t<float> {
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
        "Phase C.1 GPU bayer pipeline (normalise + debayer + cam-to-output "
        "matrix + optional curve). Returns float32 (H, W, 3) RGB in the "
        "target colour space. For correctness testing against process_frame.");
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
