#include <motioncam/Trimmer.hpp>
#include <motioncam/Decoder.hpp>
#include <motioncam/Container.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
    #define FSEEK _fseeki64
    #define FTELL _ftelli64
#else
    #define _FILE_OFFSET_BITS 64
    #define FSEEK fseeko
    #define FTELL ftello
#endif

namespace motioncam {

namespace {

// Tiny RAII wrapper so we close the output file even on exceptions.
struct FileCloser {
    FILE* f;
    ~FileCloser() { if (f) std::fclose(f); }
};

template<typename T>
void Write(FILE* f, const T& v) {
    if (std::fwrite(&v, sizeof(T), 1, f) != 1)
        throw IOException("write failed");
}

void WriteBytes(FILE* f, const void* data, size_t n) {
    if (n == 0) return;
    if (std::fwrite(data, 1, n, f) != n)
        throw IOException("write failed");
}

int64_t Tell(FILE* f) {
    int64_t pos = FTELL(f);
    if (pos < 0) throw IOException("ftell failed");
    return pos;
}

}  // namespace

void TrimMcraw(
    const std::string& inputPath,
    const std::string& outputPath,
    int startFrame,
    int endFrame,
    std::function<void(int, int)> progress,
    std::function<bool()> cancel)
{
    Decoder decoder(inputPath);
    const auto& frames = decoder.getFrames();
    const auto& containerMeta = decoder.getContainerMetadata();

    const int total = static_cast<int>(frames.size());
    if (startFrame < 0)        startFrame = 0;
    if (endFrame > total)      endFrame   = total;
    if (startFrame >= endFrame)
        throw IOException("trim: empty range (start=" + std::to_string(startFrame)
                          + ", end=" + std::to_string(endFrame) + ")");

    // Compute the time window we want to preserve audio for. We extend it by
    // a small grace on each side so chunks that straddle the boundary aren't
    // lost (audio is recorded in fixed-size pieces, ~tens of ms each).
    const Timestamp tsStart = frames[startFrame];
    const Timestamp tsEnd   = frames[endFrame - 1];
    const Timestamp graceNs = static_cast<Timestamp>(200'000'000);  // 200 ms
    const Timestamp keepFromTs = tsStart - graceNs;
    const Timestamp keepUntilTs = tsEnd + graceNs;

    FILE* out = std::fopen(outputPath.c_str(), "wb");
    if (!out) throw IOException("trim: failed to open output: " + outputPath);
    FileCloser closer{out};

    // 1. Header
    Header header{};
    std::memcpy(header.ident, CONTAINER_ID, sizeof(CONTAINER_ID));
    header.version = CONTAINER_VERSION;
    Write(out, header);

    // 2. Camera metadata (preserved as-is from the source).
    {
        const std::string metaJson = containerMeta.dump();
        Item item{ Type::METADATA, static_cast<uint32_t>(metaJson.size()) };
        Write(out, item);
        WriteBytes(out, metaJson.data(), metaJson.size());
    }

    // 3. For each frame in [start, end): write BUFFER + frame METADATA.
    //    We copy the compressed bayer bytes verbatim and the frame metadata
    //    JSON byte-for-byte — no re-encoding, no risk of altering pixels.
    std::vector<BufferOffset> newFrameOffsets;
    newFrameOffsets.reserve(endFrame - startFrame);

    std::vector<uint8_t> compressed;
    std::string frameMetaJson;
    const int trimCount = endFrame - startFrame;

    bool aborted = false;
    for (int i = startFrame; i < endFrame; ++i) {
        if (cancel && cancel()) {
            aborted = true;
            break;
        }

        const Timestamp ts = frames[i];
        decoder.loadFrameRaw(ts, compressed, frameMetaJson);

        const int64_t bufferItemPos = Tell(out);

        Item bufferItem{ Type::BUFFER, static_cast<uint32_t>(compressed.size()) };
        Write(out, bufferItem);
        WriteBytes(out, compressed.data(), compressed.size());

        Item metaItem{ Type::METADATA, static_cast<uint32_t>(frameMetaJson.size()) };
        Write(out, metaItem);
        WriteBytes(out, frameMetaJson.data(), frameMetaJson.size());

        newFrameOffsets.push_back({ bufferItemPos, ts });

        if (progress && ((i - startFrame) % 10 == 0 || i == endFrame - 1)) {
            progress(i - startFrame + 1, trimCount);
        }
    }

    // If cancelled with no frames written we still emit a valid (empty) MCRAW
    // so the file isn't left in a half-written state. Caller can delete it.
    if (aborted && newFrameOffsets.empty()) {
        // Skip audio + index — just close with an empty frame index.
    }

    // 4. Audio. Two paths:
    //   A) Source has per-chunk timestamps (newer format) — copy chunks whose
    //      timestamp falls inside the trim window.
    //   B) Source has no per-chunk timestamps (older format, all timestamps
    //      come back as -1) — concatenate all audio, slice the trim window
    //      based on the frame-timestamp-relative position within the clip,
    //      and write the slice as a single new chunk.
    std::vector<AudioChunk> allAudio;
    decoder.loadAudio(allAudio);

    std::vector<BufferOffset> newAudioOffsets;
    Timestamp audioStartTsMs = -1;

    bool anyValidTs = false;
    for (const auto& c : allAudio) {
        if (c.first > 0) { anyValidTs = true; break; }
    }

    if (anyValidTs) {
        // Path A: per-chunk filter.
        for (const auto& chunk : allAudio) {
            const Timestamp chunkTs = chunk.first;
            if (chunkTs <= 0)          continue;
            if (chunkTs < keepFromTs)  continue;
            if (chunkTs > keepUntilTs) continue;

            const int64_t audioItemPos = Tell(out);

            const auto& samples = chunk.second;
            const uint32_t bytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
            Item audioItem{ Type::AUDIO_DATA, bytes };
            Write(out, audioItem);
            WriteBytes(out, samples.data(), bytes);

            Item metaItem{ Type::AUDIO_DATA_METADATA, static_cast<uint32_t>(sizeof(AudioMetadata)) };
            Write(out, metaItem);
            AudioMetadata am{ chunkTs };
            Write(out, am);

            newAudioOffsets.push_back({ audioItemPos, chunkTs });
            if (audioStartTsMs < 0)
                audioStartTsMs = chunkTs / 1'000'000;
        }
    } else if (!allAudio.empty()) {
        // Path B: slice from concatenated audio.
        const int channels   = std::max(1, decoder.numAudioChannels());
        const int sampleRate = decoder.audioSampleRateHz();
        if (sampleRate > 0) {
            int64_t totalSamplesAll = 0;
            for (const auto& c : allAudio) totalSamplesAll += int64_t(c.second.size());
            const int64_t totalFramesAll = totalSamplesAll / channels;

            const Timestamp srcStartTs = frames.front();
            const Timestamp srcEndTs   = frames.back();
            const int64_t srcSpanNs    = srcEndTs - srcStartTs;
            const int64_t avgFrameNs   = (total > 1) ? (srcSpanNs / (total - 1)) : 0;

            // Trim window expressed as offset (ns) into the clip.
            const int64_t trimStartRelNs = tsStart - srcStartTs;
            const int64_t trimEndRelNs   = (tsEnd - srcStartTs) + avgFrameNs;

            int64_t startSampleFrame = trimStartRelNs * int64_t(sampleRate) / 1'000'000'000LL;
            int64_t endSampleFrame   = trimEndRelNs   * int64_t(sampleRate) / 1'000'000'000LL;
            if (startSampleFrame < 0)               startSampleFrame = 0;
            if (endSampleFrame   > totalFramesAll)  endSampleFrame   = totalFramesAll;

            if (endSampleFrame > startSampleFrame) {
                // Concatenate source audio (interleaved int16).
                std::vector<int16_t> all;
                all.reserve(static_cast<size_t>(totalSamplesAll));
                for (const auto& c : allAudio) {
                    all.insert(all.end(), c.second.begin(), c.second.end());
                }
                const int64_t startIdx = startSampleFrame * channels;
                const int64_t endIdx   = endSampleFrame   * channels;

                const int64_t audioItemPos = Tell(out);
                const uint32_t bytes = static_cast<uint32_t>((endIdx - startIdx) * int64_t(sizeof(int16_t)));
                Item audioItem{ Type::AUDIO_DATA, bytes };
                Write(out, audioItem);
                WriteBytes(out, all.data() + startIdx, bytes);

                Item metaItem{ Type::AUDIO_DATA_METADATA, static_cast<uint32_t>(sizeof(AudioMetadata)) };
                Write(out, metaItem);
                AudioMetadata am{ tsStart };
                Write(out, am);

                newAudioOffsets.push_back({ audioItemPos, tsStart });
                audioStartTsMs = tsStart / 1'000'000;
            }
        }
    }

    // 5. AUDIO_INDEX (only if we actually wrote audio chunks).
    if (!newAudioOffsets.empty()) {
        const uint32_t audioIndexSize =
            static_cast<uint32_t>(sizeof(AudioIndex)
                                  + newAudioOffsets.size() * sizeof(BufferOffset));
        Item idxItem{ Type::AUDIO_INDEX, audioIndexSize };
        Write(out, idxItem);

        AudioIndex audioIdx{};
        audioIdx.numOffsets       = static_cast<int64_t>(newAudioOffsets.size());
        audioIdx.startTimestampMs = audioStartTsMs >= 0 ? audioStartTsMs : 0;
        Write(out, audioIdx);
        WriteBytes(out, newAudioOffsets.data(),
                   newAudioOffsets.size() * sizeof(BufferOffset));
    }

    // 6. Frame index data — sequence of BufferOffset structs at this position.
    const int64_t indexDataOffset = Tell(out);
    WriteBytes(out, newFrameOffsets.data(),
               newFrameOffsets.size() * sizeof(BufferOffset));

    // 7. Trailer: BUFFER_INDEX item + struct (used by Decoder::readIndex).
    Item indexItem{ Type::BUFFER_INDEX, static_cast<uint32_t>(sizeof(BufferIndex)) };
    Write(out, indexItem);

    BufferIndex bufIdx{};
    bufIdx.magicNumber     = static_cast<int32_t>(INDEX_MAGIC_NUMBER);
    bufIdx.numOffsets      = static_cast<int32_t>(newFrameOffsets.size());
    bufIdx.indexDataOffset = indexDataOffset;
    Write(out, bufIdx);
}

}  // namespace motioncam
