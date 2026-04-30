package com.example.qwen3_tts.domain.pipeline

import android.content.Context
import android.content.res.AssetManager
import java.io.Closeable
import java.io.File
import com.example.qwen3_tts.tokenizer.QwenTextTokenizer
import com.example.qwen3_tts.config.QwenConfigLoader
import com.example.qwen3_tts.data.repository.*
import com.example.qwen3_tts.inference.runners.*
import com.example.qwen3_tts.audio.WavFileWriter

class QwenTtsPipeline(
    private val context: Context,
    private val tag: String = "Qwen3TTS"
) : Closeable {

    data class PipelineResult(
        val outputWav: File,
        val finalWaveform: FloatArray,
        val chunks: List<String>,
        val totalDurationSec: Double,
        val selectedSpeakerName: String?,
        val selectedSpeakerId: Int
    )

    data class ProgressUpdate(
        val percent: Int,
        val message: String
    )

    private var cachedRepo: QwenModelRepository? = null
    private var cachedConfig: QwenConfigLoader.FullConfig? = null
    private var cachedTokenizer: QwenTextTokenizer? = null
    private var cachedSpeakerCatalog: SpeakerCatalogLoader.SpeakerCatalog? = null
    // Non-null when models are loaded from APK assets; null when loaded from filesDir (ZIP import).
    private var cachedAssetManager: AssetManager? = null

    private fun ensureInitialized() {
        if (cachedRepo != null && cachedConfig != null &&
            cachedTokenizer != null && cachedSpeakerCatalog != null
        ) return

        val repo = QwenModelRepository(context)

        if (repo.isAssetBundlePresent(context.assets)) {
            initFromAssets(repo, context.assets)
        } else {
            initFromFiles(repo)
        }
    }

    private fun initFromAssets(repo: QwenModelRepository, assets: AssetManager) {
        val configLoader = QwenConfigLoader(tag)

        val config = assets.open(repo.getEmbeddingAssetPath("config.json")).use {
            configLoader.load(it)
        }
        val tokenizer = QwenTextTokenizer(
            assetManager = assets,
            assetDir = QwenModelRepository.ASSET_TOKENIZER_DIR,
            tag = tag
        )
        val speakerCatalog = assets.open(repo.getEmbeddingAssetPath("speaker_ids.json")).use {
            SpeakerCatalogLoader(tag).load(it)
        }

        cachedRepo = repo
        cachedConfig = config
        cachedTokenizer = tokenizer
        cachedSpeakerCatalog = speakerCatalog
        cachedAssetManager = assets
    }

    private fun initFromFiles(repo: QwenModelRepository) {
        val missing = repo.missingRequiredFiles()
        require(missing.isEmpty()) { "Missing required files: $missing" }

        val config = QwenConfigLoader(tag).load(repo.getEmbeddingFile("config.json"))
        val tokenizer = QwenTextTokenizer(tokenizerDir = repo.getTokenizerDir(), tag = tag)
        val speakerCatalog = SpeakerCatalogLoader(tag).load(repo.getEmbeddingFile("speaker_ids.json"))

        cachedRepo = repo
        cachedConfig = config
        cachedTokenizer = tokenizer
        cachedSpeakerCatalog = speakerCatalog
        cachedAssetManager = null
    }

    fun run(
        rawInputText: String,
        language: String = "auto",
        speakerName: String? = null,
        maxSentencesPerChunk: Int = 2,
        temperature: Float = 1.0f,
        topK: Int = 50,
        repetitionPenalty: Float = 1.1f,
        minNewTokens: Int = 0,
        silenceBetweenChunksMs: Int = 250,
        onProgress: ((ProgressUpdate) -> Unit)? = null
    ): PipelineResult {
        reportProgress(onProgress, 0, "Initializing...")
        ensureInitialized()
        reportProgress(onProgress, 10, "Metadata ready")

        val repo = requireNotNull(cachedRepo)
        val config = requireNotNull(cachedConfig)
        val realTokenizer = requireNotNull(cachedTokenizer)
        val speakerCatalog = requireNotNull(cachedSpeakerCatalog)
        val assets = cachedAssetManager

        val generateCodesRunner = QwenGenerateCodesRunner(tag)
        val vocoderRunner = QwenVocoderRunner(tag)

        val normalizedText = normalizeInputText(rawInputText)
        val chunks = splitIntoSentenceChunks(normalizedText, maxSentencesPerChunk)
        val resolvedSpeakerId = speakerCatalog.idFor(speakerName)

        require(chunks.isNotEmpty()) { "No chunks produced from input text" }

        reportProgress(onProgress, 15, "Text prepared: ${chunks.size} chunk(s)")

        val sampleRate = 24000
        val silenceBetweenChunks = makeSilence(sampleRate, silenceBetweenChunksMs)
        val waveformParts = mutableListOf<FloatArray>()
        val totalChunks = chunks.size

        reportProgress(onProgress, 20, "Loading embeddings...")

        QwenEmbeddingRepository(repo, tag).use { embeddingRepository ->
            if (assets != null) {
                embeddingRepository.loadRequiredCoreEmbeddingsFromAssets(assets)
            } else {
                embeddingRepository.loadRequiredCoreEmbeddings()
            }

            for ((chunkIndex, inputText) in chunks.withIndex()) {
                val chunkBaseProgress = 20 + ((chunkIndex * 55) / totalChunks)

                reportProgress(
                    onProgress,
                    chunkBaseProgress,
                    "Generating chunk ${chunkIndex + 1}/$totalChunks"
                )

                val pipelineTokenIds = realTokenizer.buildCustomVoicePromptIds(
                    text = inputText,
                    instruct = null
                )

                if (pipelineTokenIds.isEmpty()) {
                    continue
                }

                reportProgress(
                    onProgress,
                    chunkBaseProgress + 5,
                    "Tokenized chunk ${chunkIndex + 1}/$totalChunks"
                )

                val estimatedMaxSteps = estimateMaxTimeSteps(pipelineTokenIds.size)

                val generateCodesResult = generateCodesRunner.run(
                    repo = repo,
                    config = config,
                    tokenIds = pipelineTokenIds,
                    embeddings = embeddingRepository,
                    language = language,
                    speakerId = resolvedSpeakerId,
                    maxNewTokens = estimatedMaxSteps,
                    temperature = temperature,
                    topK = topK,
                    repetitionPenalty = repetitionPenalty,
                    assetManager = assets
                )

                reportProgress(
                    onProgress,
                    chunkBaseProgress + 20,
                    "Codes generated for chunk ${chunkIndex + 1}/$totalChunks"
                )

                if (generateCodesResult.timeSteps == 0 || generateCodesResult.codes.isEmpty()) {
                    continue
                }

                reportProgress(
                    onProgress,
                    chunkBaseProgress + 30,
                    "Running vocoder for chunk ${chunkIndex + 1}/$totalChunks"
                )

                val vocoderInput = QwenVocoderRunner.VocoderInput(
                    codes = generateCodesResult.codes,
                    timeSteps = generateCodesResult.timeSteps,
                    codebooks = generateCodesResult.codebooks
                )

                val vocoderResult = if (assets != null) {
                    vocoderRunner.run(
                        assetManager = assets,
                        assetPath = repo.getModelAssetPath("vocoder.onnx"),
                        input = vocoderInput
                    )
                } else {
                    vocoderRunner.run(
                        modelFile = repo.getModelFile("vocoder.onnx"),
                        input = vocoderInput
                    )
                }

                val normalizedChunkWaveform = normalizeWaveform(
                    waveform = vocoderResult.waveform,
                    targetPeak = 0.8f
                )

                waveformParts.add(normalizedChunkWaveform)

                val chunkDoneProgress = 20 + (((chunkIndex + 1) * 55) / totalChunks)
                reportProgress(
                    onProgress,
                    chunkDoneProgress,
                    "Finished chunk ${chunkIndex + 1}/$totalChunks"
                )

                if (chunkIndex < chunks.lastIndex) {
                    waveformParts.add(silenceBetweenChunks)
                }
            }
        }

        require(waveformParts.isNotEmpty()) { "No waveform parts were generated" }

        reportProgress(onProgress, 85, "Combining waveform...")

        val finalWaveformRaw = concatWaveforms(waveformParts)
        val finalWaveform = normalizeWaveform(finalWaveformRaw, targetPeak = 0.8f)
        val finalApproxDurationSec = finalWaveform.size / sampleRate.toDouble()
        val outputWav = File(context.filesDir, "qwen3_tts/output_full.wav")

        reportProgress(onProgress, 95, "Saving WAV file...")

        WavFileWriter.writeMono16BitPcm(
            file = outputWav,
            waveform = finalWaveform,
            sampleRate = sampleRate
        )

        reportProgress(onProgress, 100, "Done")

        return PipelineResult(
            outputWav = outputWav,
            finalWaveform = finalWaveform,
            chunks = chunks,
            totalDurationSec = finalApproxDurationSec,
            selectedSpeakerName = speakerName,
            selectedSpeakerId = resolvedSpeakerId
        )
    }

    fun loadAvailableSpeakers(): List<String> {
        val repo = QwenModelRepository(context)
        return if (repo.isAssetBundlePresent(context.assets)) {
            context.assets.open(repo.getEmbeddingAssetPath("speaker_ids.json")).use {
                SpeakerCatalogLoader(tag).load(it).names()
            }
        } else {
            SpeakerCatalogLoader(tag).load(repo.getEmbeddingFile("speaker_ids.json")).names()
        }
    }

    private fun reportProgress(
        onProgress: ((ProgressUpdate) -> Unit)?,
        percent: Int,
        message: String
    ) {
        onProgress?.invoke(
            ProgressUpdate(
                percent = percent.coerceIn(0, 100),
                message = message
            )
        )
    }

    private fun normalizeInputText(text: String): String {
        return text
            .replace("\n", " ")
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    private fun estimateMaxTimeSteps(tokenCount: Int): Int {
        val estimated = tokenCount * 6
        return estimated.coerceIn(24, 192)
    }

    private fun splitIntoSentenceChunks(
        text: String,
        maxSentencesPerChunk: Int = 2
    ): List<String> {
        val sentences = text
            .split(Regex("(?<=[.!?])\\s+"))
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        if (sentences.isEmpty()) return emptyList()

        val chunks = mutableListOf<String>()
        var i = 0
        while (i < sentences.size) {
            val chunk = sentences
                .subList(i, minOf(i + maxSentencesPerChunk, sentences.size))
                .joinToString(" ")
            chunks.add(chunk)
            i += maxSentencesPerChunk
        }
        return chunks
    }

    private fun makeSilence(sampleRate: Int, durationMs: Int): FloatArray {
        val samples = (sampleRate * durationMs) / 1000
        return FloatArray(samples)
    }

    private fun concatWaveforms(parts: List<FloatArray>): FloatArray {
        val total = parts.sumOf { it.size }
        val out = FloatArray(total)
        var offset = 0
        for (part in parts) {
            part.copyInto(out, destinationOffset = offset)
            offset += part.size
        }
        return out
    }

    private fun normalizeWaveform(
        waveform: FloatArray,
        targetPeak: Float = 0.8f
    ): FloatArray {
        if (waveform.isEmpty()) return waveform

        var absMax = 0f
        for (v in waveform) {
            val a = kotlin.math.abs(v)
            if (a > absMax) absMax = a
        }

        if (absMax <= 1e-8f) {
            return waveform.copyOf()
        }

        val scale = targetPeak / absMax
        val out = FloatArray(waveform.size)
        for (i in waveform.indices) {
            out[i] = waveform[i] * scale
        }
        return out
    }

    override fun close() {
        cachedRepo = null
        cachedConfig = null
        cachedTokenizer = null
        cachedSpeakerCatalog = null
        cachedAssetManager = null
    }
}
