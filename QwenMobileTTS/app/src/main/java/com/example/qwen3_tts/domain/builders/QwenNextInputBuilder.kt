package com.example.qwen3_tts.domain.builders

import com.example.qwen3_tts.config.QwenConfigLoader
import com.example.qwen3_tts.data.repository.QwenEmbeddingRepository

class QwenNextInputBuilder(
    private val tag: String = "Qwen3TTS"
) {
    fun buildFromCodes(
        codes16: List<Int>,
        timestepIndex: Int,
        prefill: QwenPrefillBuilder.PrefillBuildResult,
        config: QwenConfigLoader.FullConfig,
        embeddings: QwenEmbeddingRepository
    ): FloatArray {
        require(codes16.size == 16) { "Expected 16 codebooks, got ${codes16.size}" }

        val hiddenSize = config.talker.hiddenSize
        val out = FloatArray(hiddenSize)

        val g0Embed = embeddings.talkerCodecEmbeddingRow(codes16[0])
        accumulate(out, g0Embed)

        for (i in 1 until 16) {
            val cpEmbed = embeddings.cpCodecEmbeddingRow(i - 1, codes16[i])
            accumulate(out, cpEmbed)
        }

        val trailing = selectTrailingOrPad(
            timestepIndex = timestepIndex,
            prefill = prefill,
            config = config,
            embeddings = embeddings
        )
        accumulate(out, trailing)
        return out
    }

    private fun selectTrailingOrPad(
        timestepIndex: Int,
        prefill: QwenPrefillBuilder.PrefillBuildResult,
        config: QwenConfigLoader.FullConfig,
        embeddings: QwenEmbeddingRepository
    ): FloatArray {
        val hiddenSize = prefill.hiddenSize
        val trailingSeqLen = prefill.trailingSeqLen

        if (timestepIndex < trailingSeqLen) {
            val offset = timestepIndex * hiddenSize
            return prefill.trailingTextHidden.copyOfRange(offset, offset + hiddenSize)
        }

        val ttsPad2048 = embeddings.textEmbeddingRow(config.tts.ttsPadTokenId)
        return embeddings.textProjection(ttsPad2048)
    }

    private fun accumulate(dst: FloatArray, src: FloatArray) {
        require(dst.size == src.size) { "Vector size mismatch: ${dst.size} vs ${src.size}" }
        for (i in dst.indices) {
            dst[i] += src[i]
        }
    }
}