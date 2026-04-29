package com.example.qwen3_tts.domain.builders

import com.example.qwen3_tts.config.QwenConfigLoader
import com.example.qwen3_tts.data.repository.QwenEmbeddingRepository

class QwenPrefillBuilder(
    private val tag: String = "Qwen3TTS"
) {
    data class PrefillBuildResult(
        val inputsEmbeds: FloatArray,
        val inputsSeqLen: Int,
        val hiddenSize: Int,
        val trailingTextHidden: FloatArray,
        val trailingSeqLen: Int
    )

    fun build(
        tokenIds: List<Int>,
        config: QwenConfigLoader.FullConfig,
        embeddings: QwenEmbeddingRepository,
        language: String = "auto",
        speakerId: Int = -1
    ): PrefillBuildResult {
        require(tokenIds.size >= 9) { "tokenIds too short for prompt split: size=${tokenIds.size}" }

        val hiddenSize = config.talker.hiddenSize
        require(hiddenSize == 1024) {
            "Current Android prefill builder expects hiddenSize=1024, got $hiddenSize"
        }

        val ttsPadProj = projectTextToken(config.tts.ttsPadTokenId, embeddings)
        val ttsBosProj = projectTextToken(config.tts.ttsBosTokenId, embeddings)
        val ttsEosProj = projectTextToken(config.tts.ttsEosTokenId, embeddings)

        val roleEmbeds = mutableListOf<FloatArray>()
        for (i in 0 until 3) {
            roleEmbeds.add(projectTextToken(tokenIds[i], embeddings))
        }

        val codecPrefix = buildCodecPrefix(config, language, speakerId)

        val talkerInputEmbeds = mutableListOf<FloatArray>()
        for (i in 0 until codecPrefix.size - 2) {
            val codecEmbed = embeddings.talkerCodecEmbeddingRow(codecPrefix[i])
            talkerInputEmbeds.add(addVectors(ttsPadProj, codecEmbed))
        }

        val lastCodecEmbed = embeddings.talkerCodecEmbeddingRow(codecPrefix[codecPrefix.size - 2])
        talkerInputEmbeds.add(addVectors(ttsBosProj, lastCodecEmbed))

        val firstTextTokenId = tokenIds[3]
        val firstTextProj = projectTextToken(firstTextTokenId, embeddings)
        val codecBosEmbed = embeddings.talkerCodecEmbeddingRow(config.talker.codecBosId)
        val firstTextPlusCodecBos = addVectors(firstTextProj, codecBosEmbed)

        val trailingEmbeds = mutableListOf<FloatArray>()
        for (i in 4 until tokenIds.size - 5) {
            trailingEmbeds.add(projectTextToken(tokenIds[i], embeddings))
        }
        trailingEmbeds.add(ttsEosProj)

        val inputsList = mutableListOf<FloatArray>()
        inputsList.addAll(roleEmbeds)
        inputsList.addAll(talkerInputEmbeds)
        inputsList.add(firstTextPlusCodecBos)

        val inputsSeqLen = inputsList.size
        val trailingSeqLen = trailingEmbeds.size

        val inputsFlat = flattenVectors(inputsList, hiddenSize)
        val trailingFlat = flattenVectors(trailingEmbeds, hiddenSize)

        return PrefillBuildResult(
            inputsEmbeds = inputsFlat,
            inputsSeqLen = inputsSeqLen,
            hiddenSize = hiddenSize,
            trailingTextHidden = trailingFlat,
            trailingSeqLen = trailingSeqLen
        )
    }

    private fun buildCodecPrefix(
        config: QwenConfigLoader.FullConfig,
        language: String,
        speakerId: Int
    ): List<Int> {
        val prefix = mutableListOf<Int>()

        if (language == "auto") {
            prefix.add(config.talker.codecNoThinkId)
            prefix.add(config.talker.codecThinkBosId)
            prefix.add(config.talker.codecThinkEosId)
        } else {
            prefix.add(config.talker.codecThinkId)
            prefix.add(config.talker.codecThinkBosId)
            val langId = config.languageIds[language.lowercase()]
                ?: error("Unknown language: $language")
            prefix.add(langId)
            prefix.add(config.talker.codecThinkEosId)
        }

        if (speakerId >= 0) {
            prefix.add(speakerId)
        }

        prefix.add(config.talker.codecPadId)
        prefix.add(config.talker.codecBosId)

        return prefix
    }

    private fun projectTextToken(
        tokenId: Int,
        embeddings: QwenEmbeddingRepository
    ): FloatArray {
        val emb2048 = embeddings.textEmbeddingRow(tokenId)
        return embeddings.textProjection(emb2048)
    }

    private fun addVectors(a: FloatArray, b: FloatArray): FloatArray {
        require(a.size == b.size) { "Vector size mismatch: ${a.size} vs ${b.size}" }
        val out = FloatArray(a.size)
        for (i in a.indices) {
            out[i] = a[i] + b[i]
        }
        return out
    }

    private fun flattenVectors(vectors: List<FloatArray>, hiddenSize: Int): FloatArray {
        val out = FloatArray(vectors.size * hiddenSize)
        var offset = 0
        for (vec in vectors) {
            require(vec.size == hiddenSize) {
                "Vector hidden size mismatch: got ${vec.size}, expected $hiddenSize"
            }
            vec.copyInto(out, offset)
            offset += hiddenSize
        }
        return out
    }
}