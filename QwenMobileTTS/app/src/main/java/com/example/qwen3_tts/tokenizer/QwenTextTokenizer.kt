package com.example.qwen3_tts.tokenizer

import android.content.res.AssetManager
import java.io.File
import java.util.concurrent.ConcurrentHashMap

private class QwenGpt2PreTokenizer {
    private val regex = Regex(
        pattern = """'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+""",
        option = RegexOption.IGNORE_CASE
    )

    fun preTokenize(text: String): List<String> {
        return regex.findAll(text).map { it.value }.toList()
    }
}

class QwenTextTokenizer private constructor(
    private val resources: QwenBpeResources.Resources,
    private val tag: String = "Qwen3TTS"
) {
    companion object {
        const val IM_START_ID = 151644
        const val IM_END_ID = 151645
        const val ASSISTANT_TOKEN_ID = 77091
        const val NEWLINE_TOKEN_ID = 198

        operator fun invoke(tokenizerDir: File, tag: String = "Qwen3TTS"): QwenTextTokenizer =
            QwenTextTokenizer(QwenBpeResources(tag).load(tokenizerDir), tag)

        operator fun invoke(assetManager: AssetManager, assetDir: String, tag: String = "Qwen3TTS"): QwenTextTokenizer =
            QwenTextTokenizer(QwenBpeResources(tag).load(assetManager, assetDir), tag)
    }

    private val byteEncoder = QwenByteEncoder()
    private val preTokenizer = QwenGpt2PreTokenizer()
    private val bpeCache = ConcurrentHashMap<String, List<String>>()

    fun encode(text: String): List<Int> {
        val pieces = preTokenizer.preTokenize(text)
        val out = mutableListOf<Int>()

        for (piece in pieces) {
            val byteLevel = byteEncoder.encodeToByteLevelPieces(piece)
            val bpeTokens = applyBpe(byteLevel)
            for (token in bpeTokens) {
                val id = resources.vocab[token]
                    ?: error("Token '$token' not found in vocab")
                out.add(id)
            }
        }
        return out
    }

    fun buildCustomVoicePromptIds(
        text: String,
        instruct: String? = null
    ): List<Int> {
        val ids = mutableListOf<Int>()

        if (!instruct.isNullOrEmpty()) {
            ids.add(IM_START_ID)
            ids.addAll(encode("user"))
            ids.add(NEWLINE_TOKEN_ID)
            ids.addAll(encode(instruct))
            ids.add(IM_END_ID)
            ids.add(NEWLINE_TOKEN_ID)
        }

        val textIds = encode(text)

        ids.add(IM_START_ID)
        ids.add(ASSISTANT_TOKEN_ID)
        ids.add(NEWLINE_TOKEN_ID)

        ids.addAll(textIds)

        ids.add(IM_END_ID)
        ids.add(NEWLINE_TOKEN_ID)
        ids.add(IM_START_ID)
        ids.add(ASSISTANT_TOKEN_ID)
        ids.add(NEWLINE_TOKEN_ID)

        return ids
    }

    private fun applyBpe(token: String): List<String> {
        bpeCache[token]?.let { return it }

        if (token.length == 1) {
            return listOf(token).also { bpeCache[token] = it }
        }

        var word = token.map { it.toString() }.toMutableList()

        while (true) {
            val pairs = getPairs(word)
            if (pairs.isEmpty()) break

            val bestPair = pairs.minByOrNull { pair ->
                resources.mergesRank[pair] ?: Int.MAX_VALUE
            } ?: break

            val bestRank = resources.mergesRank[bestPair]
            if (bestRank == null) break

            val first = bestPair.first
            val second = bestPair.second

            val newWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }

            if (newWord == word) break
            word = newWord
            if (word.size == 1) break
        }

        bpeCache[token] = word
        return word
    }

    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = mutableSetOf<Pair<String, String>>()
        if (word.size < 2) return pairs
        for (i in 0 until word.size - 1) {
            pairs.add(word[i] to word[i + 1])
        }
        return pairs
    }
}
