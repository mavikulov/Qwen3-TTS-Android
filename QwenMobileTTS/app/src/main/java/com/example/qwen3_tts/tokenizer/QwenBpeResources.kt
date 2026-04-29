package com.example.qwen3_tts.tokenizer

import org.json.JSONObject
import java.io.File

class QwenBpeResources(
    private val tag: String = "Qwen3TTS"
) {
    data class Resources(
        val vocab: Map<String, Int>,
        val mergesRank: Map<Pair<String, String>, Int>
    )

    fun load(tokenizerDir: File): Resources {
        val vocabFile = File(tokenizerDir, "vocab.json")
        val mergesFile = File(tokenizerDir, "merges.txt")

        require(vocabFile.exists()) { "Missing vocab.json at ${vocabFile.absolutePath}" }
        require(mergesFile.exists()) { "Missing merges.txt at ${mergesFile.absolutePath}" }

        val vocabJson = JSONObject(vocabFile.readText(Charsets.UTF_8))
        val vocab = mutableMapOf<String, Int>()
        for (key in vocabJson.keys()) {
            vocab[key] = vocabJson.getInt(key)
        }

        val mergesRank = mutableMapOf<Pair<String, String>, Int>()
        val lines = mergesFile.readLines(Charsets.UTF_8)
        var rank = 0
        for (line in lines) {
            val trimmed = line.trim()
            if (trimmed.isEmpty()) continue
            if (trimmed.startsWith("#")) continue

            val parts = trimmed.split(" ")
                .map { it.trim() }
                .filter { it.isNotEmpty() }

            if (parts.size != 2) continue

            mergesRank[parts[0] to parts[1]] = rank
            rank++
        }

        return Resources(
            vocab = vocab,
            mergesRank = mergesRank
        )
    }
}