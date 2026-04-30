package com.example.qwen3_tts.tokenizer

import android.content.res.AssetManager
import org.json.JSONObject
import java.io.File

class QwenBpeResources(
    private val tag: String = "Qwen3TTS"
) {
    data class Resources(
        val vocab: Map<String, Int>,
        val mergesRank: Map<Pair<String, String>, Int>
    )

    fun load(assetManager: AssetManager, assetDir: String): Resources {
        val vocabText = assetManager.open("$assetDir/vocab.json")
            .bufferedReader(Charsets.UTF_8).readText()
        val mergesLines = assetManager.open("$assetDir/merges.txt")
            .bufferedReader(Charsets.UTF_8).readLines()
        return parseResources(vocabText, mergesLines)
    }

    fun load(tokenizerDir: File): Resources {
        val vocabFile = File(tokenizerDir, "vocab.json")
        val mergesFile = File(tokenizerDir, "merges.txt")

        require(vocabFile.exists()) { "Missing vocab.json at ${vocabFile.absolutePath}" }
        require(mergesFile.exists()) { "Missing merges.txt at ${mergesFile.absolutePath}" }

        return parseResources(
            vocabText = vocabFile.readText(Charsets.UTF_8),
            mergesLines = mergesFile.readLines(Charsets.UTF_8)
        )
    }

    private fun parseResources(vocabText: String, mergesLines: List<String>): Resources {
        val vocabJson = JSONObject(vocabText)
        val vocab = mutableMapOf<String, Int>()
        for (key in vocabJson.keys()) {
            vocab[key] = vocabJson.getInt(key)
        }

        val mergesRank = mutableMapOf<Pair<String, String>, Int>()
        var rank = 0
        for (line in mergesLines) {
            val trimmed = line.trim()
            if (trimmed.isEmpty() || trimmed.startsWith("#")) continue

            val parts = trimmed.split(" ")
                .map { it.trim() }
                .filter { it.isNotEmpty() }

            if (parts.size != 2) continue

            mergesRank[parts[0] to parts[1]] = rank
            rank++
        }

        return Resources(vocab = vocab, mergesRank = mergesRank)
    }
}