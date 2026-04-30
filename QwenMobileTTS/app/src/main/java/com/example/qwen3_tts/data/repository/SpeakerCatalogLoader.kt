package com.example.qwen3_tts.data.repository

import org.json.JSONObject
import java.io.File
import java.io.InputStream

class SpeakerCatalogLoader(
    private val tag: String = "Qwen3TTS"
) {
    data class SpeakerCatalog(
        val speakersByName: Map<String, Int>
    ) {
        fun names(): List<String> = speakersByName.keys.sorted()

        fun idFor(name: String?): Int {
            if (name.isNullOrBlank()) return -1
            return speakersByName[name] ?: -1
        }
    }

    fun load(inputStream: InputStream): SpeakerCatalog {
        val text = inputStream.bufferedReader(Charsets.UTF_8).readText()
        return parseCatalog(text)
    }

    fun load(file: File): SpeakerCatalog {
        require(file.exists()) { "speaker_ids.json not found: ${file.absolutePath}" }
        return parseCatalog(file.readText(Charsets.UTF_8))
    }

    private fun parseCatalog(text: String): SpeakerCatalog {
        val json = JSONObject(text)
        val out = mutableMapOf<String, Int>()
        for (key in json.keys()) {
            out[key] = json.getInt(key)
        }
        return SpeakerCatalog(out)
    }
}