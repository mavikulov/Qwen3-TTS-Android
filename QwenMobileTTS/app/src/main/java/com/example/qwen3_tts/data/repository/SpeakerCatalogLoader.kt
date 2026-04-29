package com.example.qwen3_tts.data.repository

import org.json.JSONObject
import java.io.File

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

    fun load(file: File): SpeakerCatalog {
        require(file.exists()) { "speaker_ids.json not found: ${file.absolutePath}" }

        val json = JSONObject(file.readText(Charsets.UTF_8))
        val out = mutableMapOf<String, Int>()

        for (key in json.keys()) {
            out[key] = json.getInt(key)
        }

        return SpeakerCatalog(out)
    }
}