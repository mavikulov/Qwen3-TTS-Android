package com.example.qwen3_tts.storage

import android.content.Context
import java.io.File

class QwenStoragePaths(context: Context) {
    val rootDir: File = File(context.filesDir, "qwen3_tts")
    val modelsDir: File = File(rootDir, "models")
    val embeddingsDir: File = File(rootDir, "embeddings")
    val tokenizerDir: File = File(rootDir, "tokenizer")

    fun ensureDirectories() {
        if (!rootDir.exists())
            rootDir.mkdirs()

        if (!modelsDir.exists())
            modelsDir.mkdirs()

        if (!embeddingsDir.exists())
            embeddingsDir.mkdirs()

        if (!tokenizerDir.exists())
            tokenizerDir.mkdirs()
    }
}