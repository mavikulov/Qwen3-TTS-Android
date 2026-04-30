package com.example.qwen3_tts.data.repository

import android.content.Context
import android.content.res.AssetManager
import java.io.File
import com.example.qwen3_tts.storage.QwenStoragePaths

class QwenModelRepository(
    context: Context,
) {
    private val paths = QwenStoragePaths(context)

    companion object {
        val REQUIRED_MODEL_FILES = listOf(
            "talker_prefill.onnx",
            "talker_decode.onnx",
            "code_predictor.onnx",
            "vocoder.onnx"
        )

        val OPTIONAL_MODEL_DATA_FILES = listOf(
            "talker_prefill.onnx.data",
            "talker_decode.onnx.data",
            "code_predictor.onnx.data",
            "vocoder.onnx.data"
        )

        val REQUIRED_TOKENIZER_FILES = listOf(
            "vocab.json",
            "merges.txt"
        )

        val REQUIRED_EMBEDDING_FILES = listOf(
            "config.json",
            "text_embedding.npy",
            "text_projection_fc1_weight.npy",
            "text_projection_fc1_bias.npy",
            "text_projection_fc2_weight.npy",
            "text_projection_fc2_bias.npy",
            "talker_codec_embedding.npy"
        ) + (0..14).map { "cp_codec_embedding_$it.npy" }

        // Asset layout mirrors the bundle directory structure.
        const val ASSET_MODELS_DIR = "models"
        const val ASSET_EMBEDDINGS_DIR = "embeddings"
        const val ASSET_TOKENIZER_DIR = "tokenizer"
    }

    init {
        paths.ensureDirectories()
    }

    // ── File-based access (ZIP-import path) ──────────────────────────────────

    fun getModelFile(name: String): File = File(paths.modelsDir, name)
    fun getEmbeddingFile(name: String): File = File(paths.embeddingsDir, name)
    fun getTokenizerFile(name: String): File = File(paths.tokenizerDir, name)
    fun getTokenizerDir(): File = paths.tokenizerDir

    // ── Asset-based access ───────────────────────────────────────────────────

    fun getModelAssetPath(name: String): String = "$ASSET_MODELS_DIR/$name"
    fun getEmbeddingAssetPath(name: String): String = "$ASSET_EMBEDDINGS_DIR/$name"
    fun getTokenizerAssetPath(name: String): String = "$ASSET_TOKENIZER_DIR/$name"

    fun isAssetBundlePresent(assetManager: AssetManager): Boolean {
        return try {
            assetManager.list(ASSET_MODELS_DIR)?.contains("talker_prefill.onnx") == true
        } catch (_: Exception) {
            false
        }
    }

    // ── Validation ───────────────────────────────────────────────────────────

    fun missingRequiredFiles(): List<String> {
        val missing = mutableListOf<String>()

        for (name in REQUIRED_MODEL_FILES) {
            val f = getModelFile(name)
            if (!f.exists() || f.length() <= 0L) {
                missing.add("models/$name")
            }
        }

        for (name in REQUIRED_TOKENIZER_FILES) {
            val f = getTokenizerFile(name)
            if (!f.exists() || f.length() <= 0L) {
                missing.add("tokenizer/$name")
            }
        }

        for (name in REQUIRED_EMBEDDING_FILES) {
            val f = getEmbeddingFile(name)
            if (!f.exists() || f.length() <= 0L) {
                missing.add("embeddings/$name")
            }
        }

        return missing
    }
}
