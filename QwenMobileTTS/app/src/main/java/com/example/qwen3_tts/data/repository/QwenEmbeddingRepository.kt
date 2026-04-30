package com.example.qwen3_tts.data.repository

import android.content.res.AssetManager
import com.example.qwen3_tts.data.npy.NpyFloatRowReader
import com.example.qwen3_tts.data.npy.NpyHalfRowReader
import com.example.qwen3_tts.data.npy.NpyHalfRowReaderAsset
import com.example.qwen3_tts.data.npy.NpyInt8ScaledRowReaderAsset
import com.example.qwen3_tts.data.npy.NpyReader
import com.example.qwen3_tts.data.npy.NpyFormat
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import kotlin.math.sqrt

class QwenEmbeddingRepository(
    private val modelRepository: QwenModelRepository,
    private val tag: String = "Qwen3TTS"
) : Closeable {

    private val npyReader = NpyReader()

    private interface RowSource : Closeable {
        fun readRowAsFloat(rowIndex: Int): FloatArray
    }

    private class FloatRowSource(
        private val reader: NpyFloatRowReader
    ) : RowSource {
        override fun readRowAsFloat(rowIndex: Int): FloatArray = reader.readRow(rowIndex)
        override fun close() = reader.close()
    }

    private class HalfRowSource(
        private val reader: NpyHalfRowReader
    ) : RowSource {
        override fun readRowAsFloat(rowIndex: Int): FloatArray = reader.readRowAsFloat(rowIndex)
        override fun close() = reader.close()
    }

    private class HalfRowAssetSource(
        private val reader: NpyHalfRowReaderAsset
    ) : RowSource {
        override fun readRowAsFloat(rowIndex: Int): FloatArray = reader.readRowAsFloat(rowIndex)
        override fun close() = reader.close()
    }

    private class Int8ScaledRowAssetSource(
        private val reader: NpyInt8ScaledRowReaderAsset
    ) : RowSource {
        override fun readRowAsFloat(rowIndex: Int): FloatArray = reader.readRowAsFloat(rowIndex)
        override fun close() = reader.close()
    }

    private var textEmbeddingRowSource: RowSource? = null
    private var talkerCodecEmbeddingRowSource: RowSource? = null
    private val cpCodecEmbeddingSources = mutableMapOf<Int, RowSource>()

    private var textProjectionFc1WeightCache: NpyReader.NpyFloatArray? = null
    private var textProjectionFc1BiasCache: NpyReader.NpyFloatArray? = null
    private var textProjectionFc2WeightCache: NpyReader.NpyFloatArray? = null
    private var textProjectionFc2BiasCache: NpyReader.NpyFloatArray? = null

    // ── File-based loading (ZIP-import path) ─────────────────────────────────

    fun loadRequiredCoreEmbeddings() {
        textEmbeddingRowSource = openRowSource(modelRepository.getEmbeddingFile("text_embedding.npy"))
        talkerCodecEmbeddingRowSource = openRowSource(modelRepository.getEmbeddingFile("talker_codec_embedding.npy"))

        for (i in 0 until 15) {
            cpCodecEmbeddingSources[i] =
                openRowSource(modelRepository.getEmbeddingFile("cp_codec_embedding_$i.npy"))
        }

        textProjectionFc1WeightCache = loadSmallEmbeddingFile("text_projection_fc1_weight.npy")
        textProjectionFc1BiasCache = loadSmallEmbeddingFile("text_projection_fc1_bias.npy")
        textProjectionFc2WeightCache = loadSmallEmbeddingFile("text_projection_fc2_weight.npy")
        textProjectionFc2BiasCache = loadSmallEmbeddingFile("text_projection_fc2_bias.npy")
    }

    // ── Asset-based loading ───────────────────────────────────────────────────

    fun loadRequiredCoreEmbeddingsFromAssets(assetManager: AssetManager) {
        val textEmbPath = modelRepository.getEmbeddingAssetPath("text_embedding.npy")
        val scalesPath = modelRepository.getEmbeddingAssetPath("text_embedding_scales.npy")

        // If scales file exists → INT8 quantized; otherwise FP16
        val assetFiles = assetManager.list(QwenModelRepository.ASSET_EMBEDDINGS_DIR) ?: emptyArray()
        textEmbeddingRowSource = if ("text_embedding_scales.npy" in assetFiles) {
            Int8ScaledRowAssetSource(
                NpyInt8ScaledRowReaderAsset(assetManager, textEmbPath, scalesPath)
            )
        } else {
            HalfRowAssetSource(NpyHalfRowReaderAsset(assetManager, textEmbPath))
        }

        talkerCodecEmbeddingRowSource = HalfRowAssetSource(
            NpyHalfRowReaderAsset(
                assetManager,
                modelRepository.getEmbeddingAssetPath("talker_codec_embedding.npy")
            )
        )

        for (i in 0 until 15) {
            cpCodecEmbeddingSources[i] = HalfRowAssetSource(
                NpyHalfRowReaderAsset(
                    assetManager,
                    modelRepository.getEmbeddingAssetPath("cp_codec_embedding_$i.npy")
                )
            )
        }

        textProjectionFc1WeightCache = npyReader.readFloatArrayFromAsset(
            assetManager, modelRepository.getEmbeddingAssetPath("text_projection_fc1_weight.npy")
        )
        textProjectionFc1BiasCache = npyReader.readFloatArrayFromAsset(
            assetManager, modelRepository.getEmbeddingAssetPath("text_projection_fc1_bias.npy")
        )
        textProjectionFc2WeightCache = npyReader.readFloatArrayFromAsset(
            assetManager, modelRepository.getEmbeddingAssetPath("text_projection_fc2_weight.npy")
        )
        textProjectionFc2BiasCache = npyReader.readFloatArrayFromAsset(
            assetManager, modelRepository.getEmbeddingAssetPath("text_projection_fc2_bias.npy")
        )
    }

    // ── Row access (same for both paths) ─────────────────────────────────────

    fun textEmbeddingRow(tokenId: Int): FloatArray {
        val source = requireNotNull(textEmbeddingRowSource) {
            "text_embedding.npy row source not loaded"
        }
        return source.readRowAsFloat(tokenId)
    }

    fun talkerCodecEmbeddingRow(tokenId: Int): FloatArray {
        val source = requireNotNull(talkerCodecEmbeddingRowSource) {
            "talker_codec_embedding.npy row source not loaded"
        }
        return source.readRowAsFloat(tokenId)
    }

    fun cpCodecEmbeddingRow(groupIndex: Int, tokenId: Int): FloatArray {
        val source = cpCodecEmbeddingSources[groupIndex]
            ?: error("cp_codec_embedding_$groupIndex.npy source not loaded")
        return source.readRowAsFloat(tokenId)
    }

    fun textProjection(input2048: FloatArray): FloatArray {
        val fc1W = requireNotNull(textProjectionFc1WeightCache) { "fc1 weight not loaded" }
        val fc1B = requireNotNull(textProjectionFc1BiasCache) { "fc1 bias not loaded" }
        val fc2W = requireNotNull(textProjectionFc2WeightCache) { "fc2 weight not loaded" }
        val fc2B = requireNotNull(textProjectionFc2BiasCache) { "fc2 bias not loaded" }

        require(fc1W.shape.contentEquals(intArrayOf(2048, 2048))) {
            "Unexpected fc1 weight shape: ${fc1W.shape.contentToString()}"
        }
        require(fc1B.shape.contentEquals(intArrayOf(2048))) {
            "Unexpected fc1 bias shape: ${fc1B.shape.contentToString()}"
        }
        require(fc2W.shape.contentEquals(intArrayOf(1024, 2048))) {
            "Unexpected fc2 weight shape: ${fc2W.shape.contentToString()}"
        }
        require(fc2B.shape.contentEquals(intArrayOf(1024))) {
            "Unexpected fc2 bias shape: ${fc2B.shape.contentToString()}"
        }
        require(input2048.size == 2048) {
            "textProjection expects input size 2048, got ${input2048.size}"
        }

        val fc1Out = FloatArray(2048)
        for (row in 0 until 2048) {
            var sum = fc1B.data[row].toDouble()
            val rowOffset = row * 2048
            for (col in 0 until 2048) {
                sum += input2048[col].toDouble() * fc1W.data[rowOffset + col].toDouble()
            }
            fc1Out[row] = GeLU(sum).toFloat()
        }

        val fc2Out = FloatArray(1024)
        for (row in 0 until 1024) {
            var sum = fc2B.data[row].toDouble()
            val rowOffset = row * 2048
            for (col in 0 until 2048) {
                sum += fc1Out[col].toDouble() * fc2W.data[rowOffset + col].toDouble()
            }
            fc2Out[row] = sum.toFloat()
        }

        return fc2Out
    }

    override fun close() {
        textEmbeddingRowSource?.close()
        talkerCodecEmbeddingRowSource?.close()
        cpCodecEmbeddingSources.values.forEach { it.close() }

        textEmbeddingRowSource = null
        talkerCodecEmbeddingRowSource = null
        cpCodecEmbeddingSources.clear()
    }

    private fun openRowSource(file: File): RowSource {
        val dtype = RandomAccessFile(file, "r").use { raf ->
            NpyFormat.readMetadataFromRaf(file, raf).dtype
        }

        return when (dtype) {
            NpyFormat.DType.FLOAT32 -> FloatRowSource(NpyFloatRowReader(file))
            NpyFormat.DType.FLOAT16 -> HalfRowSource(NpyHalfRowReader(file))
            NpyFormat.DType.INT8 -> error(
                "INT8 embeddings loaded via file path require a scales file. " +
                "Use loadRequiredCoreEmbeddingsFromAssets() instead."
            )
        }
    }

    private fun loadSmallEmbeddingFile(name: String): NpyReader.NpyFloatArray {
        val file: File = modelRepository.getEmbeddingFile(name)
        return npyReader.readFloatArray(file)
    }

    private fun GeLU(x: Double): Double {
        return 0.5 * x * (1.0 + kotlin.math.tanh(sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)))
    }
}
