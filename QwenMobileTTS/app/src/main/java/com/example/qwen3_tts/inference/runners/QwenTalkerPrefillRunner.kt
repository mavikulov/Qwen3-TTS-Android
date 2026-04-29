package com.example.qwen3_tts.inference.runners

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import com.example.qwen3_tts.domain.builders.QwenPrefillBuilder

class QwenTalkerPrefillRunner(
    private val tag: String = "Qwen3TTS"
) {
    data class PrefillRunResult(
        val logits: FloatArray,
        val logitsShape: LongArray?,
        val hiddenStates: FloatArray,
        val hiddenStatesShape: LongArray?,
        val presentKeys: FloatArray,
        val presentKeysShape: LongArray?,
        val presentValues: FloatArray,
        val presentValuesShape: LongArray?
    )

    fun run(
        session: OrtSession,
        prefill: QwenPrefillBuilder.PrefillBuildResult
    ): PrefillRunResult {
        val env = OrtEnvironment.getEnvironment()

        val inputsEmbedsTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(prefill.inputsEmbeds),
            longArrayOf(1, prefill.inputsSeqLen.toLong(), prefill.hiddenSize.toLong())
        )

        val attentionMask = LongArray(prefill.inputsSeqLen) { 1L }
        val attentionMaskTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(attentionMask),
            longArrayOf(1, prefill.inputsSeqLen.toLong())
        )

        val positionIds = LongArray(3 * prefill.inputsSeqLen)
        for (ax in 0 until 3) {
            for (i in 0 until prefill.inputsSeqLen) {
                positionIds[ax * prefill.inputsSeqLen + i] = i.toLong()
            }
        }
        val positionIdsTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(positionIds),
            longArrayOf(3, 1, prefill.inputsSeqLen.toLong())
        )

        inputsEmbedsTensor.use { embeds ->
            attentionMaskTensor.use { mask ->
                positionIdsTensor.use { pos ->
                    val inputs = mapOf(
                        "inputs_embeds" to embeds,
                        "attention_mask" to mask,
                        "position_ids" to pos
                    )

                    session.run(inputs).use { outputs ->
                        val logitsTensor =
                            outputs.firstOrNull { it.key == "logits" }?.value as? OnnxTensor
                                ?: error("Missing prefill logits")
                        val hiddenStatesTensor =
                            outputs.firstOrNull { it.key == "hidden_states" }?.value as? OnnxTensor
                                ?: error("Missing prefill hidden_states")

                        val logitsShape = (logitsTensor.info as? TensorInfo)?.shape
                        val hiddenStatesShape = (hiddenStatesTensor.info as? TensorInfo)?.shape

                        val logits = logitsTensor.floatBuffer.let {
                            val arr = FloatArray(it.remaining())
                            it.get(arr)
                            arr
                        }

                        val hiddenStates = hiddenStatesTensor.floatBuffer.let {
                            val arr = FloatArray(it.remaining())
                            it.get(arr)
                            arr
                        }

                        val presentKeyTensors = outputs
                            .filter { it.key.startsWith("present_key_") }
                            .sortedBy { it.key.removePrefix("present_key_").toInt() }

                        val presentValueTensors = outputs
                            .filter { it.key.startsWith("present_value_") }
                            .sortedBy { it.key.removePrefix("present_value_").toInt() }

                        require(presentKeyTensors.isNotEmpty()) { "No prefill present_key_* outputs found" }
                        require(presentValueTensors.isNotEmpty()) { "No prefill present_value_* outputs found" }
                        require(presentKeyTensors.size == presentValueTensors.size) {
                            "Mismatch between present_key and present_value output counts"
                        }

                        val firstKeyTensor = presentKeyTensors.first().value as? OnnxTensor
                            ?: error("present_key_0 is not a tensor")
                        val firstValueTensor = presentValueTensors.first().value as? OnnxTensor
                            ?: error("present_value_0 is not a tensor")

                        val perLayerKeyShape = (firstKeyTensor.info as? TensorInfo)?.shape
                            ?: error("present_key_0 has no shape info")
                        val perLayerValueShape = (firstValueTensor.info as? TensorInfo)?.shape
                            ?: error("present_value_0 has no shape info")

                        require(perLayerKeyShape.size == 4) {
                            "Expected prefill present_key_* rank 4, got ${perLayerKeyShape.contentToString()}"
                        }
                        require(perLayerValueShape.size == 4) {
                            "Expected prefill present_value_* rank 4, got ${perLayerValueShape.contentToString()}"
                        }

                        val numLayers = presentKeyTensors.size
                        val presentKeysShape = longArrayOf(
                            numLayers.toLong(),
                            perLayerKeyShape[0],
                            perLayerKeyShape[1],
                            perLayerKeyShape[2],
                            perLayerKeyShape[3]
                        )
                        val presentValuesShape = longArrayOf(
                            numLayers.toLong(),
                            perLayerValueShape[0],
                            perLayerValueShape[1],
                            perLayerValueShape[2],
                            perLayerValueShape[3]
                        )

                        val presentKeys = FloatArray(
                            presentKeysShape.fold(1L) { acc, v -> acc * v }.toInt()
                        )
                        val presentValues = FloatArray(
                            presentValuesShape.fold(1L) { acc, v -> acc * v }.toInt()
                        )

                        var keyOffset = 0
                        for (entry in presentKeyTensors) {
                            val tensor = entry.value as? OnnxTensor
                                ?: error("${entry.key} is not a tensor")
                            val buffer = tensor.floatBuffer
                            val count = buffer.remaining()
                            buffer.get(presentKeys, keyOffset, count)
                            keyOffset += count
                        }

                        var valueOffset = 0
                        for (entry in presentValueTensors) {
                            val tensor = entry.value as? OnnxTensor
                                ?: error("${entry.key} is not a tensor")
                            val buffer = tensor.floatBuffer
                            val count = buffer.remaining()
                            buffer.get(presentValues, valueOffset, count)
                            valueOffset += count
                        }

                        return PrefillRunResult(
                            logits = logits,
                            logitsShape = logitsShape,
                            hiddenStates = hiddenStates,
                            hiddenStatesShape = hiddenStatesShape,
                            presentKeys = presentKeys,
                            presentKeysShape = presentKeysShape,
                            presentValues = presentValues,
                            presentValuesShape = presentValuesShape
                        )
                    }
                }
            }
        }
    }

    fun run(
        modelFile: File,
        prefill: QwenPrefillBuilder.PrefillBuildResult
    ): PrefillRunResult {
        require(modelFile.exists()) { "talker_prefill model does not exist: ${modelFile.absolutePath}" }
        require(modelFile.length() > 0L) { "talker_prefill model is empty: ${modelFile.absolutePath}" }

        val env = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions()
        val session = env.createSession(modelFile.absolutePath, sessionOptions)

        try {
            return run(session, prefill)
        } finally {
            session.close()
            sessionOptions.close()
        }
    }
}