package com.example.qwen3_tts.inference.runners

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import com.example.qwen3_tts.config.QwenConfigLoader
import java.nio.FloatBuffer
import java.nio.LongBuffer

class QwenTalkerDecodeRunner(
    private val tag: String = "Qwen3TTS"
) {
    data class DecodeRunResult(
        val logits: FloatArray,
        val logitsShape: LongArray?,
        val hiddenStates: FloatArray,
        val hiddenStatesShape: LongArray?,
        val presentKeys: FloatArray,
        val presentKeysShape: LongArray?,
        val presentValues: FloatArray,
        val presentValuesShape: LongArray?
    )

    fun runOneStepAfterPrefill(
        session: OrtSession,
        config: QwenConfigLoader.FullConfig,
        prefillResult: QwenTalkerPrefillRunner.PrefillRunResult,
        nextInputEmbeds: FloatArray
    ): DecodeRunResult {
        val env = OrtEnvironment.getEnvironment()

        val hiddenSize = config.talker.hiddenSize
        val numLayers = config.talker.numHiddenLayers
        val numKvHeads = config.talker.numKeyValueHeads
        val headDim = config.talker.headDim

        require(nextInputEmbeds.size == hiddenSize) {
            "nextInputEmbeds size must be $hiddenSize, got ${nextInputEmbeds.size}"
        }

        val prefillKeysShape = requireNotNull(prefillResult.presentKeysShape) {
            "prefill presentKeysShape is null"
        }

        val pastSeqLen = prefillKeysShape[3].toInt()
        val totalSeqLen = pastSeqLen + 1

        val attentionMask = LongArray(totalSeqLen) { 1L }
        val positionIds = longArrayOf(
            pastSeqLen.toLong(),
            pastSeqLen.toLong(),
            pastSeqLen.toLong()
        )

        val pastKeysShape = longArrayOf(
            numLayers.toLong(),
            1L,
            numKvHeads.toLong(),
            pastSeqLen.toLong(),
            headDim.toLong()
        )

        val pastValuesShape = longArrayOf(
            numLayers.toLong(),
            1L,
            numKvHeads.toLong(),
            pastSeqLen.toLong(),
            headDim.toLong()
        )

        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(nextInputEmbeds),
            longArrayOf(1, 1, hiddenSize.toLong())
        ).use { inputsEmbedsTensor ->
            OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(attentionMask),
                longArrayOf(1, totalSeqLen.toLong())
            ).use { attentionMaskTensor ->
                OnnxTensor.createTensor(
                    env,
                    LongBuffer.wrap(positionIds),
                    longArrayOf(3, 1, 1)
                ).use { positionIdsTensor ->
                    OnnxTensor.createTensor(
                        env,
                        FloatBuffer.wrap(prefillResult.presentKeys),
                        pastKeysShape
                    ).use { pastKeysTensor ->
                        OnnxTensor.createTensor(
                            env,
                            FloatBuffer.wrap(prefillResult.presentValues),
                            pastValuesShape
                        ).use { pastValuesTensor ->

                            val inputs = mapOf(
                                "inputs_embeds" to inputsEmbedsTensor,
                                "attention_mask" to attentionMaskTensor,
                                "position_ids" to positionIdsTensor,
                                "past_keys" to pastKeysTensor,
                                "past_values" to pastValuesTensor
                            )

                            session.run(inputs).use { outputs ->
                                return readDecodeOutputs(outputs)
                            }
                        }
                    }
                }
            }
        }
    }

    fun runOneStepFromNextInput(
        session: OrtSession,
        config: QwenConfigLoader.FullConfig,
        previousDecodeResult: DecodeRunResult,
        nextInputEmbeds: FloatArray
    ): DecodeRunResult {
        val env = OrtEnvironment.getEnvironment()

        val hiddenSize = config.talker.hiddenSize
        val numLayers = config.talker.numHiddenLayers
        val numKvHeads = config.talker.numKeyValueHeads
        val headDim = config.talker.headDim

        require(nextInputEmbeds.size == hiddenSize) {
            "nextInputEmbeds size must be $hiddenSize, got ${nextInputEmbeds.size}"
        }

        val previousPresentKeysShape = requireNotNull(previousDecodeResult.presentKeysShape) {
            "previousDecodeResult.presentKeysShape is null"
        }

        val pastSeqLen = previousPresentKeysShape[3].toInt()
        val totalSeqLen = pastSeqLen + 1

        val attentionMask = LongArray(totalSeqLen) { 1L }
        val positionIds = longArrayOf(
            pastSeqLen.toLong(),
            pastSeqLen.toLong(),
            pastSeqLen.toLong()
        )

        val pastKeysShape = longArrayOf(
            numLayers.toLong(),
            1L,
            numKvHeads.toLong(),
            pastSeqLen.toLong(),
            headDim.toLong()
        )

        val pastValuesShape = longArrayOf(
            numLayers.toLong(),
            1L,
            numKvHeads.toLong(),
            pastSeqLen.toLong(),
            headDim.toLong()
        )

        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(nextInputEmbeds),
            longArrayOf(1, 1, hiddenSize.toLong())
        ).use { inputsEmbedsTensor ->
            OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(attentionMask),
                longArrayOf(1, totalSeqLen.toLong())
            ).use { attentionMaskTensor ->
                OnnxTensor.createTensor(
                    env,
                    LongBuffer.wrap(positionIds),
                    longArrayOf(3, 1, 1)
                ).use { positionIdsTensor ->
                    OnnxTensor.createTensor(
                        env,
                        FloatBuffer.wrap(previousDecodeResult.presentKeys),
                        pastKeysShape
                    ).use { pastKeysTensor ->
                        OnnxTensor.createTensor(
                            env,
                            FloatBuffer.wrap(previousDecodeResult.presentValues),
                            pastValuesShape
                        ).use { pastValuesTensor ->

                            val inputs = mapOf(
                                "inputs_embeds" to inputsEmbedsTensor,
                                "attention_mask" to attentionMaskTensor,
                                "position_ids" to positionIdsTensor,
                                "past_keys" to pastKeysTensor,
                                "past_values" to pastValuesTensor
                            )

                            session.run(inputs).use { outputs ->
                                return readDecodeOutputs(outputs)
                            }
                        }
                    }
                }
            }
        }
    }

    private fun readDecodeOutputs(
        outputs: OrtSession.Result
    ): DecodeRunResult {
        val logitsTensor =
            outputs.firstOrNull { it.key == "logits" }?.value as? OnnxTensor
                ?: error("Missing decode logits")

        val hiddenStatesTensor =
            outputs.firstOrNull { it.key == "hidden_states" }?.value as? OnnxTensor
                ?: error("Missing decode hidden_states")

        val presentKeysTensor =
            outputs.firstOrNull { it.key == "present_keys" }?.value as? OnnxTensor
                ?: error("Missing decode present_keys")

        val presentValuesTensor =
            outputs.firstOrNull { it.key == "present_values" }?.value as? OnnxTensor
                ?: error("Missing decode present_values")

        val logitsShape = (logitsTensor.info as? TensorInfo)?.shape
        val hiddenStatesShape = (hiddenStatesTensor.info as? TensorInfo)?.shape
        val presentKeysShape = (presentKeysTensor.info as? TensorInfo)?.shape
        val presentValuesShape = (presentValuesTensor.info as? TensorInfo)?.shape

        val logits = logitsTensor.floatBuffer.toFloatArray()
        val hiddenStates = hiddenStatesTensor.floatBuffer.toFloatArray()
        val presentKeys = presentKeysTensor.floatBuffer.toFloatArray()
        val presentValues = presentValuesTensor.floatBuffer.toFloatArray()

        return DecodeRunResult(
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

    private fun java.nio.FloatBuffer.toFloatArray(): FloatArray {
        val out = FloatArray(remaining())
        get(out)
        return out
    }
}