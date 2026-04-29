package com.example.qwen3_tts.inference.runners

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer
import com.example.qwen3_tts.config.QwenConfigLoader
import com.example.qwen3_tts.data.repository.QwenEmbeddingRepository
import com.example.qwen3_tts.domain.sampling.QwenSampling

class QwenCodePredictorLoopRunner(
    private val tag: String = "Qwen3TTS"
) {
    data class FullCodebookResult(
        val codes16: List<Int>,
        val finalPresentKeys: FloatArray,
        val finalPresentKeysShape: LongArray?,
        val finalPresentValues: FloatArray,
        val finalPresentValuesShape: LongArray?,
        val hitEos: Boolean
    )

    fun runFromDecode(
        session: OrtSession,
        config: QwenConfigLoader.FullConfig,
        decodeResult: QwenTalkerDecodeRunner.DecodeRunResult,
        embeddings: QwenEmbeddingRepository,
        previousGroup0Tokens: List<Int> = emptyList(),
        temperature: Float = 1.0f,
        topK: Int = 50,
        repetitionPenalty: Float = 1.1f,
        minNewTokens: Int = 0,
        step: Int = 0
    ): FullCodebookResult {
        val hiddenShape = requireNotNull(decodeResult.hiddenStatesShape) {
            "decodeResult.hiddenStatesShape is null"
        }
        require(hiddenShape.size == 3) {
            "Expected decode hidden_states rank 3, got ${hiddenShape.contentToString()}"
        }

        val seqLen = hiddenShape[1].toInt()
        val hiddenSize = hiddenShape[2].toInt()
        val start = (seqLen - 1) * hiddenSize
        val talkerHiddenLast = decodeResult.hiddenStates.copyOfRange(start, start + hiddenSize)

        return runInternal(
            session = session,
            config = config,
            talkerLogits = decodeResult.logits,
            talkerHiddenLast = talkerHiddenLast,
            embeddings = embeddings,
            previousGroup0Tokens = previousGroup0Tokens,
            temperature = temperature,
            topK = topK,
            repetitionPenalty = repetitionPenalty,
            minNewTokens = minNewTokens,
            step = step
        )
    }

    fun runFromPrefill(
        session: OrtSession,
        config: QwenConfigLoader.FullConfig,
        prefillResult: QwenTalkerPrefillRunner.PrefillRunResult,
        embeddings: QwenEmbeddingRepository,
        previousGroup0Tokens: List<Int> = emptyList(),
        temperature: Float = 1.0f,
        topK: Int = 50,
        repetitionPenalty: Float = 1.1f,
        minNewTokens: Int = 0,
        step: Int = 0
    ): FullCodebookResult {
        val hiddenShape = requireNotNull(prefillResult.hiddenStatesShape) {
            "prefill hiddenStatesShape is null"
        }
        require(hiddenShape.size == 3) {
            "Expected prefill hidden_states rank 3, got ${hiddenShape.contentToString()}"
        }

        val seqLen = hiddenShape[1].toInt()
        val hiddenSize = hiddenShape[2].toInt()
        val start = (seqLen - 1) * hiddenSize
        val talkerHiddenLast = prefillResult.hiddenStates.copyOfRange(start, start + hiddenSize)

        return runInternal(
            session = session,
            config = config,
            talkerLogits = prefillResult.logits,
            talkerHiddenLast = talkerHiddenLast,
            embeddings = embeddings,
            previousGroup0Tokens = previousGroup0Tokens,
            temperature = temperature,
            topK = topK,
            repetitionPenalty = repetitionPenalty,
            minNewTokens = minNewTokens,
            step = step
        )
    }

    private fun runInternal(
        session: OrtSession,
        config: QwenConfigLoader.FullConfig,
        talkerLogits: FloatArray,
        talkerHiddenLast: FloatArray,
        embeddings: QwenEmbeddingRepository,
        previousGroup0Tokens: List<Int>,
        temperature: Float,
        topK: Int,
        repetitionPenalty: Float,
        minNewTokens: Int,
        step: Int
    ): FullCodebookResult {
        val env = OrtEnvironment.getEnvironment()
        val hiddenSize = config.talker.hiddenSize
        val cpLayers = config.codePredictor.numHiddenLayers
        val cpKvHeads = config.codePredictor.numKeyValueHeads
        val cpHeadDim = config.codePredictor.headDim

        require(hiddenSize == 1024) {
            "Current code predictor loop expects hiddenSize=1024, got $hiddenSize"
        }
        require(talkerHiddenLast.size == hiddenSize) {
            "talkerHiddenLast size must be $hiddenSize, got ${talkerHiddenLast.size}"
        }

        val group0Raw = QwenSampling.sampleTalkerToken(
            logits = talkerLogits,
            config = config,
            previousTokens = previousGroup0Tokens,
            temperature = temperature,
            topK = topK,
            repetitionPenalty = repetitionPenalty,
            minNewTokens = minNewTokens,
            step = step
        )

        val hitEos = group0Raw == config.talker.codecEosTokenId
        val group0 = mapTalkerTokenToCodecRange(group0Raw, config)
        val group0Embed = embeddings.talkerCodecEmbeddingRow(group0)
        val allCodes = mutableListOf<Int>()
        allCodes.add(group0)

        var pastKeys = FloatArray(0)
        var pastValues = FloatArray(0)
        var pastLen = 0

        var currentInputEmbeds = FloatArray(2 * hiddenSize).also {
            talkerHiddenLast.copyInto(it, 0)
            group0Embed.copyInto(it, hiddenSize)
        }

        var currentSeqLen = 2
        var finalPresentKeys = FloatArray(0)
        var finalPresentValues = FloatArray(0)
        val finalPresentKeysShape: LongArray? = null
        val finalPresentValuesShape: LongArray? = null

        for (groupIdx in 1 until 16) {
            val generationStep = longArrayOf((groupIdx - 1).toLong())
            val inputsEmbedsTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(currentInputEmbeds),
                longArrayOf(1, currentSeqLen.toLong(), hiddenSize.toLong())
            )

            val generationStepsTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(generationStep),
                longArrayOf(1)
            )

            val pastKeysTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(pastKeys),
                longArrayOf(cpLayers.toLong(), 1, cpKvHeads.toLong(), pastLen.toLong(), cpHeadDim.toLong())
            )

            val pastValuesTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(pastValues),
                longArrayOf(cpLayers.toLong(), 1, cpKvHeads.toLong(), pastLen.toLong(), cpHeadDim.toLong())
            )

            inputsEmbedsTensor.use { embeds ->
                generationStepsTensor.use { gen ->
                    pastKeysTensor.use { pk ->
                        pastValuesTensor.use { pv ->
                            val inputs = mapOf(
                                "inputs_embeds" to embeds,
                                "generation_steps" to gen,
                                "past_keys" to pk,
                                "past_values" to pv
                            )

                            session.run(inputs).use { outputs ->
                                val logitsTensor =
                                    outputs.firstOrNull { it.key == "logits" }?.value as? OnnxTensor
                                        ?: error("Missing code_predictor logits at groupIdx=$groupIdx")
                                val presentKeysTensor =
                                    outputs.firstOrNull { it.key == "present_keys" }?.value as? OnnxTensor
                                        ?: error("Missing code_predictor present_keys at groupIdx=$groupIdx")
                                val presentValuesTensor =
                                    outputs.firstOrNull { it.key == "present_values" }?.value as? OnnxTensor
                                        ?: error("Missing code_predictor present_values at groupIdx=$groupIdx")

                                val logits = logitsTensor.floatBuffer.let {
                                    val arr = FloatArray(it.remaining())
                                    it.get(arr)
                                    arr
                                }

                                val presentKeys = presentKeysTensor.floatBuffer.let {
                                    val arr = FloatArray(it.remaining())
                                    it.get(arr)
                                    arr
                                }

                                val presentValues = presentValuesTensor.floatBuffer.let {
                                    val arr = FloatArray(it.remaining())
                                    it.get(arr)
                                    arr
                                }

                                val predictedToken = QwenSampling.sampleCpToken(
                                    logits = logits,
                                    config = config,
                                    temperature = temperature,
                                    topK = topK
                                )

                                allCodes.add(predictedToken)
                                finalPresentKeys = presentKeys
                                finalPresentValues = presentValues
                                pastKeys = presentKeys
                                pastValues = presentValues
                                pastLen += currentSeqLen

                                if (groupIdx < 15) {
                                    val nextEmbed = embeddings.cpCodecEmbeddingRow(groupIdx - 1, predictedToken)
                                    currentInputEmbeds = nextEmbed.copyOf()
                                    currentSeqLen = 1
                                }
                            }
                        }
                    }
                }
            }
        }

        return FullCodebookResult(
            codes16 = allCodes,
            finalPresentKeys = finalPresentKeys,
            finalPresentKeysShape = finalPresentKeysShape,
            finalPresentValues = finalPresentValues,
            finalPresentValuesShape = finalPresentValuesShape,
            hitEos = hitEos
        )
    }

    private fun mapTalkerTokenToCodecRange(
        token: Int,
        config: QwenConfigLoader.FullConfig
    ): Int {
        val cpVocab = config.codePredictor.vocabSize
        return when {
            token < 0 -> 0
            token < cpVocab -> token
            token == config.talker.codecEosTokenId -> 0
            else -> token % cpVocab
        }
    }
}