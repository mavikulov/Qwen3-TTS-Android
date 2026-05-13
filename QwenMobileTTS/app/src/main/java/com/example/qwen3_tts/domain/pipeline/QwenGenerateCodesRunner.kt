package com.example.qwen3_tts.domain.pipeline

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.qwen3_tts.config.QwenConfigLoader
import com.example.qwen3_tts.data.repository.QwenEmbeddingRepository
import com.example.qwen3_tts.data.repository.QwenModelRepository
import com.example.qwen3_tts.domain.builders.QwenNextInputBuilder
import com.example.qwen3_tts.domain.builders.QwenPrefillBuilder
import com.example.qwen3_tts.inference.runners.QwenCodePredictorLoopRunner
import com.example.qwen3_tts.inference.runners.QwenTalkerDecodeRunner
import com.example.qwen3_tts.inference.runners.QwenTalkerPrefillRunner

class QwenGenerateCodesRunner(
    private val tag: String = "Qwen3TTS"
) {
    data class GenerateCodesResult(
        val codes: LongArray,
        val timeSteps: Int,
        val codebooks: Int,
        val generatedGroup0Tokens: List<Int>
    )

    fun run(
        repo: QwenModelRepository,
        config: QwenConfigLoader.FullConfig,
        tokenIds: List<Int>,
        embeddings: QwenEmbeddingRepository,
        language: String = "auto",
        speakerId: Int = -1,
        maxNewTokens: Int = 64,
        temperature: Float = 1.0f,
        topK: Int = 50,
        repetitionPenalty: Float = 1.1f,
        minNewTokens: Int = 0
    ): GenerateCodesResult {
        require(tokenIds.isNotEmpty()) { "tokenIds must not be empty" }
        require(maxNewTokens > 0) { "maxNewTokens must be > 0" }

        val env = OrtEnvironment.getEnvironment()

        val prefillBuilder = QwenPrefillBuilder(tag)
        val talkerPrefillRunner = QwenTalkerPrefillRunner(tag)
        val talkerDecodeRunner = QwenTalkerDecodeRunner(tag)
        val codePredictorLoopRunner = QwenCodePredictorLoopRunner(tag)
        val nextInputBuilder = QwenNextInputBuilder(tag)

        val prefill = prefillBuilder.build(
            tokenIds = tokenIds,
            config = config,
            embeddings = embeddings,
            language = language,
            speakerId = speakerId
        )

        val prefillRunResult = OrtSession.SessionOptions().use { options ->
            env.createSession(
                repo.getModelFile("talker_prefill.onnx").absolutePath,
                options
            ).use { prefillSession ->
                talkerPrefillRunner.run(
                    session = prefillSession,
                    prefill = prefill
                )
            }
        }

        val generatedSteps = mutableListOf<List<Int>>()
        val generatedGroup0Tokens = mutableListOf<Int>()

        OrtSession.SessionOptions().use { decodeOptions ->
            env.createSession(
                repo.getModelFile("talker_decode.onnx").absolutePath,
                decodeOptions
            ).use { decodeSession ->

                OrtSession.SessionOptions().use { cpOptions ->
                    env.createSession(
                        repo.getModelFile("code_predictor.onnx").absolutePath,
                        cpOptions
                    ).use { codePredictorSession ->

                        val timestep0 = codePredictorLoopRunner.runFromPrefill(
                            session = codePredictorSession,
                            config = config,
                            prefillResult = prefillRunResult,
                            embeddings = embeddings,
                            previousGroup0Tokens = generatedGroup0Tokens,
                            temperature = temperature,
                            topK = topK,
                            repetitionPenalty = repetitionPenalty,
                            minNewTokens = minNewTokens,
                            step = 0
                        )

                        if (timestep0.hitEos) {
                            return GenerateCodesResult(
                                codes = LongArray(0),
                                timeSteps = 0,
                                codebooks = 16,
                                generatedGroup0Tokens = emptyList()
                            )
                        }

                        generatedSteps.add(timestep0.codes16)
                        generatedGroup0Tokens.add(timestep0.codes16[0])

                        var nextInput = nextInputBuilder.buildFromCodes(
                            codes16 = timestep0.codes16,
                            timestepIndex = 0,
                            prefill = prefill,
                            config = config,
                            embeddings = embeddings
                        )

                        var decodeRunResult = talkerDecodeRunner.runOneStepAfterPrefill(
                            session = decodeSession,
                            config = config,
                            prefillResult = prefillRunResult,
                            nextInputEmbeds = nextInput
                        )

                        for (step in 1 until maxNewTokens) {
                            val stepCodes = codePredictorLoopRunner.runFromDecode(
                                session = codePredictorSession,
                                config = config,
                                decodeResult = decodeRunResult,
                                embeddings = embeddings,
                                previousGroup0Tokens = generatedGroup0Tokens,
                                temperature = temperature,
                                topK = topK,
                                repetitionPenalty = repetitionPenalty,
                                minNewTokens = minNewTokens,
                                step = step
                            )

                            if (stepCodes.hitEos) {
                                break
                            }

                            generatedSteps.add(stepCodes.codes16)
                            generatedGroup0Tokens.add(stepCodes.codes16[0])

                            if (step == maxNewTokens - 1) {
                                break
                            }

                            nextInput = nextInputBuilder.buildFromCodes(
                                codes16 = stepCodes.codes16,
                                timestepIndex = step,
                                prefill = prefill,
                                config = config,
                                embeddings = embeddings
                            )

                            decodeRunResult = talkerDecodeRunner.runOneStepFromNextInput(
                                session = decodeSession,
                                config = config,
                                previousDecodeResult = decodeRunResult,
                                nextInputEmbeds = nextInput
                            )
                        }
                    }
                }
            }
        }

        val timeSteps = generatedSteps.size
        val flatCodes = flattenCodesToB16T(generatedSteps)

        return GenerateCodesResult(
            codes = flatCodes,
            timeSteps = timeSteps,
            codebooks = 16,
            generatedGroup0Tokens = generatedGroup0Tokens.toList()
        )
    }

    private fun flattenCodesToB16T(steps: List<List<Int>>): LongArray {
        if (steps.isEmpty()) return LongArray(0)

        val timeSteps = steps.size
        val codebooks = 16
        val out = LongArray(codebooks * timeSteps)

        for (t in 0 until timeSteps) {
            val step = steps[t]
            require(step.size == codebooks) {
                "Step $t has ${step.size} codebooks, expected $codebooks"
            }

            for (g in 0 until codebooks) {
                val idx = g * timeSteps + t
                out[idx] = step[g].toLong()
            }
        }

        return out
    }
}