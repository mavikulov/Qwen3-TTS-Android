package com.example.qwen3_tts.inference.runners

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.LongBuffer

class QwenVocoderRunner(
    private val tag: String = "Qwen3TTS"
) {
    data class VocoderInput(
        val codes: LongArray,
        val timeSteps: Int,
        val codebooks: Int
    )

    data class VocoderResult(
        val waveform: FloatArray,
        val waveformShape: LongArray?
    )

    fun run(
        modelFile: File,
        input: VocoderInput
    ): VocoderResult {
        require(modelFile.exists()) { "vocoder model does not exist: ${modelFile.absolutePath}" }
        require(modelFile.length() > 0L) { "vocoder model is empty: ${modelFile.absolutePath}" }

        require(input.timeSteps > 0) {
            "input.timeSteps must be > 0"
        }
        require(input.codebooks == 16) {
            "Expected 16 codebooks, got ${input.codebooks}"
        }
        require(input.codes.isNotEmpty()) {
            "input.codes is empty"
        }

        val env = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions()
        val session = env.createSession(modelFile.absolutePath, sessionOptions)

        try {
            val inputShape = longArrayOf(
                1L,
                input.codebooks.toLong(),
                input.timeSteps.toLong()
            )

            val inputName = session.inputNames.firstOrNull()
                ?: error("Vocoder session has no inputs")

            val inputTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(input.codes),
                inputShape
            )

            inputTensor.use { tensor ->
                session.run(mapOf(inputName to tensor)).use { outputs ->
                    val waveformTensor =
                        outputs.firstOrNull()?.value as? OnnxTensor
                            ?: error("Missing vocoder output tensor")

                    val waveformShape = (waveformTensor.info as? TensorInfo)?.shape
                    val waveform = waveformTensor.floatBuffer.let { fb ->
                        val arr = FloatArray(fb.remaining())
                        fb.get(arr)
                        arr
                    }

                    return VocoderResult(
                        waveform = waveform,
                        waveformShape = waveformShape
                    )
                }
            }
        } finally {
            session.close()
            sessionOptions.close()
        }
    }
}