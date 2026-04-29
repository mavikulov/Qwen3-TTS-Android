package com.example.qwen3_tts.domain.sampling

import kotlin.math.exp
import kotlin.random.Random
import com.example.qwen3_tts.config.QwenConfigLoader

object QwenSampling {
    fun softmax(x: FloatArray): FloatArray {
        if (x.isEmpty()) return x

        var maxVal = x[0]
        for (i in 1 until x.size) {
            if (x[i] > maxVal) maxVal = x[i]
        }

        val exponents = FloatArray(x.size)
        var sum = 0.0
        for (i in x.indices) {
            if (x[i].isInfinite() && x[i] < 0f) {
                exponents[i] = 0f
            } else {
                val v = exp((x[i] - maxVal).toDouble()).toFloat()
                exponents[i] = v
                sum += v
            }
        }

        if (sum <= 0.0) {
            val fallback = FloatArray(x.size)
            fallback[0] = 1f
            return fallback
        }

        for (i in exponents.indices) {
            exponents[i] = (exponents[i] / sum).toFloat()
        }
        return exponents
    }

    fun sampleTalkerToken(
        logits: FloatArray,
        config: QwenConfigLoader.FullConfig,
        previousTokens: List<Int>,
        temperature: Float,
        topK: Int,
        repetitionPenalty: Float,
        minNewTokens: Int,
        step: Int,
        random: Random = Random.Default
    ): Int {
        val talkerVocabSize = config.talker.vocabSize
        val cpVocabSize = config.codePredictor.vocabSize
        val codecEosTokenId = config.talker.codecEosTokenId

        require(logits.size >= talkerVocabSize) {
            "logits size ${logits.size} < talker vocab size $talkerVocabSize"
        }

        val x = logits.copyOfRange(logits.size - talkerVocabSize, logits.size)

        for (token in previousTokens) {
            if (token in x.indices) {
                if (x[token] > 0f) {
                    x[token] /= repetitionPenalty
                } else {
                    x[token] *= repetitionPenalty
                }
            }
        }

        for (i in cpVocabSize until talkerVocabSize) {
            if (i != codecEosTokenId) {
                x[i] = Float.NEGATIVE_INFINITY
            }
        }

        if (step < minNewTokens && codecEosTokenId in x.indices) {
            x[codecEosTokenId] = Float.NEGATIVE_INFINITY
        }

        if (temperature > 0f) {
            for (i in x.indices) {
                if (!(x[i].isInfinite() && x[i] < 0f)) {
                    x[i] /= temperature
                }
            }
        }

        applyTopKInPlace(x, topK)

        val probs = softmax(x)
        return sampleFromProbs(probs, random)
    }

    fun sampleCpToken(
        logits: FloatArray,
        config: QwenConfigLoader.FullConfig,
        temperature: Float,
        topK: Int,
        random: Random = Random.Default
    ): Int {
        val cpVocabSize = config.codePredictor.vocabSize
        require(logits.size >= cpVocabSize) {
            "logits size ${logits.size} < cp vocab size $cpVocabSize"
        }

        val x = logits.copyOfRange(logits.size - cpVocabSize, logits.size)

        applyTopKInPlace(x, topK)

        if (temperature > 0f) {
            for (i in x.indices) {
                if (!(x[i].isInfinite() && x[i] < 0f)) {
                    x[i] /= temperature
                }
            }
        }

        val probs = softmax(x)
        return sampleFromProbs(probs, random)
    }

    private fun applyTopKInPlace(x: FloatArray, topK: Int) {
        if (topK <= 0 || topK >= x.size) return

        val indexed = x.indices
            .map { it to x[it] }
            .sortedByDescending { it.second }

        val keep = indexed.take(topK).map { it.first }.toSet()
        for (i in x.indices) {
            if (i !in keep) {
                x[i] = Float.NEGATIVE_INFINITY
            }
        }
    }

    private fun sampleFromProbs(probs: FloatArray, random: Random): Int {
        val r = random.nextFloat()
        var cumulative = 0f
        for (i in probs.indices) {
            cumulative += probs[i]
            if (r <= cumulative) return i
        }
        return probs.indices.maxByOrNull { probs[it] } ?: 0
    }
}