package com.example.qwen3_tts.config

import org.json.JSONObject
import java.io.File

class QwenConfigLoader(
    private val tag: String = "Qwen3TTS"
) {
    data class TalkerConfig(
        val hiddenSize: Int,
        val numHiddenLayers: Int,
        val numKeyValueHeads: Int,
        val headDim: Int,
        val codecBosId: Int,
        val codecPadId: Int,
        val codecEosTokenId: Int,
        val codecThinkId: Int,
        val codecNoThinkId: Int,
        val codecThinkBosId: Int,
        val codecThinkEosId: Int,
        val vocabSize: Int
    )

    data class CodePredictorConfig(
        val numHiddenLayers: Int,
        val numKeyValueHeads: Int,
        val headDim: Int,
        val vocabSize: Int
    )

    data class TtsConfig(
        val ttsPadTokenId: Int,
        val ttsBosTokenId: Int,
        val ttsEosTokenId: Int
    )

    data class FullConfig(
        val talker: TalkerConfig,
        val codePredictor: CodePredictorConfig,
        val tts: TtsConfig,
        val languageIds: Map<String, Int>
    )

    fun load(configFile: File): FullConfig {
        require(configFile.exists()) { "config.json does not exist: ${configFile.absolutePath}" }
        require(configFile.length() > 0L) { "config.json is empty: ${configFile.absolutePath}" }

        val text = configFile.readText(Charsets.UTF_8)
        val root = JSONObject(text)

        val talkerJson = root.getJSONObject("talker")
        val cpJson = root.getJSONObject("code_predictor")
        val ttsJson = root.getJSONObject("tts")
        val languageJson = root.getJSONObject("language_ids")

        val languageIds = mutableMapOf<String, Int>()
        val keys = languageJson.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            languageIds[key] = languageJson.getInt(key)
        }

        val config = FullConfig(
            talker = TalkerConfig(
                hiddenSize = talkerJson.getInt("hidden_size"),
                numHiddenLayers = talkerJson.getInt("num_hidden_layers"),
                numKeyValueHeads = talkerJson.getInt("num_key_value_heads"),
                headDim = talkerJson.getInt("head_dim"),
                codecBosId = talkerJson.getInt("codec_bos_id"),
                codecPadId = talkerJson.getInt("codec_pad_id"),
                codecEosTokenId = talkerJson.getInt("codec_eos_token_id"),
                codecThinkId = talkerJson.getInt("codec_think_id"),
                codecNoThinkId = talkerJson.getInt("codec_nothink_id"),
                codecThinkBosId = talkerJson.getInt("codec_think_bos_id"),
                codecThinkEosId = talkerJson.getInt("codec_think_eos_id"),
                vocabSize = talkerJson.getInt("vocab_size")
            ),
            codePredictor = CodePredictorConfig(
                numHiddenLayers = cpJson.getInt("num_hidden_layers"),
                numKeyValueHeads = cpJson.getInt("num_key_value_heads"),
                headDim = cpJson.getInt("head_dim"),
                vocabSize = cpJson.getInt("vocab_size")
            ),
            tts = TtsConfig(
                ttsPadTokenId = ttsJson.getInt("tts_pad_token_id"),
                ttsBosTokenId = ttsJson.getInt("tts_bos_token_id"),
                ttsEosTokenId = ttsJson.getInt("tts_eos_token_id")
            ),
            languageIds = languageIds
        )

        return config
    }
}