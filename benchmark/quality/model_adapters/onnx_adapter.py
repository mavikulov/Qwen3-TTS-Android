from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


# Token IDs from model_assets.py
END_OF_TEXT_ID = 151643
IM_START_ID = 151644
IM_END_ID = 151645
AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
TTS_PAD_ID = 151671
TTS_BOS_ID = 151672
TTS_EOD_ID = 151673
TTS_BOS_SINGLE_ID = 151674
AUDIO_PAD_ID = 151675
ASSISTANT_TOKEN_ID = 77091
NEWLINE_TOKEN_ID = 198
NUM_EMBEDDING_FILES = 15


class QwenOnnxTTSAdapter:
    """Qwen3-TTS ONNX adapter compatible with Hydra benchmark interface.

    This adapter wraps the multi-model Qwen TTS pipeline (talker + code predictor + vocoder)
    while exposing the simple generate(text, language) -> (audio, sr) interface.

    Expected model directory structure:
        model_dir/
        ├── talker_prefill.onnxещлутещлу
        ├── talker_decode.onnx
        ├── code_predictor.onnx
        ├── vocoder.onnx
        ├── tokenizer/vocab.json
        ├── tokenizer/merges.txt
        └── embeddings/
            ├── config.json
            ├── text_embedding.npy
            ├── talker_codec_embedding.npy
            ├── cp_codec_embedding_0.npy ... cp_codec_embedding_14.npy
            └── text_projection_*.npy
    """

    def __init__(
        self,
        onnx_model_path: str | Path,
        embeddings_path: str | Path | None = None,
        language_map: dict[str, str] | None = None,
        sample_rate: int = 24000,
        device: str = "cpu",
        input_text_name: str | None = None,
        language_input_name: str | None = None,
        output_audio_name: str | None = None,
        # Qwen-specific параметры
        speaker_id: int = -1,
        variant: str = "0.6b",
        max_new_tokens: int = 200,
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        min_new_tokens: int = 2,
    ) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import ByteLevelBPETokenizer
        except ImportError as exc:
            raise ImportError(
                "Packages `onnxruntime` and `tokenizers` are required for QwenOnnxTTSAdapter. "
                "Install them first: pip install onnxruntime tokenizers"
            ) from exc

        self.language_map = language_map or {}
        self.sample_rate = sample_rate
        self.model_dir = Path(onnx_model_path)
        self.emb_dir = Path(embeddings_path) if embeddings_path else self.model_dir / "embeddings"
        self.device = device
        self.speaker_id = speaker_id
        self.variant = variant.lower()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.min_new_tokens = min_new_tokens

        # Настройка провайдеров ONNX
        providers = ["CPUExecutionProvider"]
        if device.lower() in {"cuda", "gpu"} and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Загрузка моделей
        self._load_config()
        self._load_models(providers)
        self._load_tokenizer()
        self._load_assets()

        # Совместимость с интерфейсом адаптера
        self._input_text_name = input_text_name or "text"
        self._language_input_name = language_input_name or "language"
        self._output_audio_name = output_audio_name or "waveform"

    def _load_config(self):
        """Загрузка конфигурации модели."""
        config_path = self.emb_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.hidden_size = self._get_config("talker.hidden_size")
        self.num_layers = self._get_config("talker.num_hidden_layers")
        self.codec_eos_token_id = int(self._get_config("talker.codec_eos_token_id"))

    def _get_config(self, path: str) -> Any:
        """Get nested config value."""
        cur = self.config
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"Missing config key: {path}")
            cur = cur[part]
        return cur

    def _load_models(self, providers: list):
        """Загрузка всех ONNX моделей пайплайна."""
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        required_models = {
            "prefill": "talker_prefill.onnx",
            "decode": "talker_decode.onnx",
            "cp": "code_predictor.onnx",
            "vocoder": "vocoder.onnx",
        }

        self.sessions = {}
        for name, filename in required_models.items():
            path = self.model_dir / filename
            if not path.exists():
                # Пробуем альтернативные пути
                alt_path = self.model_dir / "onnx" / filename
                if alt_path.exists():
                    path = alt_path
                else:
                    raise FileNotFoundError(
                        f"Required model {filename} not found in {self.model_dir}. "
                        f"Expected structure: {self.model_dir}/{{{', '.join(required_models.values())}}}"
                    )
            self.sessions[name] = ort.InferenceSession(str(path), sess_options=sess_options, providers=providers)
            LOGGER.info(f"Loaded {name} model from {path}")

    def _load_tokenizer(self):
        """Загрузка BPE токенизатора."""
        from tokenizers import ByteLevelBPETokenizer

        tokenizer_dir = self.model_dir / "tokenizer"
        vocab_path = tokenizer_dir / "vocab.json"
        merges_path = tokenizer_dir / "merges.txt"

        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(f"Tokenizer files not found: {vocab_path} / {merges_path}")

        self.tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
        LOGGER.info("Loaded tokenizer from %s", tokenizer_dir)

    def _load_assets(self):
        """Загрузка эмбеддингов и весов проекций."""
        self.assets = {"config": self.config}

        # Text embedding
        text_emb_path = self.emb_dir / "text_embedding.npy"
        if text_emb_path.exists():
            self.assets["text_embedding"] = np.load(text_emb_path)

        # Talker codec embedding
        talker_codec_path = self.emb_dir / "talker_codec_embedding.npy"
        if talker_codec_path.exists():
            self.assets["talker_codec_embedding"] = np.load(talker_codec_path)

        # Text projection weights
        for key in ["fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"]:
            path = self.emb_dir / f"text_projection_{key}.npy"
            if path.exists():
                self.assets[f"text_projection_{key}"] = np.load(path)

        # Code predictor embeddings
        cp_tables = []
        for i in range(NUM_EMBEDDING_FILES):
            path = self.emb_dir / f"cp_codec_embedding_{i}.npy"
            if path.exists():
                cp_tables.append(np.load(path))
        if cp_tables:
            self.assets["cp_codec_embeddings"] = cp_tables

        LOGGER.info("Loaded assets from %s", self.emb_dir)

    def _encode_plain_text(self, text: str) -> list[int]:
        """Токенизация текста."""
        return self.tokenizer.encode(text).ids

    def _build_custom_voice_prompt_ids(self, text: str, instruct: str | None = None) -> np.ndarray:
        """Построение промпта для кастомного голоса."""
        ids = []

        if instruct:
            user_ids = self._encode_plain_text("user")
            instruct_ids = self._encode_plain_text(instruct)
            ids.extend([IM_START_ID])
            ids.extend(user_ids)
            ids.append(NEWLINE_TOKEN_ID)
            ids.extend(instruct_ids)
            ids.extend([IM_END_ID, NEWLINE_TOKEN_ID])

        text_ids = self._encode_plain_text(text)
        ids.extend([IM_START_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID])
        ids.extend(text_ids)
        ids.extend([IM_END_ID, NEWLINE_TOKEN_ID, IM_START_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID])

        return np.array(ids, dtype=np.int64)

    # === Helper functions from model_assets.py ===

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def _text_embedding_lookup(self, token_id: int) -> np.ndarray:
        return self.assets["text_embedding"][token_id].astype(np.float32, copy=False)

    def _text_projection(self, x_2048: np.ndarray) -> np.ndarray:
        fc1_w = self.assets["text_projection_fc1_weight"].astype(np.float64, copy=False)
        fc1_b = self.assets["text_projection_fc1_bias"].astype(np.float64, copy=False)
        fc2_w = self.assets["text_projection_fc2_weight"].astype(np.float64, copy=False)
        fc2_b = self.assets["text_projection_fc2_bias"].astype(np.float64, copy=False)

        with np.errstate(all='ignore'):
            x = x_2048.astype(np.float64, copy=False)
            x = x @ fc1_w.T + fc1_b
            x = self._gelu(x)
            x = x @ fc2_w.T + fc2_b

        if not np.isfinite(x).all():
            raise RuntimeError("Non-finite values detected in text_projection output")
        return x.astype(np.float32, copy=False)

    def _talker_codec_embedding_lookup(self, token_id: int) -> np.ndarray:
        return self.assets["talker_codec_embedding"][token_id].astype(np.float32, copy=False)

    def _cp_codec_embedding_lookup(self, group_idx_zero_based: int, token_id: int) -> np.ndarray:
        return self.assets["cp_codec_embeddings"][group_idx_zero_based][token_id].astype(np.float32, copy=False)

    def _build_prefill_embedding(
        self,
        token_ids: np.ndarray,
        language: str = "auto",
        speaker_id: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Построение эмбеддингов для prefill."""
        token_ids_list = token_ids.tolist()
        if len(token_ids_list) < 9:
            raise ValueError("Prompt token sequence is too short for prefill builder.")

        role_embeds: list[np.ndarray] = []
        for i in range(3):
            role_embeds.append(self._text_projection(self._text_embedding_lookup(token_ids_list[i])))

        codec_prefix: list[int] = []
        language_ids = self._get_config("language_ids")

        codec_think_id = int(self._get_config("talker.codec_think_id"))
        codec_nothink_id = int(self._get_config("talker.codec_nothink_id"))
        codec_think_bos_id = int(self._get_config("talker.codec_think_bos_id"))
        codec_think_eos_id = int(self._get_config("talker.codec_think_eos_id"))
        codec_pad_id = int(self._get_config("talker.codec_pad_id"))
        codec_bos_id = int(self._get_config("talker.codec_bos_id"))

        if language != "auto":
            if language.lower() not in language_ids:
                raise KeyError(f"Language '{language}' not found in config.language_ids")
            codec_prefix.extend([
                codec_think_id,
                codec_think_bos_id,
                int(language_ids[language.lower()]),
                codec_think_eos_id,
            ])
        else:
            codec_prefix.extend([
                codec_nothink_id,
                codec_think_bos_id,
                codec_think_eos_id,
            ])

        if speaker_id >= 0:
            codec_prefix.append(int(speaker_id))

        codec_prefix.append(codec_pad_id)
        codec_prefix.append(codec_bos_id)

        tts_pad_token_id = int(self._get_config("tts.tts_pad_token_id"))
        tts_bos_token_id = int(self._get_config("tts.tts_bos_token_id"))
        tts_eos_token_id = int(self._get_config("tts.tts_eos_token_id"))

        tts_pad_proj = self._text_projection(self._text_embedding_lookup(tts_pad_token_id))
        tts_bos_proj = self._text_projection(self._text_embedding_lookup(tts_bos_token_id))
        tts_eos_proj = self._text_projection(self._text_embedding_lookup(tts_eos_token_id))

        talker_input_embeds: list[np.ndarray] = []
        codec_prefix_len = len(codec_prefix)

        for i in range(max(0, codec_prefix_len - 2)):
            codec_emb = self._talker_codec_embedding_lookup(codec_prefix[i])
            talker_input_embeds.append((tts_pad_proj + codec_emb).astype(np.float32, copy=False))

        if codec_prefix_len >= 2:
            codec_emb = self._talker_codec_embedding_lookup(codec_prefix[-2])
            talker_input_embeds.append((tts_bos_proj + codec_emb).astype(np.float32, copy=False))

        all_embeds: list[np.ndarray] = []
        all_embeds.extend(role_embeds)
        all_embeds.extend(talker_input_embeds)

        token3_proj = self._text_projection(self._text_embedding_lookup(token_ids_list[3]))
        codec_bos_emb = self._talker_codec_embedding_lookup(codec_bos_id)
        all_embeds.append((token3_proj + codec_bos_emb).astype(np.float32, copy=False))

        trailing_list: list[np.ndarray] = []
        trailing_tokens = token_ids_list[4:-5]
        for tok in trailing_tokens:
            trailing_list.append(self._text_projection(self._text_embedding_lookup(tok)))

        trailing_list.append(tts_eos_proj.astype(np.float32, copy=False))

        inputs_embeds = np.stack(all_embeds, axis=0)[None, :, :]
        trailing_text_hidden = np.stack(trailing_list, axis=0)

        LOGGER.info(
            "Prefill embedding built: inputs_embeds=%s trailing_text_hidden=%s",
            inputs_embeds.shape,
            trailing_text_hidden.shape,
        )

        assert inputs_embeds.shape[-1] == self.hidden_size
        assert trailing_text_hidden.shape[-1] == self.hidden_size

        return inputs_embeds.astype(np.float32, copy=False), trailing_text_hidden.astype(np.float32, copy=False)

    # === Sampling functions from inference_engine.py ===

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        exponents = np.exp(x)
        sum_of_exponents = np.sum(exponents, axis=axis, keepdims=True)
        sum_of_exponents = np.where(sum_of_exponents == 0, 1.0, sum_of_exponents)
        return exponents / sum_of_exponents

    def _sample_talker_token(
        self,
        logits: np.ndarray,
        previous_tokens: list[int],
        step: int
    ) -> int:
        talker_vocab_size = int(self._get_config("talker.vocab_size"))
        cp_vocab_size = int(self._get_config("code_predictor.vocab_size"))

        x = logits.reshape(-1)[-talker_vocab_size:].astype(np.float32, copy=True)

        for token in previous_tokens:
            if 0 <= token < len(x):
                if x[token] > 0:
                    x[token] /= self.repetition_penalty
                else:
                    x[token] *= self.repetition_penalty

        for i in range(cp_vocab_size, talker_vocab_size):
            if i != self.codec_eos_token_id:
                x[i] = -np.inf

        if step < self.min_new_tokens and 0 <= self.codec_eos_token_id < len(x):
            x[self.codec_eos_token_id] = -np.inf

        effective_temp = self.temperature if step >= 4 else min(self.temperature, 0.1)
        if effective_temp > 0:
            x = x / effective_temp

        if self.top_k > 0 and self.top_k < len(x):
            idx = np.argpartition(-x, self.top_k)[:self.top_k]
            mask = np.full_like(x, -np.inf)
            mask[idx] = x[idx]
            x = mask

        probs = self._softmax(x)
        return int(np.random.choice(len(probs), p=probs))

    def _sample_cp_token(self, logits: np.ndarray) -> int:
        cp_vocab_size = int(self._get_config("code_predictor.vocab_size"))
        x = logits.reshape(-1)[-cp_vocab_size:].astype(np.float32, copy=True)

        if self.top_k > 0 and self.top_k < len(x):
            idx = np.argpartition(-x, self.top_k)[:self.top_k]
            mask = np.full_like(x, -np.inf)
            mask[idx] = x[idx]
            x = mask

        if self.temperature > 0:
            x = x / self.temperature

        probs = self._softmax(x)
        return int(np.random.choice(len(probs), p=probs))

    def _stack_prefill_kv(self, prefill_map: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        keys = []
        values = []
        for layer in range(self.num_layers):
            keys.append(prefill_map[f"present_key_{layer}"])
            values.append(prefill_map[f"present_value_{layer}"])
        return np.stack(keys, axis=0).astype(np.float32, copy=False), np.stack(values, axis=0).astype(np.float32, copy=False)

    def _build_position_ids_3d(self, step_position: int) -> np.ndarray:
        return np.array([[[step_position]], [[step_position]], [[step_position]]], dtype=np.int64)

    def _run_code_predictor_groups(self, last_hidden: np.ndarray, group0_token: int) -> list[int]:
        """Запуск code predictor для генерации 15 кодов (groups 1-15)."""
        cp_session = self.sessions["cp"]
        cp_num_layers = int(self._get_config("code_predictor.num_hidden_layers"))
        cp_num_kv_heads = int(self._get_config("code_predictor.num_key_value_heads"))
        cp_head_dim = int(self._get_config("code_predictor.head_dim"))

        group0_embed = self._talker_codec_embedding_lookup(group0_token)
        codes: list[int] = []
        cp_past_keys = np.empty((cp_num_layers, 1, cp_num_kv_heads, 0, cp_head_dim), dtype=np.float32)
        cp_past_values = np.empty((cp_num_layers, 1, cp_num_kv_heads, 0, cp_head_dim), dtype=np.float32)
        prev_cp_token = None

        for group_idx in range(1, 16):
            if group_idx == 1:
                inputs_embeds = np.stack([last_hidden, group0_embed], axis=0)[None, :, :].astype(np.float32)
            else:
                assert prev_cp_token is not None
                prev_cp_embed = self._cp_codec_embedding_lookup(group_idx - 2, prev_cp_token)
                inputs_embeds = prev_cp_embed[None, None, :].astype(np.float32)

            generation_steps = np.array([group_idx - 1], dtype=np.int64)

            feed = {
                "inputs_embeds": inputs_embeds,
                "generation_steps": generation_steps,
                "past_keys": cp_past_keys,
                "past_values": cp_past_values,
            }

            output_names = [x.name for x in cp_session.get_outputs()]
            outputs = cp_session.run(output_names, feed)
            cp_map = dict(zip(output_names, outputs))

            cp_logits = cp_map["logits"]
            cp_token = self._sample_cp_token(cp_logits)
            codes.append(cp_token)
            prev_cp_token = cp_token

            cp_past_keys = cp_map["present_keys"].astype(np.float32, copy=False)
            cp_past_values = cp_map["present_values"].astype(np.float32, copy=False)

        return codes

    def _generate_codes(self, token_ids: np.ndarray, language: str, speaker_id: int) -> np.ndarray:
        """Основная генерация кодеков: prefill + decode loop + code predictor."""
        inputs_embeds, trailing_text_hidden = self._build_prefill_embedding(
            token_ids=token_ids,
            language=language,
            speaker_id=speaker_id
        )

        prefill_len = inputs_embeds.shape[1]
        attention_mask = np.ones((1, prefill_len), dtype=np.int64)
        position_ids = np.zeros((3, 1, prefill_len), dtype=np.int64)

        for ax in range(3):
            position_ids[ax, 0, :] = np.arange(prefill_len, dtype=np.int64)

        LOGGER.info("Running talker_prefill")
        prefill_session = self.sessions["prefill"]
        prefill_output_names = [x.name for x in prefill_session.get_outputs()]

        prefill_outputs = prefill_session.run(
            output_names=prefill_output_names,
            input_feed={
                "inputs_embeds": inputs_embeds.astype(np.float32),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )

        prefill_map = dict(zip(prefill_output_names, prefill_outputs))

        logits = prefill_map["logits"].astype(np.float32, copy=False)
        hidden_states = prefill_map["hidden_states"].astype(np.float32, copy=False)
        past_keys, past_values = self._stack_prefill_kv(prefill_map)

        tts_pad_token_id = int(self._get_config("tts.tts_pad_token_id"))
        tts_pad_proj = self._text_projection(self._text_embedding_lookup(tts_pad_token_id))
        generated_codes = []
        generated_group0_tokens = []

        for step in range(self.max_new_tokens):
            group0_token = self._sample_talker_token(
                logits=logits,
                previous_tokens=generated_group0_tokens,
                step=step,
            )

            LOGGER.debug("step=%d group0_token=%d", step, group0_token)
            logits_min, logits_max = np.min(logits), np.max(logits)
            logits_mean = np.mean(logits)
            #print(f"[step {step}] logits range: [{logits_min:.2f}, {logits_max:.2f}] mean: {logits_mean:.2f}")

            if group0_token == self.codec_eos_token_id:
                LOGGER.info("Reached codec EOS at step %d", step)
                break

            generated_group0_tokens.append(group0_token)
            last_hidden = hidden_states[0, -1, :].astype(np.float32, copy=False)

            # Code predictor для groups 1-15
            cp_codes = self._run_code_predictor_groups(last_hidden, group0_token)

            codes_this_step = [group0_token] + cp_codes
            generated_codes.append(codes_this_step)

            # Построение следующего входа для decode
            next_input = self._talker_codec_embedding_lookup(group0_token).copy()

            for g in range(1, 16):
                cp_embed = self._cp_codec_embedding_lookup(g - 1, codes_this_step[g])
                next_input[: cp_embed.shape[0]] += cp_embed

            if step < trailing_text_hidden.shape[0]:
                next_input += trailing_text_hidden[step]
            else:
                next_input += tts_pad_proj

            next_input_embeds = next_input[None, None, :].astype(np.float32)
            new_len = prefill_len + step + 1
            decode_attention_mask = np.ones((1, new_len), dtype=np.int64)
            decode_position_ids = self._build_position_ids_3d(prefill_len + step)

            LOGGER.info("Running talker_decode step %d/%d", step + 1, self.max_new_tokens)

            decode_session = self.sessions["decode"]
            decode_output_names = [x.name for x in decode_session.get_outputs()]

            decode_outputs = decode_session.run(
                output_names=decode_output_names,
                input_feed={
                    "inputs_embeds": next_input_embeds,
                    "attention_mask": decode_attention_mask,
                    "position_ids": decode_position_ids,
                    "past_keys": past_keys.astype(np.float32),
                    "past_values": past_values.astype(np.float32),
                }
            )

            decode_map = dict(zip(decode_output_names, decode_outputs))

            logits = decode_map["logits"].astype(np.float32, copy=False)
            hidden_states = decode_map["hidden_states"].astype(np.float32, copy=False)
            past_keys = decode_map["present_keys"].astype(np.float32, copy=False)
            past_values = decode_map["present_values"].astype(np.float32, copy=False)

        T = len(generated_codes)
        LOGGER.info("Generated timesteps: %d", T)

        result = np.zeros((1, 16, T), dtype=np.int64)
        for t in range(T):
            for g in range(16):
                result[0, g, t] = generated_codes[t][g]

        return result

    def _run_vocoder(self, codes: np.ndarray) -> np.ndarray:
        """Декодирование кодеков в аудио через vocoder.onnx."""
        vocoder_session = self.sessions["vocoder"]
        vocoder_input_name = vocoder_session.get_inputs()[0].name
        LOGGER.info("Running vocoder with codes shape=%s", codes.shape)
        output_names = [x.name for x in vocoder_session.get_outputs()]
        outputs = vocoder_session.run(output_names, {vocoder_input_name: codes.astype(np.int64, copy=False)})
        waveform = outputs[0]
        LOGGER.info("Vocoder waveform shape=%s dtype=%s", waveform.shape, waveform.dtype)
        return waveform

    def _normalize_audio(self, audio: Any) -> np.ndarray:
        """Нормализация аудио в float32 [-1, 1]."""
        if isinstance(audio, list):
            audio = np.asarray(audio, dtype=np.float32)
        elif isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
        else:
            audio = np.array(audio, dtype=np.float32)

        # Clip to [-1, 1] для безопасности
        return np.clip(audio, -1.0, 1.0)

    def generate(self, text: str, language: str = "auto") -> tuple[np.ndarray, int]:
        """Generate audio for a single text sample."""
        # Построение промпта
        token_ids = self._build_custom_voice_prompt_ids(text)

        # Генерация кодеков
        codes = self._generate_codes(token_ids, language, self.speaker_id)

        # Вокодинг
        waveform = self._run_vocoder(codes)

        # Нормализация
        waveform = self._normalize_audio(waveform)

        return waveform, self.sample_rate

    def generate_batch(self, texts: list[str], languages: list[str] | None = None) -> tuple[list[np.ndarray], int]:
        """Generate audio for a batch of texts."""
        if languages is None:
            languages = ["auto"] * len(texts)

        audios = []
        for text, lang in zip(texts, languages):
            waveform, sr = self.generate(text, lang)
            audios.append(waveform)

        return audios, sr

    def save_audio(self, waveform: np.ndarray, output_path: str | Path, subtype: str = "PCM_16"):
        import soundfile as sf
        audio = np.squeeze(waveform)          # убираем все лишние размерности
        if audio.ndim == 0:
            raise ValueError(f"Waveform is scalar after squeeze, original shape: {waveform.shape}")
        audio = audio.astype(np.float32)
        sf.write(str(output_path), audio, self.sample_rate, subtype=subtype)