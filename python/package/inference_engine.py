import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

from package.model_assets import AssetManager, ProjectionUtils, PrefillEmbeddingBuilder
from package.utils import save_outputs

LOGGER = logging.getLogger(__name__)


class Sampler:
    @staticmethod
    def apply_repetition_penalty(logits: np.ndarray, previous_tokens: list[int], penalty: float) -> np.ndarray:
        x = logits.copy()
        for token in previous_tokens:
            if 0 <= token < len(x):
                if x[token] > 0:
                    x[token] /= penalty
                else:
                    x[token] *= penalty
        return x

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, keepdims=True)
        exponents = np.exp(x)
        return exponents / np.sum(exponents, keepdims=True)
        

    @staticmethod
    def apply_top_k_temperature(logits: np.ndarray, temperature: float, top_k: int) -> np.ndarray:
        x = logits.copy()
        
        if temperature > 0:
            x = x / temperature
        
        if top_k > 0 and top_k < len(x):
            idx = np.argpartition(-x, top_k)[:top_k]
            mask = np.full_like(x, -np.inf)
            mask[idx] = x[idx]
            x = mask
            
        return x

    def sample(
        self, 
        logits: np.ndarray,
        previous_tokens: list[int], 
        temperature: float,
        top_k: int,
        penalty: float
        ) -> int:
        x = self.apply_repetition_penalty(logits, previous_tokens, penalty)
        x = self.apply_top_k_temperature(x, temperature, top_k)
        probs = self.softmax(x)
        return int(np.random.choice(len(probs), p=probs))


@dataclass
class GenerationState:
    past_keys: np.ndarray | None = None
    past_values: np.ndarray | None = None
    generated_group0_tokens: list[int] = field(default_factory=list)
    generated_codes: list[list[int]] = field(default_factory=list)
    step: int = 0


class TalkerGenerator:
    def __init__(
        self,
        prefill_session: ort.InferenceSession,
        decode_session: ort.InferenceSession,
        cp_session: ort.InferenceSession,
        assets: AssetManager,
        sampler: Sampler | None = None
    ):
        self.prefill_session = prefill_session
        self.decode_session = decode_session
        self.cp_session = cp_session
        self.assets = assets
        self.config = assets.config
        self.sampler = sampler or Sampler()
        
        self.num_layers = int(self.config.get("talker.num_hidden_layers"))
        self.talker_vocab_size = int(self.config.get("talker.vocab_size"))
        self.cp_vocab_size = int(self.config.get("code_predictor.vocab_size"))
        self.codec_eos_token_id = int(self.config.get("talker.codec_eos_token_id"))
        self.cp_num_layers = int(self.config.get("code_predictor.num_hidden_layers"))
        self.cp_num_kv_heads = int(self.config.get("code_predictor.num_key_value_heads"))
        self.cp_head_dim = int(self.config.get("code_predictor.head_dim"))

        tts_pad_id = int(self.config.get("tts.tts_pad_token_id"))
        tts_bos_id = int(self.config.get("tts.tts_bos_token_id"))
        tts_eos_id = int(self.config.get("tts.tts_eos_token_id"))
        
        self._tts_pad_proj = self._project_token(tts_pad_id)
        self._tts_bos_proj = self._project_token(tts_bos_id)
        self._tts_eos_proj = self._project_token(tts_eos_id)

    def _project_token(self, token_id: int) -> np.ndarray:
        emb = self.assets.get("text_embedding")[token_id].astype(np.float32, copy=False)
        return ProjectionUtils.project_text(emb, self.assets._data)

    def _get_talker_emb(self, token_id: int) -> np.ndarray:
        return self.assets.get("talker_codec_embedding")[token_id].astype(np.float32, copy=False)

    def _get_cp_emb(self, group_idx: int, token_id: int) -> np.ndarray:
        table = self.assets.get("cp_codec_embeddings")[group_idx]
        return table[token_id].astype(np.float32, copy=False)

    def run_prefill(
        self,
        token_ids: np.ndarray,
        language: str,
        speaker_id: int,
        save_dir: Path | None = None
    ) -> tuple[np.ndarray, np.ndarray, GenerationState]:
        builder = PrefillEmbeddingBuilder(self.assets)
        
        inputs_embeds, trailing_text_hidden = builder.build(
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
        output_names = [x.name for x in self.prefill_session.get_outputs()]
        
        outputs = self.prefill_session.run(
            output_names=output_names,
            input_feed={
                "inputs_embeds": inputs_embeds.astype(np.float32),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        
        prefill_map = dict(zip(output_names, outputs))
        
        if save_dir:
            save_outputs(save_dir, "prefill", output_names, outputs)

        logits = prefill_map["logits"].astype(np.float32, copy=False)
        hidden_states = prefill_map["hidden_states"].astype(np.float32, copy=False)
        keys = [prefill_map[f"present_key_{i}"] for i in range(self.num_layers)]
        values = [prefill_map[f"present_value_{i}"] for i in range(self.num_layers)]
        
        state = GenerationState(
            past_keys=np.stack(keys, axis=0).astype(np.float32, copy=False),
            past_values=np.stack(values, axis=0).astype(np.float32, copy=False),
            trailing_text_hidden=trailing_text_hidden # Сохраняем для использования в decode
        )
        
        return logits, hidden_states, state

    def run_code_predictor_step(
        self,
        last_hidden: np.ndarray,
        group0_token: int,
        state: GenerationState,
        temperature: float,
        top_k: int,
        timestep_idx: int,
        save_dir: Path | None = None
    ) -> list[int]:
        group0_embed = self._get_talker_emb(group0_token)
        codes = []
    
        cp_past_keys = np.empty(
            (self.cp_num_layers, 1, self.cp_num_kv_heads, 0, self.cp_head_dim), 
            dtype=np.float32
        )
        
        cp_past_values = np.empty(
            (self.cp_num_layers, 1, self.cp_num_kv_heads, 0, self.cp_head_dim), 
            dtype=np.float32
        )
        
        prev_cp_token = None
        output_names = [x.name for x in self.cp_session.get_outputs()]

        for group_idx in range(1, 16):
            if group_idx == 1:
                inputs_embeds = np.stack([last_hidden, group0_embed], axis=0)[None, :, :].astype(np.float32)
            else:
                assert prev_cp_token is not None
                prev_cp_embed = self._get_cp_emb(group_idx - 2, prev_cp_token)
                inputs_embeds = prev_cp_embed[None, None, :].astype(np.float32)

            generation_steps = np.array([group_idx - 1], dtype=np.int64)
            
            feed = {
                "inputs_embeds": inputs_embeds,
                "generation_steps": generation_steps,
                "past_keys": cp_past_keys,
                "past_values": cp_past_values,
            }

            outputs = self.cp_session.run(output_names, feed)
            cp_map = dict(zip(output_names, outputs))
            cp_logits = cp_map["logits"]
            cp_token = self.sampler.sample(cp_logits, [], temperature, top_k, penalty=1.0)
            codes.append(cp_token)
            prev_cp_token = cp_token
            cp_past_keys = cp_map["present_keys"].astype(np.float32, copy=False)
            cp_past_values = cp_map["present_values"].astype(np.float32, copy=False)

            if save_dir:
                save_outputs(save_dir, f"cp_t{timestep_idx}_g{group_idx}", output_names, outputs)

        return codes

    def generate(
        self,
        token_ids: np.ndarray,
        language: str = "auto",
        speaker_id: int = -1,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        min_new_tokens: int = 0,
        save_dir: Path | None = None
    ) -> np.ndarray:
        logits, hidden_states, state = self.run_prefill(
            token_ids, language, speaker_id, save_dir
        )
    
        prefill_len = logits.shape[1]
        for step in range(max_new_tokens):
            group0_token = self.sampler.sample(
                logits=logits.reshape(-1)[-self.talker_vocab_size:],
                previous_tokens=state.generated_group0_tokens,
                temperature=temperature,
                top_k=top_k,
                penalty=repetition_penalty
            )
            
            if step < min_new_tokens:
                if group0_token == self.codec_eos_token_id:
                    pass 
            
            if group0_token == self.codec_eos_token_id and step >= min_new_tokens:
                LOGGER.info("Reached codec EOS at step %d", step)
                break

            state.generated_group0_tokens.append(group0_token)
            last_hidden = hidden_states[0, -1, :].astype(np.float32, copy=False)

            cp_codes = self.run_code_predictor_step(
                last_hidden=last_hidden,
                group0_token=group0_token,
                state=state,
                temperature=temperature,
                top_k=top_k,
                timestep_idx=step,
                save_dir=save_dir
            )

            codes_this_step = [group0_token] + cp_codes
            state.generated_codes.append(codes_this_step)
            next_input = self._get_talker_emb(group0_token).copy()
            
            for g in range(1, 16):
                cp_embed = self._get_cp_emb(g - 1, codes_this_step[g])
                if next_input.shape[0] >= cp_embed.shape[0]:
                     next_input[:cp_embed.shape[0]] += cp_embed
                else:
                     next_input += cp_embed 

            if step < state.trailing_text_hidden.shape[0]:
                next_input += state.trailing_text_hidden[step]
            else:
                next_input += self._tts_pad_proj

            next_input_embeds = next_input[None, None, :].astype(np.float32)
            new_len = prefill_len + step + 1
            decode_attention_mask = np.ones((1, new_len), dtype=np.int64)
            decode_position_ids = np.array([[[new_len - 1]]] * 3, dtype=np.int64) # [[step]] * 3
            decode_output_names = [x.name for x in self.decode_session.get_outputs()]
            
            decode_outputs = self.decode_session.run(
                output_names=decode_output_names,
                input_feed={
                    "inputs_embeds": next_input_embeds,
                    "attention_mask": decode_attention_mask,
                    "position_ids": decode_position_ids,
                    "past_keys": state.past_keys,
                    "past_values": state.past_values,
                }
            )
            
            decode_map = dict(zip(decode_output_names, decode_outputs))
            
            if save_dir:
                save_outputs(save_dir, f"decode_step_{step + 1}", decode_output_names, decode_outputs)

            logits = decode_map["logits"].astype(np.float32, copy=False)
            hidden_states = decode_map["hidden_states"].astype(np.float32, copy=False)
            state.past_keys = decode_map["present_keys"].astype(np.float32, copy=False)
            state.past_values = decode_map["present_values"].astype(np.float32, copy=False)
            state.step += 1

        T = len(state.generated_codes)
        if T == 0:
            return np.zeros((1, 16, 0), dtype=np.int64)
            
        result = np.zeros((1, 16, T), dtype=np.int64)
        for t, codes in enumerate(state.generated_codes):
            result[0, :, t] = codes

        LOGGER.info("Generation finished. Timesteps: %d", T)
        return result


class Vocoder:
    def __init__(self, vocoder_session: ort.InferenceSession):
        self.vocoder_session = vocoder_session
        
    def __call__(self, codes: np.ndarray) -> np.ndarray:
        vocoder_input_name = self.vocoder_session.get_inputs()[0].name
        output_names = [x.name for x in self.vocoder_session.get_outputs()]
        outputs = self.vocoder_session.run(output_names, {vocoder_input_name: codes.astype(np.int64, copy=False)})
        return outputs[0]
