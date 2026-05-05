import time
import logging
from pathlib import Path
from typing import Optional, Union, Any
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm
import onnxruntime as ort

from package.model_assets import AssetManager, ProjectionUtils, PrefillEmbeddingBuilder, Tokenizer
from package.utils import ModelSessions, load_yaml, resolve_path, get_param, load_config


NUM_GROUPS = 16
LOGGER = logging.getLogger(__name__)


class Sampler:
    @staticmethod
    def apply_repetition_penalty(
        logits: np.ndarray,
        previous_tokens: list[int],
        penalty: float
    ) -> np.ndarray:
        x = logits.copy()
        for prev_token in previous_tokens:
            if 0 <= prev_token < len(x):
                if x[prev_token] > 0:
                    x[prev_token] /= penalty
                else:
                    x[prev_token] *= penalty
        return x
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        exponents = np.exp(x)
        sum_of_exponents = np.sum(exponents, axis=axis, keepdims=True)
        sum_of_exponents = np.where(sum_of_exponents == 0, 1.0, sum_of_exponents)
        return exponents / sum_of_exponents
    
    @staticmethod
    def scale_by_temperature(x: np.ndarray, temperature: float) -> np.ndarray:
        if temperature > 0:
            x /= temperature
        return x
    
    @staticmethod
    def apply_top_k(
        logits: np.ndarray,
        top_k: int 
    ) -> np.ndarray:
        x = logits.copy()
        if 0 < top_k < len(x):
            index = np.argpartition(-x, top_k)[:top_k]
            mask = np.full_like(x, -np.inf)
            mask[index] = x[index]
            x = mask
            
        return x
    
    def sample_talker_token(
        self,
        logits: np.ndarray,
        previous_tokens: list[int],
        temperature: float,
        top_k: int,
        penalty: float
    ) -> int:
        x = self.apply_repetition_penalty(logits, previous_tokens, penalty)
        x = self.scale_by_temperature(x, temperature)
        x = self.apply_top_k(x, top_k)
        probs = self.softmax(x)
        return int(np.random.choice(len(probs), p=probs))
    
    def sample_cp_token(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: int
    ) -> int:
        x = self.apply_top_k(logits, top_k)
        x = self.scale_by_temperature(x, temperature)
        probs = self.softmax(x)
        return int(np.random.choice(len(probs), p=probs))
    

@dataclass
class GenerationState:
    past_keys: Optional[np.ndarray] = None
    past_values: Optional[np.ndarray] = None
    trailing_text_hidden: Optional[np.ndarray] = None
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
        sampler: Optional[Sampler] = None
    ):
        self.config = load_config(assets.config_path)
        self.prefill_session: ort.InferenceSession = prefill_session
        self.decode_session: ort.InferenceSession = decode_session
        self.cp_session: ort.InferenceSession = cp_session
        self.assets: AssetManager = assets
        self.sampler: Sampler = sampler or Sampler()
        self.num_layers: int = int(self.config["talker"]["num_hidden_layers"])
        self.talker_vocab_size: int = int(self.config["talker"]["vocab_size"])
        self.cp_vocab_size: int = int(self.config["code_predictor"]["vocab_size"])
        self.codec_eos_token_id: int = int(self.config["talker"]["codec_eos_token_id"])
        self.cp_num_layers: int = int(self.config["code_predictor"]["num_hidden_layers"])
        self.cp_num_kv_heads: int = int(self.config["code_predictor"]["num_key_value_heads"])
        self.cp_head_dim: int = int(self.config["code_predictor"]["head_dim"])
        
        tts_pad_id = int(self.config["tts"]["tts_pad_token_id"])
        tts_bos_id = int(self.config["tts"]["tts_bos_token_id"])
        tts_eos_id = int(self.config["tts"]["tts_eos_token_id"])
        
        self.tts_pad_proj: np.ndarray = self._project_token(tts_pad_id)
        self.tts_bos_proj: np.ndarray = self._project_token(tts_bos_id)
        self.tts_eos_proj: np.ndarray = self._project_token(tts_eos_id)
        
    def _project_token(self, token_idx: int) -> np.ndarray:
        embedding = self.assets.get("text_embedding")[token_idx].astype(np.float32, copy=False)
        return ProjectionUtils.project_text(embedding, self.assets.data)
    
    def _get_talker_embedding(self, token_idx: int) -> np.ndarray:
        return self.assets.get("talker_codec_embedding")[token_idx].astype(np.float32, copy=False)
    
    def _get_cp_embedding(self, group_idx: int, token_idx: int) -> np.ndarray:
        cp_table = self.assets.get("cp_codec_embeddings")[group_idx]
        return cp_table[token_idx].astype(np.float32, copy=False)
    
    def run_prefill(
        self,
        token_ids: np.ndarray,
        language: str,
        speaker_id: int
    ) -> tuple[np.ndarray, np.ndarray, GenerationState]:
        builder = PrefillEmbeddingBuilder(self.assets)
        input_embeddings, trailing_text_hidden = builder.build(
            token_ids=token_ids,
            language=language,
            speaker_id=speaker_id
        )
        
        prefill_len = input_embeddings.shape[1]
        attention_mask = np.ones((1, prefill_len), dtype=np.int64)
        position_ids = np.zeros((3, 1, prefill_len), dtype=np.int64)
        
        for ax in range(3):
            position_ids[ax, 0, :] = np.arange(prefill_len, dtype=np.int64)

        output_names = [x.name for x in self.prefill_session.get_outputs()]
        outputs = self.prefill_session.run(
            output_names=output_names,
            input_feed={
                "inputs_embeds": input_embeddings.astype(np.float32),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        
        prefill_map = dict(zip(output_names, outputs))
        logits = prefill_map["logits"].astype(np.float32, copy=False)
        hidden_states = prefill_map["hidden_states"].astype(np.float32, copy=False)
        keys = [prefill_map[f"present_key_{i}"] for i in range(self.num_layers)]
        values = [prefill_map[f"present_value_{i}"] for i in range(self.num_layers)]
        
        state = GenerationState(
            past_keys=np.stack(keys, axis=0).astype(np.float32, copy=False),
            past_values=np.stack(values, axis=0).astype(np.float32, copy=False),
            trailing_text_hidden=trailing_text_hidden
        )
        
        return logits, hidden_states, state
    
    def run_code_predictor_step(
        self,
        last_hidden: np.ndarray,
        group0_token: int,
        temperature: float,
        top_k: int
    ) -> list[int]:
        group0_embedding = self._get_talker_embedding(group0_token)
        codes = []
        cp_past_keys = np.empty(
            shape=(self.cp_num_layers, 1, self.cp_num_kv_heads, 0, self.cp_head_dim),
            dtype=np.float32
        )
        
        cp_past_values = np.empty(
            shape=(self.cp_num_layers, 1, self.cp_num_kv_heads, 0, self.cp_head_dim), 
            dtype=np.float32
        )
        
        prev_cp_token = None
        output_names = [x.name for x in self.cp_session.get_outputs()]
        
        for group_idx in range(1, NUM_GROUPS):
            if group_idx == 1:
                input_embeddings = np.stack([last_hidden, group0_embedding], axis=0)[None, :, :].astype(np.float32)
            else:
                if prev_cp_token is None:
                    raise ValueError(f"Code predictor catched a None prev_cp_token")
                prev_cp_embedding = self._get_cp_embedding(group_idx - 2, prev_cp_token)
                input_embeddings = prev_cp_embedding[None, None, :].astype(np.float32)
                
            generation_steps = np.array([group_idx - 1], dtype=np.int64)
            
            input_feed = {
                "inputs_embeds": input_embeddings,
                "generation_steps": generation_steps,
                "past_keys": cp_past_keys,
                "past_values": cp_past_values
            }
            
            outputs = self.cp_session.run(
                output_names=output_names,
                input_feed=input_feed
            )
            
            cp_map = dict(zip(output_names, outputs))
            cp_logits = cp_map["logits"]
            cp_token = self.sampler.sample_cp_token(
                logits=cp_logits.reshape(-1)[-self.cp_vocab_size:].astype(np.float32, copy=True),
                temperature=temperature,
                top_k=top_k,
            )
            codes.append(cp_token)
            prev_cp_token = cp_token
            cp_past_keys = cp_map["present_keys"].astype(np.float32, copy=False)
            cp_past_values = cp_map["present_values"].astype(np.float32, copy=False)
            
        return codes
    
    def generate(
        self,
        token_ids: np.ndarray,
        language: str = "auto",
        speaker_id: int = -1,
        max_new_tokens: int = 512,
        min_new_tokens: int = 0,
        temperature: float = 0.8,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        show_progress: bool = True
    ) -> np.ndarray:
        logits, hidden_states, state = self.run_prefill(
            token_ids=token_ids,
            language=language,
            speaker_id=speaker_id
        )
        
        if show_progress:
            pbar = tqdm(total=max_new_tokens, desc="Prefill done", unit='step', leave=False)
        
        try:
            for step in range(max_new_tokens):
                if show_progress:
                    pbar.set_description(f"Generation Step {step + 1}/{max_new_tokens}")
                    
                current_logits = logits[0, -1, :] 
                group0_token = self.sampler.sample_talker_token(
                    logits=current_logits,
                    previous_tokens=state.generated_group0_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    penalty=repetition_penalty
                )
                
                if step < min_new_tokens:
                    if group0_token == self.codec_eos_token_id:
                        pass 
                
                if group0_token == self.codec_eos_token_id and step >= min_new_tokens:
                    if show_progress:
                        pbar.set_description("EOS Reached")
                    break

                state.generated_group0_tokens.append(group0_token)
                last_hidden = hidden_states[0, -1, :].astype(np.float32, copy=False)
                cp_codes = self.run_code_predictor_step(
                    last_hidden=last_hidden,
                    group0_token=group0_token,
                    temperature=temperature,
                    top_k=top_k
                )

                current_codes = [group0_token] + cp_codes
                state.generated_codes.append(current_codes)
                next_input = self._get_talker_embedding(group0_token).copy()
                
                for g in range(1, NUM_GROUPS):
                    cp_embedding = self._get_cp_embedding(g - 1, current_codes[g])
                    if next_input.shape[0] >= cp_embedding.shape[0]:
                        next_input[:cp_embedding.shape[0]] += cp_embedding
                    else:
                        next_input += cp_embedding 

                if step < state.trailing_text_hidden.shape[0]:
                    next_input += state.trailing_text_hidden[step]
                else:
                    next_input += self.tts_pad_proj

                next_input_embeds = next_input[None, None, :].astype(np.float32)
                cache_len = state.past_keys.shape[3]
                new_len = cache_len + 1
                decode_attention_mask = np.ones((1, new_len), dtype=np.int64)
                current_pos = cache_len 
                decode_position_ids = np.full((3, 1, 1), current_pos, dtype=np.int64)
                decode_output_names = [x.name for x in self.decode_session.get_outputs()]

                decode_outputs = self.decode_session.run(
                    output_names=decode_output_names,
                    input_feed={
                        "inputs_embeds": next_input_embeds,
                        "attention_mask": decode_attention_mask,
                        "position_ids": decode_position_ids,
                        "past_keys": state.past_keys,
                        "past_values": state.past_values
                    }
                )
                
                decode_map = dict(zip(decode_output_names, decode_outputs))
                logits = decode_map["logits"].astype(np.float32, copy=False)
                hidden_states = decode_map["hidden_states"].astype(np.float32, copy=False)
                state.past_keys = decode_map["present_keys"].astype(np.float32, copy=False)
                state.past_values = decode_map["present_values"].astype(np.float32, copy=False)
                state.step += 1
                
                if show_progress:
                    pbar.update(1)
        finally:
            if show_progress:
                pbar.close()

        T = len(state.generated_codes)
        if T == 0:
            return np.zeros((1, NUM_GROUPS, 0), dtype=np.int64)
            
        result = np.zeros((1, NUM_GROUPS, T), dtype=np.int64)
        for t, codes in enumerate(state.generated_codes):
            result[0, :, t] = codes

        return result
    
    
class Vocoder:
    def __init__(self, vocoder_session: ort.InferenceSession):
        self.vocoder_session: ort.InferenceSession = vocoder_session
        
    def __call__(self, codes: np.ndarray) -> np.ndarray:
        vocoder_input_name = self.vocoder_session.get_inputs()[0].name
        output_names = [x.name for x in self.vocoder_session.get_outputs()]
        outputs = self.vocoder_session.run(output_names, {vocoder_input_name: codes.astype(np.int64, copy=False)})
        return outputs[0]


class QwenTTSPipeline:
    def __init__(self, config_path: Path, show_progress: bool = True):
        self.config: dict[str, Any] = load_yaml(config_path)
        self.paths: dict[Path, Path] = {path: resolve_path(file) for path, file in self.config['paths'].items()}
        self.params: dict[str, Union[str, float, int]] = self.config.get("inference_params", {})
        self.show_progress: bool = show_progress
        self.sessions: ModelSessions = None
        self.assets: AssetManager = None
        self.tokenizer: Tokenizer = None
        
        initial_steps = [
            ("Loading ONNX sessions", lambda: ModelSessions.load(self.paths)),
            ("Loading Assets", lambda: AssetManager(embedding_dir=self.paths["embeddings"])),
            ("Loading Tokenizer", lambda: Tokenizer(tokenizer_dir=self.paths["tokenizer"]))
        ]
        
        with tqdm(total=len(initial_steps), desc="Initializing Pipeline", unit="component", leave=False) as pbar:
            for desc, loader_func in initial_steps:
                pbar.set_description(desc)
                result = loader_func()
                if desc == "Loading ONNX sessions": 
                    self.sessions = result
                elif desc == "Loading Assets": 
                    self.assets = result
                elif desc == "Loading Tokenizer": 
                    self.tokenizer = result
                    
                pbar.update(1)
        
        LOGGER.info("Initializing Talking Generator...")
        self.generator = TalkerGenerator(
            prefill_session=self.sessions.prefill,
            decode_session=self.sessions.decode,
            cp_session=self.sessions.cp,
            assets=self.assets
        )
        
        LOGGER.info("Initializing Vocoder...")
        self.vocoder = Vocoder(self.sessions.vocoder)
        LOGGER.info("Pipeline initialized successfully")
        
    def __call__(
        self,
        text: str,
        instruct: Optional[str] = None
    ) -> np.ndarray:
        start_generation_time = time.perf_counter()
        token_ids = self.tokenizer.build_custom_voice_prompt_ids(
            text=text,
            instruct_prompt=instruct
        )
        
        codes = self.generator.generate(
            token_ids=token_ids,
            language=get_param(self.config["inference_params"], "language"),
            speaker_id=get_param(self.config["inference_params"], "max_new_tokens"),
            max_new_tokens=get_param(self.config["inference_params"], "max_new_tokens"),
            min_new_tokens=get_param(self.config["inference_params"], "min_new_tokens"),
            temperature=get_param(self.config["inference_params"], "temperature"),
            top_k=get_param(self.config["inference_params"], "top_k"),
            repetition_penalty=get_param(self.config["inference_params"], "repetition_penalty")
        )

        waveform = self.vocoder(codes)
        LOGGER.info("Pipeline finished successfully! Total time: %.2f s", time.perf_counter() - start_generation_time)
        return waveform
