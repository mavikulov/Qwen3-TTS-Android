import json
import logging
from typing import Any
from pathlib import Path

import numpy as np
from tokenizers import ByteLevelBPETokenizer

from special_tokens import special_tokens


NUM_EMBEDDING_FILES = 15
LOGGER = logging.getLogger(__name__)


class ConfigHelper:
    def __init__(self, config: dict[str, Any]):
        self._config = config
        
    @classmethod
    def from_file(cls, path: Path) -> "ConfigHelper":
        if not path.exists():
            LOGGER.error(f"Config file {path} not found")
            raise FileNotFoundError(f"Config file not found {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data)

    def get(self, path: str) -> Any:
        current = self._config
        for part in path.split('.'):
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"Missing config key: {path}")
            current = current[part]
        return current


class Tokenizer:
    def __init__(self, model_dir: Path):
        tokenizer_dir = model_dir / "tokenizer"
        vocab_path = tokenizer_dir / "vocab.json"
        merges_path = tokenizer_dir / "merges.txt"
        
        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(f"Tokenizer files not found: {vocab_path} / {merges_path}")
            
        self._tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
        self._special_ids = special_tokens
        
    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids
    
    def build_custom_voice_prompt_ids(
        self, 
        text: str, 
        instruct_prompt: str | None = None
    ) -> np.ndarray:
        indexes = []

        if instruct_prompt:
            user_ids = self.encode("user")
            instruct_ids = self.encode(instruct_prompt)
            indexes.extend([self._special_ids.IM_START_ID])
            indexes.extend(user_ids)
            indexes.append(self._special_ids.NEWLINE_TOKEN_ID)
            indexes.extend(instruct_ids)
            indexes.extend([self._special_ids.IM_END_ID, self._special_ids.NEWLINE_TOKEN_ID])

        text_ids = self.encode(text)
        
        indexes.extend([
            self._special_ids.IM_START_ID, 
            self._special_ids.ASSISTANT_TOKEN_ID, 
            self._special_ids.NEWLINE_TOKEN_ID
        ])
        
        indexes.extend(text_ids)
        
        indexes.extend([
            self._special_ids.IM_END_ID, 
            self._special_ids.NEWLINE_TOKEN_ID, 
            self._special_ids.IM_START_ID, 
            self._special_ids.ASSISTANT_TOKEN_ID, 
            self._special_ids.NEWLINE_TOKEN_ID
        ])

        voice_prompt = np.array(indexes, dtype=np.int64)
        LOGGER.debug("Prompt tokenized: shape=%s, head=%s", voice_prompt.shape, voice_prompt[:32].tolist())
        return voice_prompt


class AssetManager:
    def __init__(self, embedding_dir: Path):
        self._dir = embedding_dir
        self._data = {}
        self._load_all()

    def _load_all(self):
        config_path = self._dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        self._data["config"] = ConfigHelper.from_file(config_path)
        cp_tables = []
        
        for i in range(NUM_EMBEDDING_FILES):
            path = self._dir / f"cp_codec_embedding_{i}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing CP embedding table: {path}")
            cp_tables.append(np.load(path))
            
        self._data["cp_codec_embeddings"] = cp_tables

        file_map = {
            "text_embedding": "text_embedding.npy",
            "text_projection_fc1_weight": "text_projection_fc1_weight.npy",
            "text_projection_fc1_bias": "text_projection_fc1_bias.npy",
            "text_projection_fc2_weight": "text_projection_fc2_weight.npy",
            "text_projection_fc2_bias": "text_projection_fc2_bias.npy",
            "talker_codec_embedding": "talker_codec_embedding.npy",
        }

        for key, filename in file_map.items():
            self._data[key] = np.load(self._dir / filename)

    def get(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(f"Asset key '{key}' not found")
        return self._data[key]

    @property
    def config(self) -> ConfigHelper:
        return self._data["config"]

    def get_total_size_mb(self) -> float:
        total = 0.0
        for item in self._data.values():
            if isinstance(item, np.ndarray):
                total += item.nbytes
            elif isinstance(item, list):
                total += sum(arr.nbytes for arr in item if isinstance(arr, np.ndarray))
        return total / (1024 ** 2)


class ProjectionUtils:
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    @staticmethod
    def project_text(x_2048: np.ndarray, assets: dict[str, np.ndarray]) -> np.ndarray:
        fc1_w = assets["text_projection_fc1_weight"].astype(np.float64, copy=False)
        fc1_b = assets["text_projection_fc1_bias"].astype(np.float64, copy=False)
        fc2_w = assets["text_projection_fc2_weight"].astype(np.float64, copy=False)
        fc2_b = assets["text_projection_fc2_bias"].astype(np.float64, copy=False)
        
        with np.errstate(all='ignore'):
            x = x_2048.astype(np.float64, copy=False)
            x = x @ fc1_w.T + fc1_b
            x = ProjectionUtils.gelu(x)
            x = x @ fc2_w.T + fc2_b

        if not np.isfinite(x).all():
            raise RuntimeError("Non-finite values detected in text_projection output")
        return x.astype(np.float32, copy=False)


class PrefillEmbeddingBuilder:
    def __init__(self, assets: AssetManager):
        self.assets = assets
        self.config = assets.config

    def _get_token_embedding(self, token_id: int) -> np.ndarray:
        emb = self.assets.get("text_embedding")[token_id]
        return emb.astype(np.float32, copy=False)

    def _get_talker_embedding(self, token_id: int) -> np.ndarray:
        emb = self.assets.get("talker_codec_embedding")[token_id]
        return emb.astype(np.float32, copy=False)

    def _get_cp_embedding(self, group_idx: int, token_id: int) -> np.ndarray:
        table = self.assets.get("cp_codec_embeddings")[group_idx]
        return table[token_id].astype(np.float32, copy=False)

    def build(
        self,
        token_ids: np.ndarray,
        language: str = "auto",
        speaker_id: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        hidden_size = int(self.config.get("talker.hidden_size"))
        token_ids_list = token_ids.tolist()
        
        if len(token_ids_list) < 9:
            raise ValueError("Prompt token sequence is too short for prefill builder.")

        role_embeddings = []
        for i in range(3):
            raw_emb = self._get_token_embedding(token_ids_list[i])
            proj_emb = ProjectionUtils.project_text(raw_emb, self.assets._data)
            role_embeddings.append(proj_emb)

        language_ids = self.config.get("language_ids")
        cfg_talker = "talker"
        
        codec_ids_map = {
            "think": int(self.config.get(f"{cfg_talker}.codec_think_id")),
            "nothink": int(self.config.get(f"{cfg_talker}.codec_nothink_id")),
            "think_bos": int(self.config.get(f"{cfg_talker}.codec_think_bos_id")),
            "think_eos": int(self.config.get(f"{cfg_talker}.codec_think_eos_id")),
            "pad": int(self.config.get(f"{cfg_talker}.codec_pad_id")),
            "bos": int(self.config.get(f"{cfg_talker}.codec_bos_id")),
        }

        codec_prefix = []
        if language != "auto":
            if language.lower() not in language_ids:
                raise KeyError(f"Language '{language}' not found in config.language_ids")
            
            lang_id = int(language_ids[language.lower()])
            codec_prefix.extend([
                codec_ids_map["think"],
                codec_ids_map["think_bos"],
                lang_id,
                codec_ids_map["think_eos"],
            ])
        else:
            codec_prefix.extend([
                codec_ids_map["nothink"],
                codec_ids_map["think_bos"],
                codec_ids_map["think_eos"],
            ])

        if speaker_id >= 0:
            codec_prefix.append(int(speaker_id))

        codec_prefix.extend([codec_ids_map["pad"], codec_ids_map["bos"]])
        tts_cfg = "tts"
        
        tts_token_ids = {
            "pad": int(self.config.get(f"{tts_cfg}.tts_pad_token_id")),
            "bos": int(self.config.get(f"{tts_cfg}.tts_bos_token_id")),
            "eos": int(self.config.get(f"{tts_cfg}.tts_eos_token_id")),
        }
        
        tts_projections = {}
        for name, tid in tts_token_ids.items():
            raw = self._get_token_embedding(tid)
            tts_projections[name] = ProjectionUtils.project_text(raw, self.assets._data)

        talker_input_embeds = []
        codec_prefix_len = len(codec_prefix)

        for i in range(max(0, codec_prefix_len - 2)):
            codec_emb = self._get_talker_embedding(codec_prefix[i])
            combined = tts_projections["pad"] + codec_emb
            talker_input_embeds.append(combined.astype(np.float32, copy=False))
            
        if codec_prefix_len >= 2:
            codec_emb = self._get_talker_embedding(codec_prefix[-2])
            combined = tts_projections["bos"] + codec_emb
            talker_input_embeds.append(combined.astype(np.float32, copy=False))

        all_embeddings = role_embeddings + talker_input_embeds
        token3_raw = self._get_token_embedding(token_ids_list[3])
        token3_proj = ProjectionUtils.project_text(token3_raw, self.assets._data)
        codec_bos_emb = self._get_talker_embedding(codec_ids_map["bos"])
        all_embeddings.append((token3_proj + codec_bos_emb).astype(np.float32, copy=False))

        trailing_list = []
        trailing_tokens = token_ids_list[4:-5]
        
        for trailing_token in trailing_tokens:
            raw = self._get_token_embedding(trailing_token)
            trailing_list.append(ProjectionUtils.project_text(raw, self.assets._data))
        
        trailing_list.append(tts_projections["eos"].astype(np.float32, copy=False))
        inputs_embeds = np.stack(all_embeddings, axis=0)[None, :, :]
        trailing_text_hidden = np.stack(trailing_list, axis=0)

        LOGGER.info(
            "Prefill embedding built: inputs_embeds=%s trailing_text_hidden=%s",
            inputs_embeds.shape,
            trailing_text_hidden.shape,
        )

        assert inputs_embeds.shape[-1] == hidden_size
        assert trailing_text_hidden.shape[-1] == hidden_size
        return inputs_embeds.astype(np.float32, copy=False), trailing_text_hidden.astype(np.float32, copy=False)
