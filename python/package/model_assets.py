from pathlib import Path
from typing import Any, Optional
import logging

import numpy as np
from tokenizers import ByteLevelBPETokenizer

from package.utils import load_config
from special_tokens import special_tokens


NUM_EMBEDDING_FILES = 15
LOGGER = logging.getLogger(__name__)


class AssetManager:
    def __init__(self, embedding_dir: Path):
        self.embedding_dir: Path = embedding_dir
        self.data: dict[str, Any] = {}
        self.config_path: Path = self.embedding_dir / "config.json" 
        self.load_all_files()
        
    def load_all_files(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        cp_tables = []
        for i in range(NUM_EMBEDDING_FILES):
            path = self.embedding_dir / f"cp_codec_embedding_{i}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing CP embedding table: {path}")
            cp_tables.append(np.load(path))
        
        self.data["cp_codec_embeddings"] = cp_tables
        file_map = {
            "text_embedding": "text_embedding.npy",
            "text_projection_fc1_weight": "text_projection_fc1_weight.npy",
            "text_projection_fc1_bias": "text_projection_fc1_bias.npy",
            "text_projection_fc2_weight": "text_projection_fc2_weight.npy",
            "text_projection_fc2_bias": "text_projection_fc2_bias.npy",
            "talker_codec_embedding": "talker_codec_embedding.npy",
        }
        
        for key, filename in file_map.items():
            self.data[key] = np.load(self.embedding_dir / filename)

    def get(self, key: str) -> Any:
        return self.data.get(key, None)
    

class Tokenizer:
    def __init__(self, tokenizer_dir: Path):
        vocab_path = tokenizer_dir / "vocab.json"
        merges_path = tokenizer_dir / "merges.txt"
        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(f"Tokenizer files not found: {vocab_path} or {merges_path}")
        
        self.tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
        self.special_ids = special_tokens
        
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids
    
    def build_custom_voice_prompt_ids(
        self, 
        text: str, 
        instruct_prompt: Optional[str]
    ) -> np.ndarray:
        indexes = []
        
        if instruct_prompt:
            user_ids = self.encode("user")
            instruct_ids = self.encode(instruct_prompt)
            indexes.extend([self.special_ids.IM_START_ID])
            indexes.extend(user_ids)
            indexes.append(self.special_ids.NEWLINE_TOKEN_ID)
            indexes.extend(instruct_ids)
            indexes.extend([self.special_ids.IM_END_ID, self.special_ids.NEWLINE_TOKEN_ID])
            
        text_ids = self.encode(text)
        
        indexes.extend([
            self.special_ids.IM_START_ID,
            self.special_ids.ASSISTANT_TOKEN_ID,
            self.special_ids.NEWLINE_TOKEN_ID
        ])
        
        indexes.extend(text_ids)
        
        indexes.extend([
            self.special_ids.IM_END_ID, 
            self.special_ids.NEWLINE_TOKEN_ID, 
            self.special_ids.IM_START_ID, 
            self.special_ids.ASSISTANT_TOKEN_ID, 
            self.special_ids.NEWLINE_TOKEN_ID
        ])
        
        voice_prompt = np.array(indexes, dtype=np.int64)
        return voice_prompt


class ProjectionUtils:
    @staticmethod
    def GeLU(x: np.ndarray) -> np.ndarray:
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
            x = ProjectionUtils.GeLU(x)
            x = x @ fc2_w.T + fc2_b

        if not np.isfinite(x).all():
            raise RuntimeError("Non-finite values detected in text_projection output")
        
        return x.astype(np.float32, copy=False)


class PrefillEmbeddingBuilder:
    def __init__(self, assets: AssetManager):
        self.assets: AssetManager = assets
    
    def _get_token_embedding(self, token_id: int) -> np.ndarray:
        embedding = self.assets.get("text_embedding")[token_id]
        return embedding.astype(np.float32, copy=False)
    
    def _get_talker_embedding(self, token_id: int) -> np.ndarray:
        embedding = self.assets.get("talker_codec_embedding")[token_id]
        return embedding.astype(np.float32, copy=False)

    def _get_cp_embedding(self, group_idx: int, token_id: int) -> np.ndarray:
        table = self.assets.get("cp_codec_embeddings")[group_idx]
        return table[token_id].astype(np.float32, copy=False)
    
    def build_codec_prefixes(self, ) -> list[np.ndarray]:
        pass
    
    def build(
        self,
        token_ids: np.ndarray,
        language: str = "auto",
        speaker_id: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        config = load_config(self.assets.config_path)
        token_ids_list = token_ids.tolist()
        
        if len(token_ids_list) < 9:
            raise ValueError("Prompt token sequence is too short for prefill builder.")
        
        role_embeddings = []
        for i in range(3):
            raw_embedding = self._get_token_embedding(token_ids_list[i])
            projection_embedding = ProjectionUtils.project_text(raw_embedding, self.assets.data)
            role_embeddings.append(projection_embedding)
            
        language_ids = config["language_ids"]
        cfg_talker = "talker"
        codec_ids_map = {
            "think": int(config[f"{cfg_talker}"]["codec_think_id"]),
            "nothink": int(config[f"{cfg_talker}"]["codec_nothink_id"]),
            "think_bos": int(config[f"{cfg_talker}"]["codec_think_bos_id"]),
            "think_eos": int(config[f"{cfg_talker}"]["codec_think_eos_id"]),
            "pad": int(config[f"{cfg_talker}"]["codec_pad_id"]),
            "bos": int(config[f"{cfg_talker}"]["codec_bos_id"])
        }
        
        codec_prefix = []

        if language != "auto":
            if language.lower() not in language_ids.keys():
                raise KeyError(f"Language {language} not found in config")
            
            language_id = int(language_ids[language.lower()])
            codec_prefix.extend([
                codec_ids_map["think"],
                codec_ids_map["think_bos"],
                language_id,
                codec_ids_map["think_eos"]
            ])
        else:
            codec_prefix.extend([
                codec_ids_map["think"],
                codec_ids_map["think_bos"],
                codec_ids_map["think_eos"]
            ])
        
        if speaker_id >= 0:
            codec_prefix.extend([codec_ids_map["pad"], codec_ids_map["bos"]])
        
        tts_cfg = "tts"
        tts_token_ids = {
            "pad": int(config[f"{tts_cfg}"]["tts_pad_token_id"]),
            "bos": int(config[f"{tts_cfg}"]["tts_bos_token_id"]),
            "eos": int(config[f"{tts_cfg}"]["tts_eos_token_id"])
        }
        
        tts_projections = {}
        for name, tid in tts_token_ids.items():
            raw = self._get_token_embedding(tid)
            tts_projections[name] = ProjectionUtils.project_text(raw, self.assets.data)
        
        talker_input_embeddings = []
        codec_prefix_len = len(codec_prefix)
        
        for i in range(max(0, codec_prefix_len - 2)):
            codec_embedding = self._get_talker_embedding(codec_prefix[i])
            combined = tts_projections["pad"] + codec_embedding
            talker_input_embeddings.append(combined.astype(np.float32, copy=False))
            
        if codec_prefix_len >= 2:
            codec_embedding = self._get_talker_embedding(codec_prefix[-2])
            combined = tts_projections["bos"] + codec_embedding
            talker_input_embeddings.append(combined.astype(np.float32, copy=False))
            
        all_embeddings = role_embeddings + talker_input_embeddings
        token_raw = self._get_token_embedding(token_ids_list[3])
        token_proj = ProjectionUtils.project_text(token_raw, self.assets.data)
        codec_bos_embedding = self._get_talker_embedding(codec_ids_map["bos"])
        all_embeddings.append((token_proj + codec_bos_embedding).astype(np.float32, copy=False))
        trailing_list = []
        trailing_tokens = token_ids_list[4:-5]
        
        for trailing_token in trailing_tokens:
            raw_embedding = self._get_token_embedding(trailing_token)
            trailing_list.append(ProjectionUtils.project_text(raw_embedding, self.assets.data))
            
        trailing_list.append(tts_projections["eos"].astype(np.float64, copy=False))
        input_embeddings = np.stack(all_embeddings, axis=0)[None, :, :]
        trailing_text_hidden = np.stack(trailing_list, axis=0)
        
        LOGGER.info(
            "Prefill embedding built: inputs_embeds=%s trailing_text_hidden=%s",
            input_embeddings.shape,
            trailing_text_hidden.shape,
        )
        
        return input_embeddings.astype(np.float32, copy=False), trailing_text_hidden.astype(np.float32, copy=False)
