import json
import logging
from typing import Any, Optional
from pathlib import Path

import numpy as np
from tokenizers import ByteLevelBPETokenizer


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

LOGGER = logging.getLogger(__name__)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_config(config: dict[str, Any], path: str) -> Any:
    cur = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing config key: {path}")
        cur = cur[part]
    return cur


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exponents = np.exp(x)
    sum_of_exponents = np.sum(exponents, axis=axis, keepdims=True)
    sum_of_exponents = np.where(sum_of_exponents == 0, 1.0, sum_of_exponents)
    return exponents / sum_of_exponents


def load_tokenizer(model_dir: Path) -> ByteLevelBPETokenizer:
    tokenizer_dir = model_dir / "tokenizer"
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer files not found: {vocab_path} / {merges_path}")
    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


def encode_plain_text(tokenizer: ByteLevelBPETokenizer, text: str) -> list[int]:
    return tokenizer.encode(text).ids


def build_custom_voice_prompt_ids(
    tokenizer: ByteLevelBPETokenizer,
    text: str,
    instruct: Optional[str] = None,
) -> np.ndarray:
    ids = []

    if instruct:
        user_ids = encode_plain_text(tokenizer, "user")
        instruct_ids = encode_plain_text(tokenizer, instruct)
        ids.extend([IM_START_ID])
        ids.extend(user_ids)
        ids.append(NEWLINE_TOKEN_ID)
        ids.extend(instruct_ids)
        ids.extend([IM_END_ID, NEWLINE_TOKEN_ID])

    text_ids = encode_plain_text(tokenizer, text)
    ids.extend([IM_START_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID])
    ids.extend(text_ids)
    ids.extend([IM_END_ID, NEWLINE_TOKEN_ID, IM_START_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID])

    result = np.array(ids, dtype=np.int64)
    LOGGER.info("Prompt tokenized: shape=%s", result.shape)
    LOGGER.debug("Prompt token head: %s", result[:32].tolist())
    return result


# def load_assets(model_dir: Path) -> dict[str, Any]:
#     emb_dir = model_dir / "embeddings"
#     config = load_json(emb_dir / "config.json")
#     cp_tables = []
#     for i in range(NUM_EMBEDDING_FILES):
#         path_to_emb = emb_dir / f"cp_codec_embedding_{i}.npy"
#         if not path_to_emb.exists():
#             raise FileNotFoundError(f"Missing CP embedding table: {path_to_emb}")
#         cp_tables.append(np.load(path_to_emb))

#     assets = {
#         "config": config,
#         "text_embedding": np.load(emb_dir / "text_embedding.npy"),
#         "text_projection_fc1_weight": np.load(emb_dir / "text_projection_fc1_weight.npy"),
#         "text_projection_fc1_bias": np.load(emb_dir / "text_projection_fc1_bias.npy"),
#         "text_projection_fc2_weight": np.load(emb_dir / "text_projection_fc2_weight.npy"),
#         "text_projection_fc2_bias": np.load(emb_dir / "text_projection_fc2_bias.npy"),
#         "talker_codec_embedding": np.load(emb_dir / "talker_codec_embedding.npy"),
#         "cp_codec_embeddings": cp_tables,
#     }
    
#     return assets


def load_assets(emb_dir: Path) -> dict[str, Any]:
    config = load_json(emb_dir / "config.json")
    cp_tables = []
    for i in range(NUM_EMBEDDING_FILES):
        path_to_emb = emb_dir / f"cp_codec_embedding_{i}.npy"
        if not path_to_emb.exists():
            raise FileNotFoundError(f"Missing CP embedding table: {path_to_emb}")
        cp_tables.append(np.load(path_to_emb))

    assets = {
        "config": config,
        "text_embedding": np.load(emb_dir / "text_embedding.npy"),
        "text_projection_fc1_weight": np.load(emb_dir / "text_projection_fc1_weight.npy"),
        "text_projection_fc1_bias": np.load(emb_dir / "text_projection_fc1_bias.npy"),
        "text_projection_fc2_weight": np.load(emb_dir / "text_projection_fc2_weight.npy"),
        "text_projection_fc2_bias": np.load(emb_dir / "text_projection_fc2_bias.npy"),
        "talker_codec_embedding": np.load(emb_dir / "talker_codec_embedding.npy"),
        "cp_codec_embeddings": cp_tables,
    }
    return assets


def get_embeddings_size(assets: dict[str, Any]) -> float:
    total_size = 0.0
    for item in assets.values():
        if isinstance(item, np.ndarray):
            total_size += item.nbytes
    return total_size / (1024 ** 2)


def log_assets(assets: dict[str, Any]) -> None:
    pass
    # LOGGER.info("Loaded embedding assets")
    # for key, value in assets.items():
    #     if key == "config":
    #         LOGGER.info("  config: loaded")
    #     elif isinstance(value, np.ndarray):
    #         LOGGER.info("  %s: shape=%s dtype=%s", key, value.shape, value.dtype)
    #     elif isinstance(value, list):
    #         LOGGER.info("  %s: list[%d]", key, len(value))


def GELU(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def text_embedding_lookup(token_id: int, assets: dict[str, Any]) -> np.ndarray:
    return assets["text_embedding"][token_id].astype(np.float32, copy=False)


def text_projection(x_2048: np.ndarray, assets: dict[str, Any]) -> np.ndarray:
    fc1_w = assets["text_projection_fc1_weight"].astype(np.float64, copy=False)
    fc1_b = assets["text_projection_fc1_bias"].astype(np.float64, copy=False)
    fc2_w = assets["text_projection_fc2_weight"].astype(np.float64, copy=False)
    fc2_b = assets["text_projection_fc2_bias"].astype(np.float64, copy=False)
    
    with np.errstate(all='ignore'):
        x = x_2048.astype(np.float64, copy=False)
        x = x @ fc1_w.T + fc1_b
        x = GELU(x)
        x = x @ fc2_w.T + fc2_b

    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite values detected in text_projection output")
    return x.astype(np.float32, copy=False)


def talker_codec_embedding_lookup(token_id: int, assets: dict[str, Any]) -> np.ndarray:
    return assets["talker_codec_embedding"][token_id].astype(np.float32, copy=False)


def cp_codec_embedding_lookup(group_idx_zero_based: int, token_id: int, assets: dict[str, Any]) -> np.ndarray:
    return assets["cp_codec_embeddings"][group_idx_zero_based][token_id].astype(np.float32, copy=False)


def build_prefill_embedding(
    token_ids: np.ndarray,
    assets: dict[str, Any],
    language: str = "auto",
    speaker_id: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    config = assets["config"]
    hidden_size = int(get_config(config, "talker.hidden_size"))

    token_ids_list = token_ids.tolist()
    if len(token_ids_list) < 9:
        raise ValueError("Prompt token sequence is too short for prefill builder.")

    role_embeds: list[np.ndarray] = []
    for i in range(3):
        role_embeds.append(text_projection(text_embedding_lookup(token_ids_list[i], assets), assets))

    codec_prefix: list[int] = []
    language_ids = get_config(config, "language_ids")

    codec_think_id = int(get_config(config, "talker.codec_think_id"))
    codec_nothink_id = int(get_config(config, "talker.codec_nothink_id"))
    codec_think_bos_id = int(get_config(config, "talker.codec_think_bos_id"))
    codec_think_eos_id = int(get_config(config, "talker.codec_think_eos_id"))
    codec_pad_id = int(get_config(config, "talker.codec_pad_id"))
    codec_bos_id = int(get_config(config, "talker.codec_bos_id"))

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

    tts_pad_token_id = int(get_config(config, "tts.tts_pad_token_id"))
    tts_bos_token_id = int(get_config(config, "tts.tts_bos_token_id"))
    tts_eos_token_id = int(get_config(config, "tts.tts_eos_token_id"))

    tts_pad_proj = text_projection(text_embedding_lookup(tts_pad_token_id, assets), assets)
    tts_bos_proj = text_projection(text_embedding_lookup(tts_bos_token_id, assets), assets)
    tts_eos_proj = text_projection(text_embedding_lookup(tts_eos_token_id, assets), assets)

    talker_input_embeds: list[np.ndarray] = []
    codec_prefix_len = len(codec_prefix)

    for i in range(max(0, codec_prefix_len - 2)):
        codec_emb = talker_codec_embedding_lookup(codec_prefix[i], assets)
        talker_input_embeds.append((tts_pad_proj + codec_emb).astype(np.float32, copy=False))

    if codec_prefix_len >= 2:
        codec_emb = talker_codec_embedding_lookup(codec_prefix[-2], assets)
        talker_input_embeds.append((tts_bos_proj + codec_emb).astype(np.float32, copy=False))

    all_embeds: list[np.ndarray] = []
    all_embeds.extend(role_embeds)
    all_embeds.extend(talker_input_embeds)

    token3_proj = text_projection(text_embedding_lookup(token_ids_list[3], assets), assets)
    codec_bos_emb = talker_codec_embedding_lookup(codec_bos_id, assets)
    all_embeds.append((token3_proj + codec_bos_emb).astype(np.float32, copy=False))

    trailing_list: list[np.ndarray] = []
    trailing_tokens = token_ids_list[4:-5]
    for tok in trailing_tokens:
        trailing_list.append(text_projection(text_embedding_lookup(tok, assets), assets))
    
    trailing_list.append(tts_eos_proj.astype(np.float32, copy=False))

    inputs_embeds = np.stack(all_embeds, axis=0)[None, :, :]
    trailing_text_hidden = np.stack(trailing_list, axis=0)

    LOGGER.info(
        "Prefill embedding built: inputs_embeds=%s trailing_text_hidden=%s",
        inputs_embeds.shape,
        trailing_text_hidden.shape,
    )

    assert inputs_embeds.shape[-1] == hidden_size
    assert trailing_text_hidden.shape[-1] == hidden_size

    return inputs_embeds.astype(np.float32, copy=False), trailing_text_hidden.astype(np.float32, copy=False)
