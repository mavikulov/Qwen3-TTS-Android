from __future__ import annotations

import time
import logging
import argparse
from typing import Any
from pathlib import Path

from package.model_assets import AssetManager, Tokenizer
from package.inference_engine import TalkerGenerator, Vocoder
from package.utils import ModelSessions

from package.utils import set_seed, setup_logging, read_config, resolve_path, get_model_size_mb, save_wav


LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Inference Pipeline")
    parser.add_argument("--config_file", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--save-dir", default=None, help="Output/debug directory")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    setup_logging(args.log_level)
    
    LOGGER.info(f"Loading config from {args.config}")
    config = read_config(args.config_file)
    paths = config['paths']
    parameters = config['model_parameters']
    
    model_dir = resolve_path(paths['model_dir'])
    embeddings_dir = resolve_path(paths['embeddings_dir'])
    save_dir = resolve_path(args.save_dir) if args.save_dir else model_dir / "_debug_outputs"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    def get_param(key: str, cli_val: Any, cfg_key: str) -> Any:
        if cli_val is not None:
            return cli_val
        return parameters.get(cfg_key, cli_val)

    language = get_param("language", args.language, "language")
    speaker_id = get_param("speaker_id", args.speaker_id, "speaker_id")
    max_tokens = get_param("max_tokens", args.max_tokens, "max_new_tokens")
    temperature = get_param("temperature", args.temperature, "temperature")
    top_k = get_param("top_k", args.top_k, "top_k")
    rep_penalty = get_param("rep_penalty", args.repetition_penalty, "repetition_penalty")
    min_new_tokens = parameters.get("min_new_tokens", 0)

    model_files = {
        "prefill": model_dir / "talker_prefill.onnx",
        "decode": model_dir / "talker_decode.onnx",
        "cp": model_dir / "code_predictor.onnx",
        "vocoder": model_dir / "vocoder.onnx",
    }

    LOGGER.info("Checking model files...")
    for name, path in model_files.items():
        if not path.exists():
            LOGGER.error("Missing model file: %s", path)
            raise FileNotFoundError(f"Missing model: {path}")
        size_mb = get_model_size_mb(path)
        LOGGER.info("Loaded %s (%.2f MB)", name, size_mb)

    t0 = time.perf_counter()
    sessions = ModelSessions.load(model_files, provider="CPUExecutionProvider")
    LOGGER.info("Sessions loaded in %.2f seconds", time.perf_counter() - t0)

    LOGGER.info("Loading assets (embeddings, config)...")
    t0 = time.perf_counter()
    assets = AssetManager(embeddings_dir)
    LOGGER.info("Assets loaded in %.2f seconds (Size: %.2f MB)", time.perf_counter() - t0, assets.get_total_size_mb())

    LOGGER.info("Loading tokenizer...")
    tokenizer = Tokenizer(model_dir)

    LOGGER.info("Tokenizing input text...")
    LOGGER.debug("Text: '%s'", args.text)
    if args.instruct:
        LOGGER.debug("Instruction: '%s'", args.instruct)
        
    token_ids = tokenizer.build_custom_voice_prompt_ids(
        text=args.text,
        instruct=args.instruct
    )
    
    LOGGER.info("Token sequence length: %d", len(token_ids))
    LOGGER.info("Starting generation...")
    
    generator = TalkerGenerator(
        prefill_session=sessions.prefill,
        decode_session=sessions.decode,
        cp_session=sessions.cp,
        assets=assets
    )

    t0 = time.perf_counter()
    codes = generator.generate(
        token_ids=token_ids,
        language=language,
        speaker_id=speaker_id,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=rep_penalty,
        min_new_tokens=min_new_tokens,
        save_dir=None
    )
    
    gen_time = time.perf_counter() - t0
    LOGGER.info("Generation completed in %.2f seconds", gen_time)
    LOGGER.info("Generated codes shape: %s", codes.shape)

    LOGGER.info("Running vocoder...")
    t0 = time.perf_counter()
    vocoder = Vocoder(sessions.vocoder)
    waveform = vocoder(codes)
    voc_time = time.perf_counter() - t0
    
    LOGGER.info("Vocoder completed in %.2f seconds", voc_time)
    LOGGER.info("Waveform shape: %s, dtype: %s", waveform.shape, waveform.dtype)

    wav_path = save_dir / "output.wav"
    save_wav(wav_path, waveform, sample_rate=24000)
    LOGGER.info("Audio saved to: %s", wav_path)
    LOGGER.info("Pipeline finished successfully! Total time: %.2f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
    