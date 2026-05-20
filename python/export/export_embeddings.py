import argparse
import json
import sys
from pathlib import Path

from export_utils import configure_output_encoding


try:
    import compat_patches
except ImportError:
    print("WARNING: compat_patches.py not found. Model loading may fail with newer transformers.")

import numpy as np
import torch

try:
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
except ImportError:
    print("ERROR: qwen-tts package not found.")
    print("       Install with: pip install qwen-tts")
    sys.exit(1)


def save_tensor(tensor, path):
    """Save a PyTorch tensor as a NumPy .npy file."""
    np.save(str(path), tensor.detach().cpu().float().numpy())
    print(f"  ✓ {path.name}  {tuple(tensor.shape)}")


def main():
    configure_output_encoding()
    parser = argparse.ArgumentParser(
        description="Extract embedding weights as NumPy arrays for Python3 inference"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/Qwen3-TTS-0.6B-CustomVoice",
        help="Path to the local Qwen3-TTS model directory (download first with download_models.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx/embeddings",
        help="Directory to save .npy embedding files",
    )
    args = parser.parse_args()

    import os
    if not os.path.isdir(args.model_dir):
        print(f"ERROR: Model directory not found: {args.model_dir}")
        if "/" in args.model_dir and not os.path.exists(args.model_dir):
            print(f"\n  This script requires a LOCAL directory with downloaded model weights.")
            print("  To download models first, run:  python download_models.py")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_dir} ...")
    config = Qwen3TTSConfig.from_pretrained(args.model_dir)
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_dir,
        config=config,
        dtype=torch.float32,
    )
    model.eval()

    talker = model.talker
    config = model.config
    talker_config = config.talker_config
    print("\nText embedding:")
    text_emb = talker.model.text_embedding.weight
    save_tensor(text_emb, output_dir / "text_embedding.npy")

    print("\nText projection (ResizeMLP):")
    proj = talker.text_projection
    save_tensor(proj.linear_fc1.weight, output_dir / "text_projection_fc1_weight.npy")
    save_tensor(proj.linear_fc1.bias, output_dir / "text_projection_fc1_bias.npy")
    save_tensor(proj.linear_fc2.weight, output_dir / "text_projection_fc2_weight.npy")
    save_tensor(proj.linear_fc2.bias, output_dir / "text_projection_fc2_bias.npy")

    print("\nTalker codec embedding (group 0):")
    codec_emb = talker.model.codec_embedding.weight
    save_tensor(codec_emb, output_dir / "talker_codec_embedding.npy")

    print("\nCode Predictor codec embeddings (31 tables):")
    cp_embeddings = talker.code_predictor.model.codec_embedding
    for i, emb in enumerate(cp_embeddings):
        save_tensor(emb.weight, output_dir / f"cp_codec_embedding_{i}.npy")

    cp = talker.code_predictor
    if hasattr(cp, "small_to_mtp_projection") and cp.small_to_mtp_projection is not None:
        proj_layer = cp.small_to_mtp_projection
        if hasattr(proj_layer, "weight"):
            print("\nCode Predictor projection (small_to_mtp):")
            save_tensor(proj_layer.weight, output_dir / "cp_projection_weight.npy")
            if hasattr(proj_layer, "bias") and proj_layer.bias is not None:
                save_tensor(proj_layer.bias, output_dir / "cp_projection_bias.npy")
            else:
                print("  (no bias)")
    else:
        print("\nCode Predictor projection: not present (0.6B model, skip)")

    print("\nCodec head:")
    codec_head_w = talker.codec_head.weight
    save_tensor(codec_head_w, output_dir / "codec_head_weight.npy")

    print("\nSpeaker IDs:")
    spk_ids = talker_config.spk_id or {}
    with open(output_dir / "speaker_ids.json", "w") as f:
        json.dump(spk_ids, f, indent=2)
    print(f"  ✓ speaker_ids.json  ({len(spk_ids)} speakers)")

    print("\nConfig:")
    meta = {
        "talker": {
            "hidden_size": talker_config.hidden_size,
            "text_hidden_size": talker_config.text_hidden_size,
            "vocab_size": talker_config.vocab_size,
            "num_hidden_layers": talker_config.num_hidden_layers,
            "num_attention_heads": talker_config.num_attention_heads,
            "num_key_value_heads": talker_config.num_key_value_heads,
            "head_dim": getattr(talker_config, "head_dim", 128),
            "num_code_groups": talker_config.num_code_groups,
            "codec_eos_token_id": talker_config.codec_eos_token_id,
            "codec_think_id": talker_config.codec_think_id,
            "codec_nothink_id": talker_config.codec_nothink_id,
            "codec_think_bos_id": talker_config.codec_think_bos_id,
            "codec_think_eos_id": talker_config.codec_think_eos_id,
            "codec_pad_id": talker_config.codec_pad_id,
            "codec_bos_id": talker_config.codec_bos_id,
            "rope_theta": talker_config.rope_theta,
        },
        "code_predictor": {
            "hidden_size": talker_config.code_predictor_config.hidden_size,
            "vocab_size": talker_config.code_predictor_config.vocab_size,
            "num_hidden_layers": talker_config.code_predictor_config.num_hidden_layers,
            "num_attention_heads": talker_config.code_predictor_config.num_attention_heads,
            "num_key_value_heads": talker_config.code_predictor_config.num_key_value_heads,
            "head_dim": getattr(talker_config.code_predictor_config, "head_dim", 128),
            "rope_theta": talker_config.code_predictor_config.rope_theta,
        },
        "tts": {
            "tts_bos_token_id": config.tts_bos_token_id,
            "tts_eos_token_id": config.tts_eos_token_id,
            "tts_pad_token_id": config.tts_pad_token_id,
            "im_start_token_id": config.im_start_token_id,
            "im_end_token_id": config.im_end_token_id,
        },
        "language_ids": talker_config.codec_language_id or {},
        "speaker_dialect": talker_config.spk_is_dialect or {},
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ config.json")

    print(f"\nAll embeddings saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
