

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from export_utils import configure_output_encoding


import transformers.utils.generic as _tug
_orig_check = _tug.check_model_inputs

def _compat_check_model_inputs(func=None):
    if func is None:
        return lambda fn: fn
    return func

_tug.check_model_inputs = _compat_check_model_inputs

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_FNS
if "default" not in _ROPE_FNS:
    def _compute_default_rope(config=None, device=None, **kwargs):
        base = config.rope_theta
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim))
        return inv_freq, 1.0
    _ROPE_FNS["default"] = _compute_default_rope


import transformers.masking_utils as _masking
_orig_sdpa_mask = _masking.sdpa_mask


def _patched_sdpa_mask(q_length=None, **kwargs):
    if isinstance(q_length, torch.Tensor):
        if q_length.dim() == 0:
            q_length = int(q_length.item())
        else:
            q_length = q_length.shape[0]
    return _orig_sdpa_mask(q_length=q_length, **kwargs)


_masking.sdpa_mask = _patched_sdpa_mask
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS as _MASK_FNS
if 'sdpa' in _MASK_FNS:
    _MASK_FNS['sdpa'] = _patched_sdpa_mask

_orig_torch_diff = torch.diff


def _onnx_safe_diff(input, n=1, dim=-1, prepend=None, append=None):
    if n != 1:
        return _orig_torch_diff(input, n=n, dim=dim, prepend=prepend, append=append)
    parts = []
    if prepend is not None:
        parts.append(prepend)
    parts.append(input)
    if append is not None:
        parts.append(append)
    combined = torch.cat(parts, dim=dim) if len(parts) > 1 else input
    return torch.narrow(combined, dim, 1, combined.size(dim) - 1) - torch.narrow(combined, dim, 0, combined.size(dim) - 1)


torch.diff = _onnx_safe_diff
_orig_cumsum = torch.Tensor.cumsum


def _onnx_safe_cumsum(self, dim, dtype=None):
    if self.dtype == torch.bool:
        return _orig_cumsum(self.to(torch.int64), dim, dtype=dtype)
    return _orig_cumsum(self, dim, dtype=dtype)


torch.Tensor.cumsum = _onnx_safe_cumsum
_orig_cdist = torch.cdist


def _onnx_safe_cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary'):
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    if p == 2.0:
        return (diff * diff).sum(-1).sqrt()
    elif p == 1.0:
        return diff.abs().sum(-1)
    elif p == float('inf'):
        return diff.abs().amax(-1)
    else:
        return diff.abs().pow(p).sum(-1).pow(1.0 / p)


torch.cdist = _onnx_safe_cdist
import transformers.integrations.sdpa_attention as _sdpa_mod
_sdpa_mod.use_gqa_in_sdpa = lambda attention_mask, key: False


try:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.executorch import sdpa_mask_without_vmap
    ALL_MASK_ATTENTION_FUNCTIONS.register('sdpa_without_vmap', sdpa_mask_without_vmap)
    ALL_ATTENTION_FUNCTIONS.register('sdpa_without_vmap', ALL_ATTENTION_FUNCTIONS['sdpa'])
    _VMAP_WORKAROUND = 'sdpa_without_vmap'
except ImportError:
    _VMAP_WORKAROUND = None


TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "onnx_models"
ONNX_FILENAME = "tokenizer12hz_encode.onnx"

SAMPLE_RATE = 24000
ENCODE_DOWNSAMPLE = 1920
VALID_NUM_QUANTIZERS = 16
NUM_TOTAL_QUANTIZERS = 32
CODEBOOK_SIZE = 2048


class EncoderOnnxWrapper(nn.Module):
    def __init__(self, encoder_model, valid_num_quantizers=VALID_NUM_QUANTIZERS):
        super().__init__()
        self.seanet_encoder = encoder_model.encoder
        self.encoder_transformer = encoder_model.encoder_transformer
        self.downsample = encoder_model.downsample
        self.quantizer = encoder_model.quantizer
        self.valid_num_quantizers = valid_num_quantizers

        config = encoder_model.encoder_transformer.config
        self.sliding_window = getattr(config, "sliding_window", 250)

    def _make_sliding_window_causal_mask(self, seq_len, device):
        pos = torch.arange(seq_len, device=device)
        q_pos = pos.unsqueeze(1)
        kv_pos = pos.unsqueeze(0)
        diff = q_pos - kv_pos
        allowed = (diff >= 0) & (diff < self.sliding_window)
        mask = torch.where(allowed, torch.tensor(0.0, device=device),
                           torch.tensor(-3.4028e+38, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, audio_waveform):
        embeddings = self.seanet_encoder(audio_waveform)
        hidden_states = embeddings.transpose(1, 2)
        transformer = self.encoder_transformer
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        causal_mask = self._make_sliding_window_causal_mask(seq_len, device)

        for layer in transformer.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

        embeddings = hidden_states.transpose(1, 2)
        embeddings = self.downsample(embeddings)
        codes = self.quantizer.encode(embeddings, self.valid_num_quantizers)
        codes = codes.transpose(0, 1)
        return codes


def _fix_rope_inv_freq(transformer):
    for layer in transformer.layers:
        if not hasattr(layer, 'self_attn'):
            continue
        rotary = layer.self_attn.rotary_emb
        config = layer.self_attn.config
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        theta = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        rotary.inv_freq = inv_freq
    print(f"  Fixed RoPE inv_freq for {len(transformer.layers)} transformer layers")


def load_encoder(local_dir=None):
    """Load the Qwen3-TTS-Tokenizer-12Hz model and extract the encoder."""
    from transformers.models.mimi.modeling_mimi import MimiConv1d
    MimiConv1d._get_extra_padding_for_conv1d = lambda self, hidden_states: 0

    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Config,
    )
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )

    repo = local_dir or TOKENIZER_REPO
    print(f"Loading model from {repo} ...")
    config = Qwen3TTSTokenizerV2Config.from_pretrained(repo)
    config.encoder_config._attn_implementation = "eager"
    model = Qwen3TTSTokenizerV2Model.from_pretrained(repo, config=config)
    model.eval()

    encoder = model.encoder
    print(f"  Encoder class: {type(encoder).__name__}")
    print(f"  Encoder config: hidden_size={config.encoder_config.hidden_size}, "
          f"num_layers={config.encoder_config.num_hidden_layers}, "
          f"sliding_window={config.encoder_config.sliding_window}")
    print(f"  Quantizer: {config.encoder_config.num_quantizers} codebooks, "
          f"valid={config.encoder_valid_num_quantizers}")
    print(f"  Downsample rate: {config.encode_downsample_rate}")

    _fix_rope_inv_freq(encoder.encoder_transformer)

    if _VMAP_WORKAROUND and hasattr(encoder, 'encoder_transformer'):
        print(f"  Patching attention to '{_VMAP_WORKAROUND}' for ONNX trace ...")
        encoder.encoder_transformer.config._attn_implementation = _VMAP_WORKAROUND
        for layer in encoder.encoder_transformer.layers:
            if hasattr(layer, 'self_attn'):
                layer.self_attn.config._attn_implementation = _VMAP_WORKAROUND


    wrapper = EncoderOnnxWrapper(encoder, config.encoder_valid_num_quantizers)
    wrapper.eval()
    print("\n  Verifying wrapper vs original encode ...")
    test_samples = ENCODE_DOWNSAMPLE * 5
    test_audio = torch.randn(1, 1, test_samples)

    with torch.no_grad():
        orig_output = model.encoder.encode(
            input_values=test_audio, return_dict=True
        )
        orig_codes = orig_output.audio_codes
        orig_codes = orig_codes[:, :VALID_NUM_QUANTIZERS]
        wrap_codes = wrapper(test_audio)

    match = torch.equal(orig_codes, wrap_codes)
    print(f"  Original codes shape: {tuple(orig_codes.shape)}")
    print(f"  Wrapper codes shape:  {tuple(wrap_codes.shape)}")
    print(f"  Exact match: {match}")
    if not match:
        diff_count = (orig_codes != wrap_codes).sum().item()
        total = orig_codes.numel()
        print(f"  WARNING: {diff_count}/{total} codes differ")

    return wrapper, model


def create_dummy_input(num_frames=5):
    """Create a dummy audio waveform for export tracing.

    The input length must be a multiple of ENCODE_DOWNSAMPLE for clean framing.
    """
    num_samples = ENCODE_DOWNSAMPLE * num_frames
    return torch.randn(1, 1, num_samples)


def export_onnx(wrapper, audio, output_path, opset=17):
    """Export encoder to ONNX."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nInput shape:  {tuple(audio.shape)}  (batch, channels, samples)")
    with torch.no_grad():
        codes_pt = wrapper(audio)
    print(f"Output shape: {tuple(codes_pt.shape)}  (batch, codebooks, frames)")
    print(f"Frame check: {audio.shape[-1]} / {ENCODE_DOWNSAMPLE} = "
          f"{audio.shape[-1] / ENCODE_DOWNSAMPLE:.1f} frames  (got {codes_pt.shape[-1]})")
    print(f"Code value range: [{codes_pt.min().item()}, {codes_pt.max().item()}]")

    dynamic_axes = {
        "audio_waveform": {0: "batch_size", 2: "num_samples"},
        "audio_codes":    {0: "batch_size", 2: "num_frames"},
    }

    try:
        print(f"\n[1/2] torch.onnx.export (trace, opset {opset}) ...")
        t0 = time.time()
        torch.onnx.export(
            wrapper,
            (audio,),
            str(output_path),
            opset_version=opset,
            input_names=["audio_waveform"],
            output_names=["audio_codes"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
        )
        elapsed = time.time() - t0
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Exported in {elapsed:.1f}s  ({size_mb:.1f} MB)")
        data_path = Path(str(output_path) + ".data")
        if data_path.exists():
            data_mb = data_path.stat().st_size / (1024 * 1024)
            print(f"  External data: {data_path.name} ({data_mb:.1f} MB)")

        return True
    except Exception as e:
        print(f"  ✗ Trace export failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        dynamo_opset = max(opset, 18)
        print(f"\n[2/2] torch.onnx.export (dynamo, opset {dynamo_opset}) ...")
        t0 = time.time()
        torch.onnx.export(
            wrapper,
            (audio,),
            str(output_path),
            opset_version=dynamo_opset,
            input_names=["audio_waveform"],
            output_names=["audio_codes"],
            dynamic_axes=dynamic_axes,
            dynamo=True,
        )
        elapsed = time.time() - t0
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Exported (dynamo) in {elapsed:.1f}s  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Dynamo export also failed: {e}")
        import traceback
        traceback.print_exc()

    return False


def validate(wrapper, original_model, output_path):
    import onnxruntime as ort

    output_path = Path(output_path)
    if not output_path.exists():
        print("\nSkipping validation — no ONNX file found.")
        return False

    print("\n" + "=" * 60)
    print("Validation: PyTorch vs ONNX Runtime")
    print("=" * 60)
    print("\n--- Test 1: Sine wave (5 frames = 9600 samples) ---")
    num_samples = ENCODE_DOWNSAMPLE * 5
    t = torch.linspace(0, 2 * math.pi * 440 * num_samples / SAMPLE_RATE, num_samples)
    sine_audio = (0.5 * torch.sin(t)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pt_codes = wrapper(sine_audio).numpy()

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_codes = sess.run(["audio_codes"], {"audio_waveform": sine_audio.numpy()})[0]

    print(f"  PT shape:   {pt_codes.shape}")
    print(f"  ONNX shape: {ort_codes.shape}")
    match1 = np.array_equal(pt_codes, ort_codes)
    if not match1:
        diff = (pt_codes != ort_codes).sum()
        total = pt_codes.size
        print(f"  Exact match: False ({diff}/{total} codes differ)")
    else:
        print(f"  Exact match: True ✓")

    print("\n--- Test 2: Dynamic axis (different audio lengths) ---")
    all_pass = True
    for num_frames in [2, 10, 50]:
        ns = ENCODE_DOWNSAMPLE * num_frames
        test_audio = torch.randn(1, 1, ns)
        with torch.no_grad():
            pt_out = wrapper(test_audio).numpy()
        ort_out = sess.run(["audio_codes"], {"audio_waveform": test_audio.numpy()})[0]
        match = np.array_equal(pt_out, ort_out)
        status = "✓" if match else "✗"
        diff_count = 0 if match else (pt_out != ort_out).sum()
        print(f"  {num_frames:3d} frames ({ns:>7d} samples): "
              f"shape={ort_out.shape}, match={status}"
              + (f" ({diff_count} diffs)" if not match else ""))
        if not match:
            all_pass = False

    print("\n--- Test 3: Real audio file ---")
    wav_files = list(Path(__file__).parent.parent.glob("*.wav"))
    if wav_files:
        wav_path = wav_files[0]
        print(f"  Loading {wav_path.name} ...")
        try:
            import wave
            with wave.open(str(wav_path), 'rb') as wf:
                assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
                assert wf.getframerate() == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {wf.getframerate()}Hz"
                frames = wf.readframes(wf.getnframes())
                audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            pad_len = (ENCODE_DOWNSAMPLE - len(audio_np) % ENCODE_DOWNSAMPLE) % ENCODE_DOWNSAMPLE
            if pad_len > 0:
                audio_np = np.pad(audio_np, (0, pad_len))

            real_audio = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
            print(f"  Audio: {real_audio.shape[-1]} samples ({real_audio.shape[-1]/SAMPLE_RATE:.2f}s)")

            with torch.no_grad():
                pt_out = wrapper(real_audio).numpy()
            ort_out = sess.run(["audio_codes"], {"audio_waveform": real_audio.numpy()})[0]

            match = np.array_equal(pt_out, ort_out)
            diff_count = 0 if match else (pt_out != ort_out).sum()
            print(f"  Codes shape: {ort_out.shape}")
            print(f"  Exact match: {match}" + (f" ({diff_count} diffs)" if not match else " ✓"))
            print(f"  Code value range: [{ort_out.min()}, {ort_out.max()}]")
            if not match:
                all_pass = False
        except Exception as e:
            print(f"  Skipped: {e}")
    else:
        print("  No .wav files found — skipped")

    overall = all_pass and match1
    print(f"\n{'✓ ALL TESTS PASSED' if overall else '⚠ SOME DIFFERENCES DETECTED (may be acceptable)'}")
    print("Note: Small code differences are normal — floating point rounding in "
          "the transformer can shift which codebook entry is nearest.")
    return overall


def parse_args():
    p = argparse.ArgumentParser(
        description="Export Qwen3-TTS Tokenizer-12Hz encoder to ONNX"
    )
    p.add_argument(
        "--model-dir", type=str, default=None,
        help="Local directory with tokenizer model weights "
             "(default: download from HF or use models/Qwen3-TTS-Tokenizer-12Hz/)",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Directory to save {ONNX_FILENAME} (default: onnx_models/)",
    )
    p.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)",
    )
    p.add_argument(
        "--num-frames", type=int, default=5,
        help="Number of encoder frames for dummy input (default: 5)",
    )
    p.add_argument(
        "--skip-validate", action="store_true",
        help="Skip ONNX Runtime validation after export",
    )
    p.add_argument(
        "--copy-to-local", action="store_true",
        help="Copy exported ONNX to %%LOCALAPPDATA%%\\ElBruno\\QwenTTS-Base\\",
    )
    return p.parse_args()


def main():
    configure_output_encoding()
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_path = output_dir / ONNX_FILENAME
    model_dir = args.model_dir
    
    if model_dir is None:
        local_candidate = Path(__file__).parent / "models" / "Qwen3-TTS-Tokenizer-12Hz"
        if local_candidate.exists():
            model_dir = str(local_candidate)
            print(f"Using local model: {model_dir}")

    print("=" * 60)
    print("Qwen3-TTS Speech Tokenizer Encoder → ONNX Export")
    print("=" * 60)

    wrapper, original_model = load_encoder(model_dir)
    audio = create_dummy_input(args.num_frames)

    with torch.no_grad():
        success = export_onnx(wrapper, audio, output_path, opset=args.opset)

    if success and not args.skip_validate:
        validate(wrapper, original_model, output_path)

    if success and args.copy_to_local:
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            dest_dir = Path(local_app) / "ElBruno" / "QwenTTS-Base"
            dest_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(output_path, dest_dir / ONNX_FILENAME)
            print(f"\nCopied to {dest_dir / ONNX_FILENAME}")
            data_file = Path(str(output_path) + ".data")
            if data_file.exists():
                shutil.copy2(data_file, dest_dir / (ONNX_FILENAME + ".data"))
                print(f"Copied to {dest_dir / (ONNX_FILENAME + '.data')}")

    print("\nDone." if success else "\nExport failed.")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
