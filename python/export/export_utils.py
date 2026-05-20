import os
import re
import sys
from pathlib import Path


def configure_output_encoding():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


SUPPORTED_CUSTOMVOICE_REPOS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
]

SUPPORTED_BASE_REPOS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
]

SUPPORTED_TOKENIZER_REPOS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
]

ALL_SUPPORTED_REPOS = (
    SUPPORTED_CUSTOMVOICE_REPOS
    + SUPPORTED_BASE_REPOS
    + SUPPORTED_TOKENIZER_REPOS
)

KNOWN_UNSUPPORTED_REPOS = [
    "Qwen/Qwen3-TTS-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-0.6B-Base",
    "Qwen/Qwen3-TTS-1.7B-Base",
]

HF_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")


class ExportValidationError(Exception):
    def __init__(self, message: str, suggestion: str | None = None):
        self.suggestion: str | None = suggestion
        full_msg = message
        if suggestion:
            full_msg += f"\n\nSuggestion: {suggestion}"
        super().__init__(full_msg)


def is_hf_repo_id(path_or_repo: str) -> bool:
    if os.path.sep in path_or_repo or (os.name == "nt" and "\\" in path_or_repo):
        return False
    if path_or_repo.startswith((".", "/", "~")):
        return False
    if ":" in path_or_repo and len(path_or_repo) > 1 and path_or_repo[1] == ":":
        return False 
    if not HF_REPO_PATTERN.match(path_or_repo):
        return False

    org = path_or_repo.split("/")[0]
    path_like_prefixes = {
        "models", "model", "src", "python", "data", "output", "outputs",
        "onnx", "onnx_models", "onnx_runtime", "weights", "checkpoints",
        "tmp", "temp", "build", "dist", "lib", "bin", "var", "etc",
        "home", "usr", "opt",
    }
    if org.lower() in path_like_prefixes:
        return False
    return True


def validate_model_dir(model_dir: str, require_config: bool = True) -> Path:
    if is_hf_repo_id(model_dir):
        suggestion = _suggest_for_repo_id(model_dir)
        raise ExportValidationError(
            f"'{model_dir}' looks like a HuggingFace repo ID, not a local directory.\n"
            f"The export scripts require locally downloaded model files.",
            suggestion=suggestion,
        )

    path = Path(model_dir)

    if not path.exists():
        raise ExportValidationError(
            f"Model directory does not exist: {path.resolve()}\n"
            f"Download the model first with:\n"
            f"  python download_models.py --model customvoice",
        )

    if not path.is_dir():
        raise ExportValidationError(
            f"Model path is not a directory: {path.resolve()}",
        )

    if require_config:
        config_path = path / "config.json"
        if not config_path.exists():
            raise ExportValidationError(
                f"No config.json found in {path.resolve()}\n"
                f"This directory does not appear to contain a valid Qwen3-TTS model.",
                suggestion="Ensure you downloaded the full model, not just weights.",
            )

    return path.resolve()


def validate_repo_id(repo_id: str, script_type: str = "lm") -> str:
    if not HF_REPO_PATTERN.match(repo_id):
        raise ExportValidationError(
            f"'{repo_id}' is not a valid HuggingFace repo ID.\n"
            f"Expected format: org/model-name (e.g., Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)",
        )

    if repo_id in KNOWN_UNSUPPORTED_REPOS:
        supported = _get_supported_for_type(script_type)
        raise ExportValidationError(
            f"Repo '{repo_id}' is not supported by the export scripts.\n"
            f"The non-12Hz Qwen3-TTS models use a different architecture that\n"
            f"is incompatible with ONNX export (causes vmap/functorch errors).",
            suggestion=f"Use one of these repos instead:\n  "
            + "\n  ".join(supported),
        )

    return repo_id


def validate_model_config_for_lm_export(config) -> dict:
    errors = []

    if not hasattr(config, "talker_config"):
        errors.append("Missing 'talker_config' — this model may not be a Qwen3-TTS model.")

    if hasattr(config, "talker_config"):
        tc = config.talker_config
        required_attrs = [
            "num_hidden_layers",
            "num_key_value_heads",
            "hidden_size",
            "vocab_size",
        ]
        for attr in required_attrs:
            if not hasattr(tc, attr):
                errors.append(f"Missing talker_config.{attr}")

        if hasattr(tc, "code_predictor_config"):
            cp = tc.code_predictor_config
            for attr in required_attrs:
                if not hasattr(cp, attr):
                    errors.append(f"Missing talker_config.code_predictor_config.{attr}")
        else:
            errors.append("Missing 'talker_config.code_predictor_config'")

        if not hasattr(tc, "num_code_groups"):
            errors.append("Missing 'talker_config.num_code_groups'")

    if errors:
        raise ExportValidationError(
            "Model config is missing required attributes for LM export:\n  "
            + "\n  ".join(errors),
            suggestion="Ensure you are using a Qwen3-TTS model (0.6B or 1.7B CustomVoice/Base).",
        )

    return {"valid": True}


def validate_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ExportValidationError(
            f"Cannot create output directory '{path.resolve()}': {e}",
        )
    return path.resolve()


def _suggest_for_repo_id(repo_id: str) -> str:
    if repo_id in KNOWN_UNSUPPORTED_REPOS:
        corrected = repo_id.replace("Qwen/Qwen3-TTS-", "Qwen/Qwen3-TTS-12Hz-")
        return (
            f"The repo '{repo_id}' uses a non-12Hz architecture that is incompatible\n"
            f"with these export scripts. Use the 12Hz variant instead:\n"
            f"  1. Download: python download_models.py\n"
            f"     (this downloads {corrected})\n"
            f"  2. Then export: python export_lm.py --model-dir models/{_repo_to_local(corrected)}"
        )

    if repo_id in ALL_SUPPORTED_REPOS:
        local_name = _repo_to_local(repo_id)
        return (
            f"Download the model first, then use the local path:\n"
            f"  1. python download_models.py\n"
            f"  2. python export_lm.py --model-dir models/{local_name}"
        )

    return (
        f"Download a supported model first:\n"
        f"  python download_models.py --model customvoice\n"
        f"Then run the export with the local path:\n"
        f"  python export_lm.py --model-dir models/Qwen3-TTS-0.6B-CustomVoice"
    )


def _repo_to_local(repo_id: str) -> str:
    name = repo_id.split("/")[-1]
    name = name.replace("12Hz-", "")
    return name


def _get_supported_for_type(script_type: str) -> list[str]:
    if script_type in ("vocoder", "speech_tokenizer"):
        return SUPPORTED_TOKENIZER_REPOS
    if script_type == "speaker_encoder":
        return SUPPORTED_BASE_REPOS
    return SUPPORTED_CUSTOMVOICE_REPOS + SUPPORTED_BASE_REPOS
