import sys
import yaml
import random
import logging
from pathlib import Path
from dataclasses import dataclass

import onnx
import onnxruntime as ort
import numpy as np
import soundfile as sf


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelSessions:
    prefill: ort.InferenceSession
    decode: ort.InferenceSession
    cp: ort.InferenceSession
    vocoder: ort.InferenceSession
    
    @classmethod
    def load(cls, model_files: dict, loader_func):
        return cls(
            prefill=loader_func(model_files["prefill"]),
            decode=loader_func(model_files["decode"]),
            cp=loader_func(model_files["cp"]),
            vocoder=loader_func(model_files["vocoder"]),
        )


def get_model_size_mb(path_to_model: Path) -> float:
    model = onnx.load(path_to_model)
    return round(model.ByteSize() / (1024 * 1024), 2)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def load_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = True
    
    return ort.InferenceSession(
        path_or_bytes=str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )


def resolve_path(raw_path: Path) -> Path:
    return Path(raw_path).expanduser().resolve()


def set_seed(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def read_config(path: Path) -> dict[str, dict[str, str | float]]:
    with open(path, "r", encoding='utf-8') as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.SafeLoader)


def save_wav(path: Path, signal: np.ndarray, sample_rate: int = 24000) -> None:
    arr = signal
    if arr.ndim == 3: 
        arr = arr[0, 0]
    elif arr.ndim == 2: 
        arr = arr[0]
    elif arr.ndim != 1:
        raise ValueError(f"Unexpected waveform shape: {signal.shape}")
    sf.write(str(path), arr, sample_rate)
