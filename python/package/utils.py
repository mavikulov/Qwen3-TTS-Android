import sys
import json
import yaml
import random
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Any, Optional

import onnx
import numpy as np
import soundfile as sf
import onnxruntime as ort


@dataclass(frozen=True)
class ModelSessions:
    prefill: ort.InferenceSession
    decode: ort.InferenceSession
    cp: ort.InferenceSession
    vocoder: ort.InferenceSession
    
    @classmethod
    def load(cls, model_files: dict):
        return cls(
            prefill=ort.InferenceSession(model_files["talker_prefill"]),
            decode=ort.InferenceSession(model_files["talker_decode"]),
            cp=ort.InferenceSession(model_files["code_predictor"]),
            vocoder=ort.InferenceSession(model_files["vocoder"])
        )


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def resolve_path(raw_path: str) -> Path:
    return Path(raw_path).expanduser().resolve()


def get_param(config: dict, key: str) -> Union[int, float, str]:
    return config.get(key, None)


def load_yaml(yaml_path: Path) -> Any:
    with open(yaml_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def load_config(json_path: Path) -> Any:
    with open(json_path, "r") as json_file:
        return json.load(json_file)
    

def get_model_size_mb(path_to_model: Path) -> float:
    model = onnx.load(path_to_model)
    return round(model.ByteSize() / (1024 * 1024), 2)


def save_wav(path: Path, waveform: np.ndarray, sample_rate: int = 24000) -> None:
    arr = waveform
    if arr.ndim == 3: 
        arr = arr[0, 0]
    elif arr.ndim == 2: 
        arr = arr[0]
    elif arr.ndim != 1:
        raise ValueError(f"Unexpected waveform shape: {waveform.shape}")
    sf.write(str(path), arr, sample_rate)
