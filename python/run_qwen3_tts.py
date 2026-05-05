from pathlib import Path
import argparse
import logging
import datetime

from package.utils import set_seed, resolve_path, setup_logging, save_wav
from package.inference_engine import QwenTTSPipeline


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--text", type=str, default="Hello from inference pipeline")
    parser.add_argument("--instruct", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup_logging(level="INFO")
    pipeline = QwenTTSPipeline(config_path=args.config)
    waveform = pipeline(text="I would like a cup of tea please")
    
    save_dir = resolve_path("./saved_wav")
    save_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wav_path = save_dir / f"output_wav_{timestamp}.wav"
    save_wav(wav_path, waveform, sample_rate=24000)
    LOGGER.info("Audio saved to: %s", wav_path)
