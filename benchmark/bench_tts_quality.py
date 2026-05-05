from __future__ import annotations

import json
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import hydra
from hydra.utils import instantiate
import numpy as np
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf

from quality.datasets import ensure_manifests, setup_hf_cache
from quality.reporting import aggregate_results, write_quality_reports


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _score_metric(metric: Any, wav: Any, sample_rate: int | None, reference_text: str, lang_key: str) -> dict[str, Any]:
    try:
        return metric.score(wav, sample_rate, reference_text, lang_key)
    except TypeError:
        pass
    try:
        return metric.score(wav, reference_text, lang_key)
    except TypeError:
        pass
    try:
        return metric.score(wav, sample_rate)
    except TypeError:
        pass
    if isinstance(wav, (np.ndarray, list)) and sample_rate is not None:
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, np.asarray(wav, dtype=np.float32), sample_rate)
            tmp_path = Path(tmp.name)
        try:
            try:
                return metric.score(tmp_path, reference_text, lang_key)
            except TypeError:
                return metric.score(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return metric.score(wav)


def _score_metrics(metrics: dict[str, Any], wav: Any, sample_rate: int | None, reference_text: str, lang_key: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for metric in metrics.values():
        row.update(_score_metric(metric, np.squeeze(wav), sample_rate, reference_text, lang_key))
    return row


@hydra.main(version_base=None, config_path="configs", config_name="quality_lite")
def main(cfg: DictConfig) -> None:
    _set_seed(int(cfg.run.seed))

    project_root = Path(hydra.utils.get_original_cwd())
    hf_cache_root = setup_hf_cache(project_root)
    result_dir = (project_root / cfg.run.result_dir).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = result_dir / run_id
    
    save_audio = bool(cfg.run.get("save_audio", True))
    audio_dir = run_dir / "audio" if save_audio else None
    if save_audio and audio_dir is not None:
        audio_dir.mkdir(parents=True, exist_ok=True)

    manifests = ensure_manifests(
        project_root=project_root,
        dataset_cfg=cfg.dataset,
        seed=int(cfg.run.seed),
    )

    adapter = instantiate(cfg.model)

    metrics: dict[str, Any] = {}
    for metric_name, metric_cfg in cfg.metrics.items():
        if not metric_cfg.enabled:
            continue

        overrides: dict[str, Any] = {}
        if metric_name == "wer_cer" and metric_cfg.metric.get("download_root") is None:
            overrides["download_root"] = str((hf_cache_root / "whisper").resolve())

        metrics[metric_name] = instantiate(metric_cfg.metric, **overrides)

    all_rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    for lang_key, samples in manifests.items():
        language_name = cfg.model.language_map[lang_key]
        for idx, sample in enumerate(samples):
            t0 = time.perf_counter()
            audio, sr = adapter.generate(text=sample["text"], language=language_name)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            wav_path = None
            if save_audio and audio_dir is not None:
                wav_path = audio_dir / f"{lang_key}_{idx:04d}.wav"
                sf.write(wav_path, np.squeeze(audio), sr)

            row: dict[str, Any] = {
                "sample_id": sample["id"],
                "lang": lang_key,
                "text": sample["text"],
                "audio_path": str(wav_path) if wav_path is not None else None,
                "sample_rate": int(sr),
                "latency_ms": latency_ms,
            }

            if metrics:
                row.update(_score_metrics(metrics, audio, int(sr), sample["text"], lang_key))

            all_rows.append(row)

    elapsed_s = time.perf_counter() - started_at
    summary = aggregate_results(all_rows)

    meta = {
        "run_id": run_id,
        "config_name": cfg.run.config_name,
        "model_id": cfg.model.get("variant", "qwen3_tts_onnx"),
        "voice": str(cfg.model.get("speaker_id", "default")),
        "seed": int(cfg.run.seed),
        "metrics_enabled": list(metrics.keys()),
        "duration_s": elapsed_s,
    }

    write_quality_reports(run_dir=run_dir, rows=all_rows, summary=summary, meta=meta)

    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2, ensure_ascii=False)
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    print(f"Quality benchmark completed. Run dir: {run_dir}")


if __name__ == "__main__":
    main()