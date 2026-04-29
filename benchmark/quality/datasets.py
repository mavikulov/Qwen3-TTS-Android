from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _pick_text(example: dict[str, Any]) -> str | None:
    for key in ("text", "sentence", "transcript", "normalized_text"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _sample_dataset(
    dataset_name: str,
    split: str,
    count: int,
    seed: int,
    lang: str,
    id_key: str = "id",
    shuffle_buffer_size: int | None = None,
) -> list[dict[str, str]]:
    from datasets import load_dataset

    # Streaming mode lets us stop as soon as we collected `count` usable samples.
    ds = load_dataset(dataset_name, split=split, streaming=True)
    usable: list[dict[str, str]] = []
    buffer_size = shuffle_buffer_size if shuffle_buffer_size is not None else max(1000, count * 10)
    ds = ds.shuffle(buffer_size=buffer_size, seed=seed)

    for i, ex in enumerate(ds):
        text = _pick_text(ex)
        if not text:
            continue
        sample_id = str(ex.get(id_key, f"{lang}-{i}"))
        usable.append({"id": sample_id, "text": text, "lang": lang})
        if len(usable) >= count:
            break

    if len(usable) < count:
        raise ValueError(f"{dataset_name}:{split} has only {len(usable)} usable samples, requested {count}")
    return usable


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_manifests(project_root: Path, dataset_cfg: Any, seed: int) -> dict[str, list[dict[str, str]]]:
    """Returns dict with `ru` and `en` sampled rows.

    If manifests don't exist or recreate=true, creates frozen manifests.
    """
    manifests_dir = (project_root / str(dataset_cfg.manifests_dir)).resolve()
    ru_path = manifests_dir / str(dataset_cfg.ru_manifest_name)
    en_path = manifests_dir / str(dataset_cfg.en_manifest_name)

    recreate = bool(dataset_cfg.recreate_manifests)
    if recreate or not ru_path.exists() or not en_path.exists():
        ru_rows = _sample_dataset(
            dataset_name=str(dataset_cfg.ru.hf_dataset),
            split=str(dataset_cfg.ru.split),
            count=int(dataset_cfg.ru.count),
            seed=seed,
            lang="ru",
            id_key=str(dataset_cfg.ru.id_field),
        )
        en_rows = _sample_dataset(
            dataset_name=str(dataset_cfg.en.hf_dataset),
            split=str(dataset_cfg.en.split),
            count=int(dataset_cfg.en.count),
            seed=seed + 1,
            lang="en",
            id_key=str(dataset_cfg.en.id_field),
        )
        _write_manifest(ru_path, ru_rows)
        _write_manifest(en_path, en_rows)
    else:
        ru_rows = _read_manifest(ru_path)
        en_rows = _read_manifest(en_path)

    return {"ru": ru_rows, "en": en_rows}


def setup_hf_cache(project_root: Path) -> Path:
    """
    Force HuggingFace/Transformers/Datasets caches into the project workspace.

    This is important for reproducibility and for avoiding downloads into global user caches.
    """
    hf_cache_root = (project_root / "benchmark" / ".hf_cache").resolve()
    hub_cache = hf_cache_root / "hub"
    datasets_cache = hf_cache_root / "datasets"
    transformers_cache = hf_cache_root / "transformers"
    whisper_cache = hf_cache_root / "whisper"

    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    whisper_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_HOME"] = str(transformers_cache)

    return hf_cache_root
