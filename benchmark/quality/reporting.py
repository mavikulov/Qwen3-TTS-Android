from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def by_lang(lang: str) -> list[dict[str, Any]]:
        return [r for r in rows if r.get("lang") == lang]

    def aggregate_chunk(chunk: list[dict[str, Any]]) -> dict[str, Any]:
        w = [r["wer"] for r in chunk if r.get("wer") is not None]
        c = [r["cer"] for r in chunk if r.get("cer") is not None]
        d = [r["dnsmos_overall"] for r in chunk if r.get("dnsmos_overall") is not None]
        l = [r["latency_ms"] for r in chunk if r.get("latency_ms") is not None]
        return {
            "count": len(chunk),
            "wer_mean": _mean_or_none(w),
            "cer_mean": _mean_or_none(c),
            "dnsmos_mean": _mean_or_none(d),
            "latency_ms_mean": _mean_or_none(l),
        }

    ru = by_lang("ru")
    en = by_lang("en")
    return {
        "ru": aggregate_chunk(ru),
        "en": aggregate_chunk(en),
        "overall": aggregate_chunk(rows),
    }


def write_quality_reports(run_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any], meta: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "per_sample.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary}, f, ensure_ascii=False, indent=2)

    if rows:
        fields = sorted({k for r in rows for k in r.keys()})
        with open(run_dir / "per_sample.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
