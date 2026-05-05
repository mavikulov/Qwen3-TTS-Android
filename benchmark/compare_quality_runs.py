from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare quality benchmark runs")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories containing summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for run in args.runs:
        summary_path = Path(run) / "summary.json"
        data = _load_summary(summary_path)
        meta = data["meta"]
        overall = data["summary"]["overall"]
        rows.append(
            {
                "run_id": meta.get("run_id", Path(run).name),
                "config_name": meta.get("config_name", "n/a"),
                "model_id": meta.get("model_id", "n/a"),
                "wer": overall.get("wer_mean"),
                "cer": overall.get("cer_mean"),
                "dnsmos": overall.get("dnsmos_mean"),
            }
        )

    print("\n## Quality leaderboard\n")
    print("| run_id | config_name | model_id | WER | CER | DNSMOS |")
    print("| --- | --- | --- | ---: | ---: | ---: |")
    for r in rows:
        print(
            f"| {r['run_id']} | {r['config_name']} | {r['model_id']} | "
            f"{(r['wer'] if r['wer'] is not None else float('nan')):.4f} | "
            f"{(r['cer'] if r['cer'] is not None else float('nan')):.4f} | "
            f"{(r['dnsmos'] if r['dnsmos'] is not None else float('nan')):.4f} |"
        )


if __name__ == "__main__":
    main()
