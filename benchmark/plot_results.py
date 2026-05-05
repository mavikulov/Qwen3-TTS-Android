import argparse
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


class BenchmarkDataLoader:
    def __init__(self, file_paths: list[str], labels: list[str]):
        if len(file_paths) != len(labels):
            raise ValueError("Количество файлов должно совпадать с количеством лейблов")
        
        self.labels: list[str] = labels
        self.data: list[list[dict[str, Any]]] = self._load_all_files(file_paths)
        self.concurrencies: list[int] = self._get_all_concurrencies()

    def _load_all_files(self, file_paths: list[str]) -> list[list[dict[str, Any]]]:
        all_results = []
        for f in file_paths:
            path = Path(f)
            if not path.exists():
                raise FileNotFoundError(f"Файл результатов не найден: {f}")
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError(f"Ожидался список результатов в {f}, получен {type(data)}")
            all_results.append(data)
        return all_results

    def _get_all_concurrencies(self) -> list[int]:
        all_sets = [set(r["concurrency"] for r in results) for results in self.data]
        return sorted(set.union(*all_sets))

    def get_metrics_for_plotting(self) -> dict[str, dict[str, list[Optional[float]]]]:
        metrics_map = {
            "ttfp": "mean_ttfp_ms",
            "e2e": "mean_e2e_ms",
            "rtf": "mean_rtf",
            "throughput": "audio_throughput"
        }
        
        result_data = {
            "ttfp": {label: [] for label in self.labels},
            "e2e": {label: [] for label in self.labels},
            "rtf": {label: [] for label in self.labels},
            "throughput": {label: [] for label in self.labels},
        }

        for results, label in zip(self.data, self.labels):
            conc_map = {r["concurrency"]: r for r in results}
            for c in self.concurrencies:
                r = conc_map.get(c)
                result_data["ttfp"][label].append(r[metrics_map["ttfp"]] if r else None)
                result_data["e2e"][label].append(r[metrics_map["e2e"]] if r else None)
                result_data["rtf"][label].append(r[metrics_map["rtf"]] if r else None)
                result_data["throughput"][label].append(r[metrics_map["throughput"]] if r else None)
        
        return result_data

    def get_single_summary_data(self, index: int) -> dict[str, list[float]]:
        results = self.data[index]
        label = self.labels[index]
        concurrencies = [r["concurrency"] for r in results]
        
        data = {
            "ttfp": {
                "mean": [r["mean_ttfp_ms"] for r in results],
                "median": [r["median_ttfp_ms"] for r in results],
                "p90": [r["p90_ttfp_ms"] for r in results],
                "p99": [r["p99_ttfp_ms"] for r in results],
            },
            "e2e": {
                "mean": [r["mean_e2e_ms"] for r in results],
                "median": [r["median_e2e_ms"] for r in results],
                "p90": [r["p90_e2e_ms"] for r in results],
                "p99": [r["p99_e2e_ms"] for r in results],
            },
            "rtf": {
                "mean": [r["mean_rtf"] for r in results],
                "median": [r["median_rtf"] for r in results],
            },
            "concurrencies": concurrencies,
            "label": label
        }
        return data

    def print_comparison_table(self):
        concurrencies = self.concurrencies
        labels = self.labels
        all_results = self.data

        print("\n## Benchmark Results\n")
        header = "| Metric | Concurrency |"
        sep = "| --- | --- |"
        for label in labels:
            header += f" {label} |"
            sep += " --- |"
        print(header)
        print(sep)

        metrics_config = [
            ("TTFP (ms)", "mean_ttfp_ms", ".1f"),
            ("E2E (ms)", "mean_e2e_ms", ".1f"),
            ("RTF", "mean_rtf", ".3f"),
            ("Throughput (audio-s/s)", "audio_throughput", ".2f"),
        ]

        for metric, key, fmt in metrics_config:
            for c in concurrencies:
                row = f"| {metric} | {c} |"
                for results in all_results:
                    conc_map = {r["concurrency"]: r for r in results}
                    val = conc_map.get(c, {}).get(key, 0)
                    row += f" {val:{fmt}} |"
                print(row)

        if len(all_results) == 2:
            print(f"\n## Improvement ({labels[0]} vs {labels[1]})\n")
            print("| Metric | Concurrency | Improvement |")
            print("| --- | --- | --- |")
            for metric, key in [("TTFP", "mean_ttfp_ms"), ("E2E", "mean_e2e_ms"), ("RTF", "mean_rtf")]:
                for c in concurrencies:
                    m0 = {r["concurrency"]: r for r in all_results[0]}
                    m1 = {r["concurrency"]: r for r in all_results[1]}
                    v0 = m0.get(c, {}).get(key, 0)
                    v1 = m1.get(c, {}).get(key, 0)
                    if v1 > 0:
                        pct = (v1 - v0) / v1 * 100
                        print(f"| {metric} | {c} | {pct:+.1f}% |")


class BenchmarkPlotter:
    def __init__(self, output_dir: str):
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors: list[str] = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107"]

    def plot_comparison(
        self,
        labels: list[str],
        concurrencies: list[int],
        plot_data: dict[str, dict[str, list[Optional[float]]]],
        output_filename: str,
        title_prefix: str = "Qwen3-TTS"
    ):
        n_configs = len(labels)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{title_prefix} Performance Benchmark", fontsize=16, fontweight="bold")

        x = np.arange(len(concurrencies))
        width = 0.35 if n_configs == 2 else 0.5
        if n_configs > 1:
            offsets = np.linspace(-width / 2 * (n_configs - 1), width / 2 * (n_configs - 1), n_configs)
        else:
            offsets = [0]

        metric_titles = {
            "ttfp": ("TTFP (ms)", "Time to First Audio Packet (TTFP)", ".1f"),
            "e2e": ("E2E Latency (ms)", "End-to-End Latency (E2E)", ".1f"),
            "rtf": ("RTF", "Real-Time Factor (RTF)", ".3f"),
            "throughput": ("Audio-sec / Wall-sec", "Audio Throughput", ".2f")
        }

        for idx, (key, (ylabel, title, fmt)) in enumerate(metric_titles.items()):
            ax = axes.flat[idx]
            data_dict = plot_data[key]
            bars = []
            
            for i, label in enumerate(labels):
                values = data_dict[label]
                plot_values = [v if v is not None else 0 for v in values]
                color = self.colors[i % len(self.colors)]
                bar = ax.bar(x + offsets[i], plot_values, width, label=label, color=color, alpha=0.85)
                bars.append(bar)
                
                max_val = max((v for v in values if v is not None), default=1)
                for rect, val in zip(bar, values):
                    if val is not None and val > 0:
                        ax.text(
                            rect.get_x() + rect.get_width() / 2,
                            rect.get_height() + max_val * 0.02,
                            f"{val:{fmt}}",
                            ha="center", va="bottom", fontsize=9, fontweight="bold"
                        )
            
            ax.set_xlabel("Concurrency", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in concurrencies])
            ax.legend(fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)

        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to {output_path}")
        plt.close()

    def plot_single_summary(self, data: dict[str, Any], output_filename: str):
        concurrencies = data["concurrencies"]
        label = data["label"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Qwen3-TTS Benchmark - {label}", fontsize=15, fontweight="bold")

        x = np.arange(len(concurrencies))
        w = 0.2

        ax = axes[0]
        metrics = ["mean", "median", "p90", "p99"]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
        ttfp_data = data["ttfp"]
        
        for i, metric in enumerate(metrics):
            vals = ttfp_data[metric]
            ax.bar(x + (i - 1.5) * w, vals, w, label=metric, color=colors[i])
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in concurrencies])
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("TTFP (ms)")
        ax.set_title("Time to First Audio Packet")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1]
        e2e_data = data["e2e"]
        for i, metric in enumerate(metrics):
            vals = e2e_data[metric]
            ax.bar(x + (i - 1.5) * w, vals, w, label=metric, color=colors[i])
            
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in concurrencies])
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("E2E Latency (ms)")
        ax.set_title("End-to-End Latency")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        ax = axes[2]
        rtf_data = data["rtf"]
        ax.bar(x - 0.15, rtf_data["mean"], 0.3, label="mean", color="#2196F3")
        ax.bar(x + 0.15, rtf_data["median"], 0.3, label="median", color="#4CAF50")
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in concurrencies])
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("RTF")
        ax.set_title("Real-Time Factor")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Summary plot saved to {output_path}")
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Qwen3-TTS benchmark results")
    parser.add_argument(
        "--results", type=str, nargs="+", required=True, 
        help="Path(s) to result JSON files (one per config)"
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", required=True, 
        help="Labels for each config (must match --results count)"
    )
    parser.add_argument(
        "--output", type=str, default="results/qwen3_tts_benchmark.png", 
        help="Output image path (filename only or full path)"
    )
    parser.add_argument(
        "--title", type=str, default="Qwen3-TTS", 
        help="Title prefix for the plot"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert len(args.results) == len(args.labels), "--results and --labels must have the same count"
    loader = BenchmarkDataLoader(args.results, args.labels)
    loader.print_comparison_table()
    output_path = Path(args.output)
    plotter = BenchmarkPlotter(output_dir=str(output_path.parent))

    if len(loader.data) == 1:
        summary_data = loader.get_single_summary_data(0)
        plotter.plot_single_summary(summary_data, output_path.name)
    else:
        plot_data = loader.get_metrics_for_plotting()
        plotter.plot_comparison(
            labels=loader.labels,
            concurrencies=loader.concurrencies,
            plot_data=plot_data,
            output_filename=output_path.name,
            title_prefix=args.title
        )
