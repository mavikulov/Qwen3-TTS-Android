"""Prepare frozen manifests for quality benchmark datasets."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from quality.datasets import ensure_manifests, setup_hf_cache


@hydra.main(version_base=None, config_path="configs", config_name="quality_big")
def main(cfg: DictConfig) -> None:
    project_root = Path(hydra.utils.get_original_cwd())
    setup_hf_cache(project_root)
    manifests = ensure_manifests(
        project_root=project_root,
        dataset_cfg=cfg.dataset,
        seed=int(cfg.run.seed),
    )
    print(f"Prepared manifests: RU={len(manifests['ru'])}, EN={len(manifests['en'])}")


if __name__ == "__main__":
    main()
