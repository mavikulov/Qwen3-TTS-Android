from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


class DNSMOSMetric:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        try:
            from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
        except ImportError as e:
            raise ImportError(
                "Package `torchmetrics` is required for DNSMOS metric. Install dependencies first."
            ) from e
        self._scorer = DeepNoiseSuppressionMeanOpinionScore(fs=self.sample_rate, personalized=False)

    def _resample_if_needed(self, audio: np.ndarray, src_sr: int) -> torch.Tensor:
        mono = audio.mean(axis=1) if audio.ndim > 1 else audio
        wav = torch.tensor(mono, dtype=torch.float32).unsqueeze(0)
        if src_sr == self.sample_rate:
            return wav
        new_len = int(round(wav.shape[-1] * self.sample_rate / src_sr))
        return F.interpolate(wav.unsqueeze(1), size=new_len, mode="linear", align_corners=False).squeeze(1)

    def score(self, wav_path: str | Path | np.ndarray, sample_rate: int | None = None) -> dict[str, Any]:
        if isinstance(wav_path, (str, Path)):
            audio, sr = sf.read(str(wav_path), dtype="float32")
        else:
            if sample_rate is None:
                raise ValueError("sample_rate must be provided when scoring from audio data")
            audio = np.asarray(wav_path, dtype=np.float32)
            sr = int(sample_rate)

        wav = self._resample_if_needed(audio, int(sr))
        result = self._scorer(wav).detach().cpu().flatten().tolist()
        # torchmetrics DNSMOS order: [p808, sig, bak, overall]
        p808, sig, bak, overall = (result + [None, None, None, None])[:4]
        return {
            "dnsmos_overall": float(overall) if overall is not None else None,
            "dnsmos_sig": float(sig) if sig is not None else None,
            "dnsmos_bak": float(bak) if bak is not None else None,
            "dnsmos_p808": float(p808) if p808 is not None else None,
        }
