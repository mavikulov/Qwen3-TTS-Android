from __future__ import annotations

from pathlib import Path

import numpy as np
from jiwer import cer, wer


class WhisperWerCerMetric:
    """Whisper-based ASR metric with WER/CER."""

    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "float32",
        normalize_text: bool = True,
        beam_size: int = 1,
        download_root: str | Path | None = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Package `faster-whisper` is required for WER/CER metric. Install dependencies first."
            ) from e
        # faster-whisper uses `download_root` to control where model files are stored.
        kwargs: dict[str, str] = {}
        if download_root is not None:
            kwargs["download_root"] = str(download_root)
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type, **kwargs)
        self._normalize_text = normalize_text
        self._beam_size = beam_size

    def _norm(self, text: str) -> str:
        out = text.strip()
        if self._normalize_text:
            out = out.lower()
        return out

    def score(
        self,
        wav_path: str | Path | np.ndarray,
        sample_rate: int | None,
        reference_text: str,
        lang_key: str,
    ) -> dict[str, str | float]:
        language = "ru" if lang_key == "ru" else "en"
        source = str(wav_path) if isinstance(wav_path, (str, Path)) else wav_path
        segments, _ = self._model.transcribe(source, language=language, beam_size=self._beam_size)
        hyp = " ".join(seg.text.strip() for seg in segments).strip()
        ref_n = self._norm(reference_text)
        hyp_n = self._norm(hyp)
        return {
            "asr_hyp": hyp,
            "wer": float(wer(ref_n, hyp_n)),
            "cer": float(cer(ref_n, hyp_n)),
        }
