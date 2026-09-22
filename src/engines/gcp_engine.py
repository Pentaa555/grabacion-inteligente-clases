"""Motor GCP: Google Cloud Speech-to-Text con diarización de hablantes.

Requiere:
  - `pip install google-cloud-speech`
  - Credenciales, en cualquiera de estas formas (Application Default Credentials):
      * `gcloud auth application-default login`  (recomendado, sin clave JSON), o
      * GOOGLE_APPLICATION_CREDENTIALS -> ruta a clave JSON de Service Account.

Usa la API v1 con `SpeakerDiarizationConfig`. El resultado de GCP entrega las
palabras con su `speaker_tag`; aquí se agrupan en segmentos por hablante contiguo.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..types import Segment, TranscriptionResult
from .base import TranscriptionEngine


def _adc_path() -> Path:
    """Ruta por defecto del archivo ADC de gcloud."""
    return Path.home() / ".config" / "gcloud" / "application_default_credentials.json"


class GCPEngine(TranscriptionEngine):
    name = "gcp"

    def is_available(self) -> tuple[bool, str]:
        try:
            import google.cloud.speech  # noqa: F401
        except ImportError:
            return False, ("Falta la librería: pip install google-cloud-speech")
        # Aceptar clave JSON explícita O Application Default Credentials.
        has_key = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
        has_adc = _adc_path().exists()
        if not (has_key or has_adc):
            return False, (
                "Sin credenciales GCP. Ejecuta 'gcloud auth application-default "
                "login' o exporta GOOGLE_APPLICATION_CREDENTIALS."
            )
        return True, ("GCP disponible (clave JSON)." if has_key
                      else "GCP disponible (ADC).")

    def transcribe(self, audio_path, progress_cb=None) -> TranscriptionResult:
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(f"Motor gcp no disponible: {reason}")

        from google.cloud import speech

        client = speech.SpeechClient()
        audio_bytes = Path(audio_path).read_bytes()
        audio = speech.RecognitionAudio(content=audio_bytes)

        diarization = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=int(self.options.get("min_speakers", 2)),
            max_speaker_count=int(self.options.get("max_speakers", 6)),
        )
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=int(self.options.get("sample_rate", 16000)),
            language_code=self.language,
            enable_automatic_punctuation=True,
            diarization_config=diarization,
        )

        # long_running maneja audios > 1 min; espera hasta 10 min.
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=600)

        # El último `result` contiene las palabras con speaker_tag.
        if not response.results:
            return TranscriptionResult(self.name, self.language, [])
        words = response.results[-1].alternatives[0].words
        segments = _group_words_by_speaker(words)
        return TranscriptionResult(self.name, self.language, segments)


def _group_words_by_speaker(words) -> list[Segment]:
    """Agrupa palabras contiguas del mismo speaker_tag en un Segment."""
    segments: list[Segment] = []
    cur_tag = None
    cur_words: list[str] = []
    cur_start = 0.0
    cur_end = 0.0

    def _secs(dur) -> float:
        # google.protobuf.Duration -> segundos
        return dur.total_seconds() if hasattr(dur, "total_seconds") else float(dur)

    for w in words:
        tag = getattr(w, "speaker_tag", 0)
        start = _secs(w.start_time)
        end = _secs(w.end_time)
        if tag != cur_tag and cur_words:
            segments.append(Segment(
                start=cur_start, end=cur_end,
                speaker=f"Hablante {cur_tag}", text=" ".join(cur_words),
            ))
            cur_words = []
        if not cur_words:
            cur_start = start
            cur_tag = tag
        cur_words.append(w.word)
        cur_end = end

    if cur_words:
        segments.append(Segment(
            start=cur_start, end=cur_end,
            speaker=f"Hablante {cur_tag}", text=" ".join(cur_words),
        ))
    return segments
