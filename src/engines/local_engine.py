"""Motor local: faster-whisper + pyannote.audio (stack offline del PDF).

Requiere:
  - pip install faster-whisper pyannote.audio transformers huggingface_hub
  - Token de Hugging Face para pyannote: HUGGINGFACE_TOKEN
  - GPU NVIDIA recomendada (>= 8 GB VRAM); en CPU es lento pero funcional.

Flujo:
  1. faster-whisper transcribe (palabras + timestamps).
  2. pyannote.audio diariza (segmentos de hablante).
  3. Se asigna cada segmento de transcripción al hablante más cercano en el tiempo.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..types import Segment, TranscriptionResult
from .base import TranscriptionEngine


class LocalEngine(TranscriptionEngine):
    name = "local"

    def is_available(self) -> tuple[bool, str]:
        try:
            import faster_whisper
            import pyannote.audio  # noqa: F401
        except ImportError:
            return False, "Faltan dependencias: pip install faster-whisper pyannote.audio"
        if not os.environ.get("HUGGINGFACE_TOKEN"):
            return False, ("HUGGINGFACE_TOKEN no definido (requerido por pyannote "
                           "para descargar el modelo de diarización).")
        return True, "Motor local disponible."

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(f"Motor local no disponible: {reason}")

        import faster_whisper
        from pyannote.audio import Pipeline
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_size = self.options.get("model", "base")  # base/small/medium/large-v3
        lang = self.language.split("-")[0]  # "es-CO" -> "es"

        print(f"[*] faster-whisper: cargando modelo '{model_size}' ({device}) ...")
        model = faster_whisper.WhisperModel(
            model_size, device=device, compute_type="int8" if device == "cpu" else "float16"
        )

        print(f"[*] Transcribiendo audio ({lang}) ...")
        # vad_filter descarta tramos que cree silencio; en audio real con pausas
        # recorta voz de más (se observó que perdía todo tras la 1ª pausa), así que
        # por defecto va DESACTIVADO. Se puede reactivar con options["vad_filter"]=True.
        use_vad = bool(self.options.get("vad_filter", False))
        segments, info = model.transcribe(
            str(audio_path), language=lang,
            word_timestamps=True, vad_filter=use_vad,
        )

        # Convertir generador a lista.
        transcript_segments = []
        for seg in segments:
            transcript_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": [
                    {"start": w.start, "end": w.end, "word": w.word}
                    for w in (seg.words or [])
                ],
            })

        print(f"[*] Diarización con pyannote.audio (device={device}) ...")
        diar_segments = []
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=os.environ.get("HUGGINGFACE_TOKEN"),
            )
            if device == "cpu":
                # En CPU el pipeline es más lento; ajustar si es necesario.
                pipeline.to(torch.device("cpu"))

            # Cargar audio con soundfile (evita torchcodec)
            import soundfile as sf
            waveform, sr = sf.read(str(audio_path), dtype="float32")
            # pyannote espera (channels, time) -> (1, samples) para mono
            if waveform.ndim == 1:
                waveform = waveform[None, :]
            audio_dict = {"waveform": torch.from_numpy(waveform), "sample_rate": sr}

            diar = pipeline(audio_dict,
                            min_speakers=self.options.get("min_speakers", 1),
                            max_speakers=self.options.get("max_speakers", 6))

            # Obtener segmentos de diarización como lista.
            diar_annotation = diar.speaker_diarization  # Annotation
            for turn, _, speaker in diar_annotation.itertracks(yield_label=True):
                diar_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                })
        except Exception as e:  # noqa: BLE001
            # pyannote puede fallar de forma intermitente (p. ej. IndexError con
            # arrays vacíos en clips cortos/degenerados). No perdemos la
            # transcripción: degradamos a un único hablante.
            print(f"[!] Diarización falló ({type(e).__name__}: {e}); "
                  f"asignando 'SPEAKER_00' a todos los segmentos.")
            diar_segments = []

        # Asignar cada segmento de transcripción al hablante más cercano.
        # Si no hubo diarización, todos quedan como SPEAKER_00.
        final_segments = self._assign_speakers(transcript_segments, diar_segments)

        return TranscriptionResult(self.name, self.language, final_segments)

    def _assign_speakers(self, transcript: list, diarization: list) -> list[Segment]:
        """Asigna el hablante más cercano en el tiempo a cada segmento transcrito."""
        result = []
        for t in transcript:
            t_start, t_end = t["start"], t["end"]
            # Buscar el segmento de diarización que más se solapa.
            # Si no hubo diarización, todos quedan como SPEAKER_00.
            best_speaker = "SPEAKER_00"
            best_overlap = 0.0
            for d in diarization:
                d_start, d_end = d["start"], d["end"]
                overlap = max(0.0, min(t_end, d_end) - max(t_start, d_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = d["speaker"]
            result.append(Segment(
                start=t_start, end=t_end,
                speaker=best_speaker, text=t["text"],
            ))
        return result
