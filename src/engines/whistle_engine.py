"""Motor Whistle/Cactus + pyannote para ASR por ventanas cortas.

Whistle admite como máximo 30 segundos por llamada. Este motor usa ventanas
solapadas de 28 s, reconstruye timestamps absolutos y después reutiliza la
misma diarización de pyannote que el motor local.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from ..types import Segment, TranscriptionResult
from .local_engine import LocalEngine


class WhistleEngine(LocalEngine):
    """Whistle para ASR y pyannote para atribución de hablantes."""

    name = "whistle"

    def is_available(self) -> tuple[bool, str]:
        try:
            import needle  # noqa: F401
        except ImportError:
            return False, "Falta cactus-needle: pip install -r requirements-whistle.txt"

        if not os.environ.get("HUGGINGFACE_TOKEN"):
            return False, (
                "HUGGINGFACE_TOKEN no definido (requerido por pyannote "
                "para la diarización)."
            )
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            return False, "Falta pyannote.audio: instala requirements-local.txt"
        return True, "Whistle + pyannote disponibles."

    def transcribe(self, audio_path, progress_cb=None) -> TranscriptionResult:
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(f"Motor whistle no disponible: {reason}")

        import needle
        import numpy as np
        import soundfile as sf
        import torch
        from pyannote.audio import Pipeline

        started = time.perf_counter()
        info = sf.info(str(audio_path))
        if info.samplerate != 16000:
            raise RuntimeError(
                f"Whistle requiere audio a 16 kHz; recibido: {info.samplerate} Hz"
            )

        window_sec = float(self.options.get("whistle_window_sec", 28.0))
        overlap_sec = float(self.options.get("whistle_overlap_sec", 4.0))
        if not 1.0 <= window_sec <= 30.0:
            raise ValueError("whistle_window_sec debe estar entre 1 y 30 segundos")
        if not 0.0 <= overlap_sec < window_sec:
            raise ValueError("whistle_overlap_sec debe ser >= 0 y menor que la ventana")
        step_sec = window_sec - overlap_sec

        total_sec = info.frames / info.samplerate
        total_frames = info.frames
        window_frames = max(1, int(round(window_sec * info.samplerate)))
        step_frames = max(1, int(round(step_sec * info.samplerate)))
        keywords = self.options.get("keywords")
        words: list[dict] = []
        window_count = 0
        start_frame = 0

        print(
            f"[*] Whistle: ventanas de {window_sec:.1f}s "
            f"(solapamiento {overlap_sec:.1f}s) ..."
        )
        while start_frame < total_frames:
            frames_to_read = min(window_frames, total_frames - start_frame)
            with sf.SoundFile(str(audio_path)) as audio:
                audio.seek(start_frame)
                wave = audio.read(frames=frames_to_read, dtype="float32")
            if wave.ndim > 1:
                wave = wave.mean(axis=1)
            wave = np.ascontiguousarray(wave, dtype=np.float32)

            result = needle.transcribe(
                wave,
                language=self.language.split("-")[0],
                keywords=keywords,
                word_timestamps=True,
            )
            window_start = start_frame / info.samplerate
            is_last = start_frame + step_frames >= total_frames
            ownership_boundary = window_start + step_sec

            for item in result.get("words", []):
                try:
                    local_start = float(item["start"])
                    local_end = float(item["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                text = str(item.get("word", "")).strip()
                if not text or local_end <= local_start:
                    continue
                absolute_start = window_start + max(0.0, local_start)
                absolute_end = min(
                    total_sec, window_start + max(local_start, local_end)
                )
                midpoint = (absolute_start + absolute_end) / 2.0
                # Cada palabra pertenece a una sola ventana. Las ventanas
                # siguientes aportan contexto, pero no duplican el solapamiento.
                if not is_last and midpoint >= ownership_boundary:
                    continue
                words.append({
                    "start": absolute_start,
                    "end": absolute_end,
                    "word": text,
                })

            window_count += 1
            processed = min(total_sec, window_start + frames_to_read / info.samplerate)
            if progress_cb is not None:
                try:
                    progress_cb(processed)
                except Exception:  # noqa: BLE001
                    pass
            start_frame += step_frames

        transcript_segments = self._words_to_segments(words)
        print(
            f"[*] Whistle: {window_count} ventanas, "
            f"{len(words)} palabras, {len(transcript_segments)} segmentos"
        )

        # La diarización opera sobre el audio completo por ventanas grandes,
        # igual que LocalEngine; solo el ASR está limitado a 30 s.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        diar_segments = []
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=os.environ.get("HUGGINGFACE_TOKEN"),
            )
            pipeline.to(torch.device(device))
            diar_segments = self._diarize_chunked(
                pipeline,
                audio_path,
                min_speakers=self.options.get("min_speakers", 1),
                max_speakers=self.options.get("max_speakers", 6),
                chunk_sec=float(self.options.get("chunk_sec", 600.0)),
                overlap_sec=float(self.options.get("overlap_sec", 30.0)),
            )
        except MemoryError:
            print(
                "[!] Diarización sin memoria suficiente; se conserva un único hablante."
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[!] Diarización falló ({type(exc).__name__}: {exc}); "
                "asignando SPEAKER_00."
            )

        final_segments = self._assign_speakers(transcript_segments, diar_segments)
        teacher_label, teacher_method = "", ""
        if self.options.get("identify_teacher", True):
            teacher_label, teacher_method = self._identify_teacher(final_segments)

        return TranscriptionResult(
            self.name,
            self.language,
            final_segments,
            model="whistle",
            processing_seconds=time.perf_counter() - started,
            teacher=teacher_label,
            teacher_method=teacher_method,
        )

    @staticmethod
    def _words_to_segments(words: list[dict]) -> list[dict]:
        """Agrupa palabras absolutas en segmentos manejables para subtítulos."""
        if not words:
            return []
        ordered = sorted(words, key=lambda item: (item["start"], item["end"]))
        segments: list[dict] = []
        current: list[dict] = []

        def flush() -> None:
            if not current:
                return
            segments.append({
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(item["word"] for item in current).strip(),
            })
            current.clear()

        for item in ordered:
            if current:
                gap = item["start"] - current[-1]["end"]
                duration = item["end"] - current[0]["start"]
                if gap > 1.0 or duration >= 6.0:
                    flush()
            current.append(item)
        flush()
        return segments
