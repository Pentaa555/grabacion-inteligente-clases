"""Motor mock: genera segmentos sintéticos sin dependencias externas.

Sirve para validar el pipeline completo (captura -> subtítulos) sin credenciales
GCP ni GPU. Reparte la duración real del audio entre dos hablantes alternados.
"""
from __future__ import annotations

from pathlib import Path

from ..types import Segment, TranscriptionResult
from .base import TranscriptionEngine


_FRASES = [
    "Buenos días, hoy vamos a repasar el tema de la clase anterior.",
    "Profesor, ¿podría explicar de nuevo el segundo ejemplo?",
    "Claro, el segundo ejemplo aplica la misma fórmula pero con otros datos.",
    "Entonces el resultado cambia según el coeficiente inicial.",
    "Exacto. Recuerden entregar el taller antes del viernes.",
    "¿El taller es individual o en grupo?",
    "Es en parejas. Pueden coordinarlo entre ustedes.",
]


class MockEngine(TranscriptionEngine):
    name = "mock"

    def is_available(self) -> tuple[bool, str]:
        return True, "Motor mock siempre disponible."

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        # Duración real si es posible, si no un valor por defecto.
        from ..capture import probe_duration

        total = probe_duration(audio_path) or 21.0
        n = len(_FRASES)
        step = total / n
        segments = []
        for i, frase in enumerate(_FRASES):
            start = round(i * step, 2)
            end = round(min((i + 1) * step, total), 2)
            speaker = "Docente" if i % 2 == 0 else "Estudiante"
            segments.append(Segment(start=start, end=end, speaker=speaker, text=frase))
        return TranscriptionResult(
            engine=self.name, language=self.language, segments=segments
        )
