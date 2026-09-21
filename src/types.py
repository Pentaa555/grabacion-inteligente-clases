"""Tipos de datos compartidos por el pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class Segment:
    """Un tramo de audio atribuido a un hablante."""

    start: float          # segundos
    end: float            # segundos
    speaker: str          # p. ej. "Hablante 1" / "Docente"
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TranscriptionResult:
    """Resultado de un motor de transcripción + diarización."""

    engine: str
    language: str
    segments: List[Segment] = field(default_factory=list)

    @property
    def speakers(self) -> List[str]:
        seen: List[str] = []
        for s in self.segments:
            if s.speaker not in seen:
                seen.append(s.speaker)
        return seen

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "language": self.language,
            "speakers": self.speakers,
            "segments": [asdict(s) for s in self.segments],
        }
