"""Interfaz común de los motores de ASR + diarización."""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Callable

from ..types import TranscriptionResult


class TranscriptionEngine(abc.ABC):
    """Contrato que todo backend (gcp, local, mock) debe cumplir."""

    name: str = "base"

    def __init__(self, language: str = "es-CO", **kwargs) -> None:
        self.language = language
        self.options = kwargs

    @abc.abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """(disponible, motivo). Comprueba credenciales/dependencias."""

    @abc.abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        progress_cb: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe y diariza el audio, devolviendo segmentos por hablante.

        `progress_cb`, si se pasa, se invoca con los SEGUNDOS de audio ya
        procesados (creciente). El llamador lo convierte a porcentaje contra la
        duración total. Es opcional: un motor puede ignorarlo.
        """
