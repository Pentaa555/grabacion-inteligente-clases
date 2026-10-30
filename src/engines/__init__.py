"""Motores de transcripción + diarización (backend intercambiable)."""
from __future__ import annotations

from .base import TranscriptionEngine
from .mock_engine import MockEngine
from .gcp_engine import GCPEngine
from .local_engine import LocalEngine
from .whistle_engine import WhistleEngine

_ENGINES = {
    "mock": MockEngine,
    "gcp": GCPEngine,
    "local": LocalEngine,
    "whistle": WhistleEngine,
}


def get_engine(name: str, **kwargs) -> TranscriptionEngine:
    """Devuelve una instancia del motor solicitado."""
    key = name.lower()
    if key not in _ENGINES:
        raise ValueError(
            f"Motor desconocido: {name!r}. Opciones: {', '.join(_ENGINES)}"
        )
    return _ENGINES[key](**kwargs)


def available_engines() -> list[str]:
    return list(_ENGINES)
