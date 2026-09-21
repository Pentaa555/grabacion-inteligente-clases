"""Generación de subtítulos: SRT, WebVTT y JSON, con etiqueta de hablante."""
from __future__ import annotations

import json
from pathlib import Path

from .types import TranscriptionResult


def _fmt_ts(seconds: float, sep: str) -> str:
    """Formatea segundos a HH:MM:SS<sep>mmm. sep = ',' (SRT) o '.' (VTT)."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}{sep}{ms:03d}"


def write_srt(result: TranscriptionResult, path: str | Path) -> Path:
    path = Path(path)
    lines = []
    for i, seg in enumerate(result.segments, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_ts(seg.start, ',')} --> {_fmt_ts(seg.end, ',')}")
        lines.append(f"[{seg.speaker}] {seg.text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_vtt(result: TranscriptionResult, path: str | Path) -> Path:
    path = Path(path)
    lines = ["WEBVTT", ""]
    for seg in result.segments:
        lines.append(f"{_fmt_ts(seg.start, '.')} --> {_fmt_ts(seg.end, '.')}")
        lines.append(f"<v {seg.speaker}>{seg.text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_json(result: TranscriptionResult, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def write_all(result: TranscriptionResult, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "srt": write_srt(result, out_dir / "subtitulos.srt"),
        "vtt": write_vtt(result, out_dir / "subtitulos.vtt"),
        "json": write_json(result, out_dir / "transcripcion.json"),
    }
