"""Captura de webcam + micrófono y extracción de audio, mediante FFmpeg.

P1 -- Captura SEPARADA y sincronizada:
    `record_synced()` graba video (v4l2) y audio (ALSA) en DOS procesos FFmpeg
    concurrentes que arrancan casi al mismo instante. Se mide el desfase real de
    arranque entre ambos y, al muxear, se compensa con `-itsoffset` para anclar
    los subtítulos al reloj del AUDIO (referencia temporal maestra). Esto evita
    el truncado de audio que ocurría al capturar todo en un solo FFmpeg
    (desincronización v4l2+ALSA que cortaba el audio a ~20s).

    `record()` (captura conjunta en un solo FFmpeg) se conserva como modo legacy.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


def _require_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("FFmpeg no está instalado o no está en el PATH.")
    return ff


# --------------------------------------------------------------------------- #
# Modo legacy: un solo FFmpeg (puede truncar el audio en clips largos).
# --------------------------------------------------------------------------- #
def record(
    out_path: str | Path,
    duration: int = 30,
    video_device: str = "/dev/video0",
    audio_device: str = "default",
    resolution: str = "1280x720",
    fps: int = 30,
) -> Path:
    """Graba `duration` segundos de video (v4l2) + audio (ALSA) a `out_path`.

    Modo LEGACY (un solo proceso FFmpeg). Para grabaciones fiables prefiere
    `record_synced`, que evita el truncado de audio observado en este entorno.
    """
    ffmpeg = _require_ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg, "-y",
        "-f", "v4l2", "-framerate", str(fps), "-video_size", resolution,
        "-i", video_device,
        "-f", "alsa", "-i", audio_device,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


# --------------------------------------------------------------------------- #
# P1: captura separada y sincronizada.
# --------------------------------------------------------------------------- #
@dataclass
class SyncedCapture:
    """Resultado de una captura separada."""

    video_path: Path          # video sin audio (contexto visual: tablero, etc.)
    audio_path: Path          # audio crudo WAV (fuente para ASR/diarización)
    muxed_path: Path | None   # video+audio remuxeados y sincronizados (o None)
    audio_offset: float       # segundos que el audio arrancó DESPUÉS del video
    video_duration: float
    audio_duration: float


def record_synced(
    out_prefix: str | Path,
    duration: int = 30,
    video_device: str = "/dev/video0",
    audio_device: str = "default",
    resolution: str = "1280x720",
    fps: int = 30,
    sample_rate: int = 16000,
    mux: bool = True,
) -> SyncedCapture:
    """Graba video y audio en procesos FFmpeg SEPARADOS y los sincroniza.

    Estrategia:
      1. Lanzar el proceso de VIDEO (v4l2 -> archivo, sin audio) y anotar su t0.
      2. Lanzar el proceso de AUDIO (ALSA -> WAV mono 16 kHz) y anotar su t0.
      3. Esperar a que ambos terminen (cada uno se autolimita con `-t duration`).
      4. `audio_offset = t0_audio - t0_video` es cuánto arrancó el audio DESPUÉS
         del video. Al muxear se aplica `-itsoffset audio_offset` a la pista de
         audio para realinearla con el video.

    El audio WAV mono 16 kHz es directamente la entrada del ASR; el `.flac` para
    el motor se obtiene con `extract_audio(audio_path)` o convirtiendo el WAV.
    """
    ffmpeg = _require_ffmpeg()
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    video_path = out_prefix.with_name(out_prefix.name + "_video.mkv")
    audio_path = out_prefix.with_name(out_prefix.name + "_audio.wav")

    video_cmd = [
        ffmpeg, "-y",
        "-f", "v4l2", "-framerate", str(fps), "-video_size", resolution,
        "-i", video_device,
        "-t", str(duration),
        "-an",                                   # sin audio en el proceso de video
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        str(video_path),
    ]
    audio_cmd = [
        ffmpeg, "-y",
        "-f", "alsa", "-i", audio_device,
        "-t", str(duration),
        "-vn",                                   # sin video en el proceso de audio
        "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",                     # WAV PCM, sin pérdida
        str(audio_path),
    ]

    # Arranque casi simultáneo; medimos el desfase real de inicio.
    t0_video = time.monotonic()
    p_video = subprocess.Popen(video_cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    t0_audio = time.monotonic()
    p_audio = subprocess.Popen(audio_cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)

    audio_offset = round(t0_audio - t0_video, 3)  # >0: audio arrancó después

    rc_video = p_video.wait()
    rc_audio = p_audio.wait()
    if rc_video != 0:
        raise RuntimeError(f"FFmpeg (video) terminó con código {rc_video}")
    if rc_audio != 0:
        raise RuntimeError(f"FFmpeg (audio) terminó con código {rc_audio}")

    v_dur = probe_duration(video_path)
    a_dur = probe_duration(audio_path)

    muxed_path: Path | None = None
    if mux:
        muxed_path = out_prefix.with_name(out_prefix.name + "_synced.mkv")
        mux_media(video_path, audio_path, muxed_path, audio_offset=audio_offset)

    return SyncedCapture(
        video_path=video_path,
        audio_path=audio_path,
        muxed_path=muxed_path,
        audio_offset=audio_offset,
        video_duration=v_dur,
        audio_duration=a_dur,
    )


def mux_media(
    video_path: str | Path,
    audio_path: str | Path,
    out_path: str | Path,
    audio_offset: float = 0.0,
) -> Path:
    """Combina un video (sin audio) y un audio en un MKV, corrigiendo el desfase.

    `audio_offset` (segundos) es cuánto arrancó el audio DESPUÉS del video; se
    aplica con `-itsoffset` a la entrada de audio para realinearla. Copia ambos
    streams sin recodificar (rápido y sin pérdida).
    """
    ffmpeg = _require_ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg, "-y", "-i", str(video_path)]
    if audio_offset:
        cmd += ["-itsoffset", str(audio_offset)]
    cmd += [
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def extract_audio(
    video_path: str | Path,
    audio_out: str | Path | None = None,
    sample_rate: int = 16000,
) -> Path:
    """Extrae el audio a FLAC mono `sample_rate` Hz (óptimo para ASR/diarización)."""
    ffmpeg = _require_ffmpeg()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if audio_out is None:
        audio_out = video_path.with_suffix(".flac")
    audio_out = Path(audio_out)
    audio_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "flac",
        str(audio_out),
    ]
    subprocess.run(cmd, check=True)
    return audio_out


def probe_duration(media_path: str | Path) -> float:
    """Duración en segundos vía ffprobe; 0.0 si no se puede determinar."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)],
            check=True, capture_output=True, text=True,
        )
        return float(out.stdout.strip() or 0.0)
    except (subprocess.CalledProcessError, ValueError):
        return 0.0
