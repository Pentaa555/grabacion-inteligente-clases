"""P2 -- Reducción de ruido antes del ASR, mediante filtros de FFmpeg.

Elegimos una cadena de filtros de FFmpeg (afftdn + highpass/lowpass + loudnorm)
en lugar de DeepFilterNet porque:
  * NO añade dependencias pesadas (torch/onnx extra) -- crítico en este equipo,
    donde la RAM ha llegado a ~0.5-1 GB libre y DeepFilterNet pesa varios cientos
    de MB además de PyTorch.
  * Corre en CPU en tiempo casi real y no compite por memoria con Whisper/pyannote.
  * Mejora la relación señal/ruido y normaliza el volumen, que es lo que degradaba
    la transcripción (audio de webcam con mucho silencio/ruido de fondo).

DeepFilterNet queda como mejora futura (mejor supresión de ruido estacionario),
seleccionable con `method="deepfilternet"` cuando el hardware lo permita.

Cadena `ffmpeg`:
  highpass=f=80        -> quita retumbe/DC y ruido de baja frecuencia (ventiladores)
  lowpass=f=8000       -> recorta siseo agudo fuera de la banda de voz
  afftdn=nf=-25        -> denoise espectral (FFT), reducción de ruido estacionario
  loudnorm             -> normaliza el volumen (EBU R128) para nivelar la voz
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _require_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("FFmpeg no está instalado o no está en el PATH.")
    return ff


# Cadena de filtros por defecto (banda de voz + denoise espectral + normalización).
_FFMPEG_VOICE_CHAIN = (
    "highpass=f=80,"
    "lowpass=f=8000,"
    "afftdn=nf=-25,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)


def denoise(
    audio_in: str | Path,
    audio_out: str | Path | None = None,
    method: str = "ffmpeg",
    sample_rate: int = 16000,
    filter_chain: str | None = None,
) -> Path:
    """Aplica reducción de ruido a `audio_in` y devuelve la ruta del audio limpio.

    Parameters
    ----------
    audio_in : ruta del audio de entrada (wav/flac/...).
    audio_out : destino; por defecto `<nombre>_denoised.flac` junto al de entrada.
    method : "ffmpeg" (por defecto, ligero) o "deepfilternet" (si está instalado).
    sample_rate : Hz de salida (16000 recomendado para ASR).
    filter_chain : cadena de filtros FFmpeg alternativa; si None usa la de voz.
    """
    audio_in = Path(audio_in)
    if not audio_in.exists():
        raise FileNotFoundError(audio_in)
    if audio_out is None:
        audio_out = audio_in.with_name(audio_in.stem + "_denoised.flac")
    audio_out = Path(audio_out)
    audio_out.parent.mkdir(parents=True, exist_ok=True)

    if method == "deepfilternet":
        return _denoise_deepfilternet(audio_in, audio_out, sample_rate)

    ffmpeg = _require_ffmpeg()
    chain = filter_chain or _FFMPEG_VOICE_CHAIN
    cmd = [
        ffmpeg, "-y", "-i", str(audio_in),
        "-af", chain,
        "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "flac",
        str(audio_out),
    ]
    subprocess.run(cmd, check=True)
    return audio_out


def _denoise_deepfilternet(audio_in: Path, audio_out: Path, sample_rate: int) -> Path:
    """Reducción de ruido con DeepFilterNet (mejora futura; requiere el paquete).

    Se aísla en su propia función para no imponer la dependencia pesada salvo que
    el usuario la elija explícitamente con method="deepfilternet".
    """
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio  # type: ignore
    except ImportError as e:  # pragma: no cover - dependencia opcional
        raise RuntimeError(
            "DeepFilterNet no está instalado. Instálalo con "
            "`pip install deepfilternet` (pesa varios cientos de MB + torch), "
            "o usa method='ffmpeg' (por defecto, ligero)."
        ) from e

    model, df_state, _ = init_df()
    audio, _ = load_audio(str(audio_in), sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    save_audio(str(audio_out), enhanced, df_state.sr())
    return audio_out
