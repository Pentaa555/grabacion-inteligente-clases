"""CLI del prototipo: record | process | run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import capture
from . import denoise as denoise_mod
from .engines import get_engine, available_engines
from . import subtitles


def _do_record(args) -> tuple[Path, Path | None]:
    """Graba y devuelve (media_a_procesar, video_sincronizado_o_None).

    Por defecto usa captura SEPARADA y sincronizada (P1). Con --legacy-capture
    vuelve al modo de un solo FFmpeg.
    """
    if getattr(args, "legacy_capture", False):
        out = capture.record(args.out, duration=args.duration,
                             video_device=args.video_device,
                             audio_device=args.audio_device)
        print(f"[+] Grabación (legacy) guardada en {out}")
        return out, None

    prefix = Path(args.out).with_suffix("")
    cap = capture.record_synced(
        prefix, duration=args.duration,
        video_device=args.video_device, audio_device=args.audio_device,
    )
    print(f"[+] Video : {cap.video_path} ({cap.video_duration:.2f}s)")
    print(f"[+] Audio : {cap.audio_path} ({cap.audio_duration:.2f}s)")
    print(f"[+] Desfase de arranque audio->video: {cap.audio_offset:+.3f}s")
    if cap.muxed_path:
        print(f"[+] Sincronizado: {cap.muxed_path}")
    # Verificación de integridad: el audio ya no debe truncarse.
    if cap.audio_duration and cap.video_duration:
        drift = abs(cap.audio_duration - cap.video_duration)
        if drift > 1.0:
            print(f"[!] Aviso: video y audio difieren {drift:.2f}s en duración.",
                  file=sys.stderr)
    # Procesamos el AUDIO (reloj maestro para los subtítulos).
    return cap.audio_path, cap.muxed_path


def _process(
    video_or_audio: Path,
    engine_name: str,
    lang: str,
    out_root: Path,
    model: str = "base",
    denoise: bool = False,
    denoise_method: str = "ffmpeg",
) -> int:
    # Extraer audio si es video; si ya es audio, usarlo tal cual.
    suffix = video_or_audio.suffix.lower()
    if suffix in {".flac", ".wav"}:
        audio = video_or_audio
    else:
        print(f"[*] Extrayendo audio de {video_or_audio.name} ...")
        audio = capture.extract_audio(video_or_audio)
        print(f"[+] Audio: {audio}")

    # P2: reducción de ruido opcional, antes del ASR.
    if denoise:
        print(f"[*] Reduciendo ruido (method={denoise_method}) ...")
        audio = denoise_mod.denoise(audio, method=denoise_method)
        print(f"[+] Audio limpio: {audio}")

    engine = get_engine(engine_name, language=lang, model=model)
    ok, reason = engine.is_available()
    if not ok:
        print(f"[!] Motor '{engine_name}' no disponible: {reason}", file=sys.stderr)
        if engine_name != "mock":
            print("    Sugerencia: prueba --engine mock para validar el pipeline.",
                  file=sys.stderr)
        return 2

    print(f"[*] Transcribiendo + diarizando con motor '{engine_name}' ({lang}) ...")
    result = engine.transcribe(audio)

    out_dir = out_root / video_or_audio.stem
    paths = subtitles.write_all(result, out_dir)
    print(f"[+] Hablantes detectados: {', '.join(result.speakers) or '(ninguno)'}")
    print(f"[+] Segmentos: {len(result.segments)}")
    for kind, p in paths.items():
        print(f"[+] {kind.upper():4} -> {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prototipo",
        description="Grabación inteligente de clases (híbrido: gcp | local | mock).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="Grabar webcam + micrófono.")
    p_rec.add_argument("--duration", type=int, default=30)
    p_rec.add_argument("--out", default="grabacion.mkv")
    p_rec.add_argument("--video-device", default="/dev/video0")
    p_rec.add_argument("--audio-device", default="default")
    p_rec.add_argument("--legacy-capture", action="store_true",
                       help="Usar captura conjunta en un solo FFmpeg (modo antiguo).")

    p_proc = sub.add_parser("process", help="Procesar una grabación existente.")
    p_proc.add_argument("input")
    p_proc.add_argument("--engine", default="mock", choices=available_engines())
    p_proc.add_argument("--lang", default="es-CO")
    p_proc.add_argument("--out-root", default="salida")
    p_proc.add_argument("--model", default="base",
                        help="Modelo de Whisper: base, small, medium, large-v3")
    p_proc.add_argument("--denoise", action="store_true",
                        help="Reducir ruido antes del ASR (P2).")
    p_proc.add_argument("--denoise-method", default="ffmpeg",
                        choices=["ffmpeg", "deepfilternet"],
                        help="Motor de denoise: ffmpeg (ligero) o deepfilternet.")

    p_run = sub.add_parser("run", help="Grabar y luego procesar.")
    p_run.add_argument("--duration", type=int, default=30)
    p_run.add_argument("--out", default="grabacion.mkv")
    p_run.add_argument("--engine", default="mock", choices=available_engines())
    p_run.add_argument("--lang", default="es-CO")
    p_run.add_argument("--out-root", default="salida")
    p_run.add_argument("--video-device", default="/dev/video0")
    p_run.add_argument("--audio-device", default="default")
    p_run.add_argument("--model", default="base",
                       help="Modelo de Whisper: base, small, medium, large-v3")
    p_run.add_argument("--legacy-capture", action="store_true",
                       help="Usar captura conjunta en un solo FFmpeg (modo antiguo).")
    p_run.add_argument("--denoise", action="store_true",
                       help="Reducir ruido antes del ASR (P2).")
    p_run.add_argument("--denoise-method", default="ffmpeg",
                       choices=["ffmpeg", "deepfilternet"],
                       help="Motor de denoise: ffmpeg (ligero) o deepfilternet.")

    args = parser.parse_args(argv)

    if args.cmd == "record":
        _do_record(args)
        return 0

    if args.cmd == "process":
        return _process(
            Path(args.input), args.engine, args.lang, Path(args.out_root),
            getattr(args, "model", "base"),
            denoise=getattr(args, "denoise", False),
            denoise_method=getattr(args, "denoise_method", "ffmpeg"),
        )

    if args.cmd == "run":
        media, _ = _do_record(args)
        return _process(
            media, args.engine, args.lang, Path(args.out_root),
            getattr(args, "model", "base"),
            denoise=getattr(args, "denoise", False),
            denoise_method=getattr(args, "denoise_method", "ffmpeg"),
        )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
