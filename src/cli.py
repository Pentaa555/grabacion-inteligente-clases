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


def _process_board(
    video: Path,
    out_root: Path,
    sample_fps: float = 1.0,
    median_window: int = 7,
    min_gap: float = 20.0,
    max_width: int = 1280,
    roi: tuple[float, float, float, float] | None = None,
    person_segmentation: bool = False,
    person_model: str = "yolo11n-seg.pt",
    person_conf: float = 0.25,
    stabilize: bool = False,
    debug: bool = False,
) -> int:
    """Extrae el contenido del tablero del VIDEO y genera el "cuaderno digital".

    Ruta clásica (OpenCV), inspirada en FCN-LectureNet (IEEE Access 2021):
    reconstruye el tablero sin el profesor, aísla la tinta y selecciona los
    instantes de "tablero más lleno" como páginas del cuaderno.
    """
    from . import board  # import diferido: solo carga OpenCV si se usa 'board'

    person_segmenter = None
    if person_segmentation:
        try:
            from .person_segmentation import PersonSegmenter
            print(f"[*] Cargando segmentador de personas: {person_model} ...")
            person_segmenter = PersonSegmenter(person_model, confidence=person_conf)
            print(f"[+] Segmentación activa en device={person_segmenter.device}")
        except (RuntimeError, OSError) as e:
            print(f"[!] No se pudo activar segmentación de personas: {e}",
                  file=sys.stderr)
            return 2

    suffix = video.suffix.lower()
    if suffix in {".flac", ".wav", ".mp3", ".m4a", ".aac", ".ogg"}:
        print(f"[!] '{video.name}' es AUDIO, no video. El módulo de tablero "
              f"necesita el video de la clase (donde se ve el tablero).",
              file=sys.stderr)
        return 2
    if not video.exists():
        print(f"[!] No existe el archivo: {video}", file=sys.stderr)
        return 2

    out_dir = out_root / f"{video.stem}_tablero"
    print(f"[*] Extrayendo tablero de {video.name} "
          f"(muestreo {sample_fps} fps, ventana mediana {median_window}) ...")

    def _progress(frac: float) -> None:
        print(f"\r[*] Analizando video: {frac*100:5.1f}%", end="", file=sys.stderr)

    try:
        result = board.extract_board(
            video, out_dir,
            sample_fps=sample_fps, median_window=median_window,
            min_gap_seconds=min_gap, max_width=max_width,
            roi=roi,
            person_segmenter=person_segmenter,
            person_model=person_model if person_segmenter else "",
            stabilize=stabilize,
            save_debug=debug, progress_cb=_progress,
        )
    except MemoryError:
        print("\n[!] Memoria insuficiente durante el tablero. "
              "La versión actual procesa por ventanas; reduce --sample-fps o "
              "--max-width si el video sigue siendo muy grande.", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"\n[!] Error procesando el video: {e}", file=sys.stderr)
        return 2
    print()  # cierra la línea de progreso

    print(f"[+] Frames muestreados: {result.frames_sampled} "
          f"({result.video_seconds:.1f}s de video)")
    print(f"[+] Tiempo de cómputo: {result.processing_seconds:.2f}s "
          f"(device={result.device})")
    if result.person_segmentation:
        print(f"[+] Frames con persona: {result.person_frames_detected}/"
              f"{result.frames_sampled} "
              f"(cobertura media={result.person_mean_coverage:.1%})")
    if result.stabilized:
        print(f"[+] Frames alineados con ECC: {result.frames_aligned}/"
              f"{max(0, result.frames_sampled - 1)}")
    print(f"[+] Páginas del cuaderno (key-frames): {len(result.keyframes)}")
    for k in result.keyframes:
        print(f"    #{k.index} @ {k.minute}  tinta={k.ink_pixels}px")
    if result.roi is not None:
        roi_text = ",".join(f"{value:.3f}" for value in result.roi)
        print(f"[+] ROI ({result.roi_source}): {roi_text}")
    else:
        print("[+] ROI: encuadre completo (no se detectó una pizarra confiable)")
    print(f"[+] Salida en: {out_dir}")
    print(f"[+] JSON de evidencia: {out_dir / 'cuaderno.json'}")
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

    p_board = sub.add_parser(
        "board",
        help="Extraer el contenido del tablero de un VIDEO (cuaderno digital).",
    )
    p_board.add_argument("input", help="Ruta del video de la clase (mp4/mkv/...).")
    p_board.add_argument("--out-root", default="salida")
    p_board.add_argument("--sample-fps", type=float, default=1.0,
                         help="Frames muestreados por segundo de video (def. 1.0).")
    p_board.add_argument("--median-window", type=int, default=7,
                         help="Ventana (frames) de la mediana temporal que quita "
                              "al profesor (def. 7).")
    p_board.add_argument("--min-gap", type=float, default=20.0,
                         help="Separación mínima en segundos entre páginas (def. 20).")
    p_board.add_argument("--max-width", type=int, default=1280,
                         help="Ancho máximo del frame; redimensiona para acelerar.")
    p_board.add_argument("--roi", default=None,
                         help="Override opcional x0,y0,x1,y1. Si se omite, "
                              "detecta automáticamente la superficie de la pizarra.")
    p_board.add_argument("--person-segmentation", action="store_true",
                         help="Detectar personas con YOLO-seg y excluirlas de la "
                              "reconstrucción temporal (requiere ultralytics).")
    p_board.add_argument("--person-model", default="yolo11n-seg.pt",
                         help="Pesos YOLO-seg para personas (def. yolo11n-seg.pt).")
    p_board.add_argument("--person-conf", type=float, default=0.25,
                         help="Confianza mínima del detector de personas (def. 0.25).")
    p_board.add_argument("--stabilize", action="store_true",
                         help="Alinear frames con ECC antes de reconstruir para "
                              "compensar movimiento de cámara.")
    p_board.add_argument("--debug", action="store_true",
                         help="Guarda también los frames originales (con profesor).")

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

    if args.cmd == "board":
        roi = None
        if args.roi:
            try:
                parts = [float(x) for x in args.roi.split(",")]
                if len(parts) != 4:
                    raise ValueError
                roi = (parts[0], parts[1], parts[2], parts[3])
            except ValueError:
                print(f"[!] --roi inválido: '{args.roi}'. Formato esperado: "
                      f"x0,y0,x1,y1 con fracciones 0..1.", file=sys.stderr)
                return 2
        return _process_board(
            Path(args.input), Path(args.out_root),
            sample_fps=args.sample_fps,
            median_window=args.median_window,
            min_gap=args.min_gap,
            max_width=args.max_width,
            roi=roi,
            person_segmentation=args.person_segmentation,
            person_model=args.person_model,
            person_conf=args.person_conf,
            stabilize=args.stabilize,
            debug=args.debug,
        )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
