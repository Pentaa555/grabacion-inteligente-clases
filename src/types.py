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
    is_teacher: bool = False   # True si este hablante fue identificado como el docente

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TranscriptionResult:
    """Resultado de un motor de transcripción + diarización."""

    engine: str
    language: str
    segments: List[Segment] = field(default_factory=list)
    model: str = ""       # p. ej. "medium" / "large-v3" (vacío si no aplica)
    processing_seconds: float = 0.0   # tiempo real que tardó el pipeline (s)
    denoise: bool = False             # si se aplicó reducción de ruido
    teacher: str = ""     # etiqueta del hablante identificado como docente ("" si no aplica)
    teacher_method: str = ""  # cómo se identificó ("mayor_tiempo_de_habla", "muestra_de_voz", ...)

    @property
    def speakers(self) -> List[str]:
        seen: List[str] = []
        for s in self.segments:
            if s.speaker not in seen:
                seen.append(s.speaker)
        return seen

    def speaker_stats(self) -> List[dict]:
        """Desglose por hablante: tiempo total de habla, nº de intervenciones y %.

        Sirve de evidencia para justificar la identificación del docente
        (el docente es quien acumula más tiempo de habla en una clase magistral).
        """
        agg: dict[str, dict] = {}
        for s in self.segments:
            a = agg.setdefault(s.speaker, {
                "speaker": s.speaker,
                "is_teacher": s.is_teacher,
                "seconds": 0.0,
                "turns": 0,
            })
            a["seconds"] += s.duration
            a["turns"] += 1
            a["is_teacher"] = a["is_teacher"] or s.is_teacher
        total = sum(a["seconds"] for a in agg.values()) or 1.0
        stats = []
        for a in agg.values():
            stats.append({
                "speaker": a["speaker"],
                "is_teacher": a["is_teacher"],
                "seconds": round(a["seconds"], 2),
                "turns": a["turns"],
                "share": round(a["seconds"] / total, 4),  # fracción del tiempo total
            })
        # Ordenar por tiempo de habla descendente (el docente suele quedar primero).
        stats.sort(key=lambda x: -x["seconds"])
        return stats

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "model": self.model,
            "language": self.language,
            "processing_seconds": round(self.processing_seconds, 2),
            "denoise": self.denoise,
            "speakers": self.speakers,
            "teacher": self.teacher,
            "teacher_method": self.teacher_method,
            "speaker_stats": self.speaker_stats(),
            "segments": [asdict(s) for s in self.segments],
        }


# ---------------------------------------------------------------------------
# Extracción del tablero (feature "tablero -> cuaderno digital")
#
# Método inspirado en FCN-LectureNet (Davila et al., IEEE Access 2021):
# muestrear frames -> reconstruir el tablero sin el profesor -> binarizar el
# contenido manuscrito -> detectar los momentos de "tablero más completo"
# (key-frames) para resumir visualmente la clase. Esta ruta es la CLÁSICA
# (OpenCV: mediana temporal + umbral adaptativo), equivalente ligero del
# método neuronal del paper.
# ---------------------------------------------------------------------------


@dataclass
class BoardKeyframe:
    """Un instante de la clase en que el tablero alcanzó un máximo de contenido.

    Representa una "página" del cuaderno digital: justo antes de un borrado o
    al final de un bloque de escritura, el tablero contiene lo máximo escrito.
    """

    index: int            # orden del key-frame (0, 1, 2, ...)
    time: float           # segundos desde el inicio del video
    frame_number: int     # nº de frame muestreado del que proviene
    ink_pixels: int       # cantidad de tinta (píxeles de contenido) detectada
    raw_image: str = ""   # ruta al frame original (con profesor y fondo)
    clean_image: str = "" # ruta a la reconstrucción del tablero sin el profesor
    binary_image: str = ""# ruta a la imagen binaria (solo tinta: blanco sobre negro)
    person_mask_image: str = ""  # máscara de personas usada para reconstruir (opcional)

    @property
    def minute(self) -> str:
        """El tiempo como MM:SS, para mostrar al usuario."""
        m, s = divmod(int(self.time), 60)
        return f"{m:02d}:{s:02d}"


@dataclass
class BoardResult:
    """Resultado de la extracción del tablero: el "cuaderno digital" de la clase."""

    method: str                               # "classic_opencv" | "fcn_lecturenet" | ...
    source_video: str = ""                    # ruta del video procesado
    fps_sampled: float = 1.0                  # frames por segundo muestreados
    frames_sampled: int = 0                   # nº total de frames analizados
    video_seconds: float = 0.0                # duración del video
    processing_seconds: float = 0.0           # tiempo real de cómputo
    device: str = "cpu"                        # "cpu" | "cuda"
    person_segmentation: bool = False          # si se excluyeron personas al reconstruir
    person_model: str = ""                     # p. ej. yolo11n-seg.pt
    person_frames_detected: int = 0            # frames con al menos una persona
    person_mean_coverage: float = 0.0          # fracción media del frame cubierta
    stabilized: bool = False                   # si se alinearon frames antes de reconstruir
    frames_aligned: int = 0                    # frames alineados con ECC
    roi: tuple[float, float, float, float] | None = None
    roi_source: str = "full_frame_fallback"    # automatic | manual | full_frame_fallback
    memory_mode: str = "windowed"              # no conserva stacks de toda la clase
    keyframes: List["BoardKeyframe"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "source_video": self.source_video,
            "fps_sampled": self.fps_sampled,
            "frames_sampled": self.frames_sampled,
            "video_seconds": round(self.video_seconds, 2),
            "processing_seconds": round(self.processing_seconds, 2),
            "device": self.device,
            "person_segmentation": self.person_segmentation,
            "person_model": self.person_model,
            "person_frames_detected": self.person_frames_detected,
            "person_mean_coverage": round(self.person_mean_coverage, 4),
            "stabilized": self.stabilized,
            "frames_aligned": self.frames_aligned,
            "roi": self.roi,
            "roi_source": self.roi_source,
            "memory_mode": self.memory_mode,
            "num_keyframes": len(self.keyframes),
            "keyframes": [asdict(k) for k in self.keyframes],
        }
