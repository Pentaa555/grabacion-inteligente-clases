"""Extracción del tablero: convertir lo escrito en el tablero en un "cuaderno digital".

Feature inspirada en FCN-LectureNet (Davila et al., *FCN-LectureNet: Extractive
Summarization of Whiteboard and Chalkboard Lecture Videos*, IEEE Access 2021).

El método del paper usa una red neuronal para separar el contenido manuscrito del
profesor y del fondo. Aquí implementamos la ruta CLÁSICA equivalente con OpenCV,
que corre sin entrenar nada y es un primer entregable rápido:

    1. Muestreo        -> extraer N frames/seg del video.
    2. Fondo limpio    -> mediana temporal: el profesor se mueve, el tablero no,
                          así que la mediana de una ventana de frames "borra" al
                          profesor y deja el tablero.
    3. Binarización    -> umbral adaptativo sobre el tablero limpio -> solo tinta.
    4. Curva de tinta  -> medir cuánta tinta hay en cada instante (contenido vs
                          tiempo) y detectar máximos locales y borrados.
    5. Key-frames      -> los instantes de "tablero más lleno" = páginas del cuaderno.
    6. Exportar        -> PNG (crudo, limpio, binario) + JSON de evidencia.

La ruta neuronal (fiel al paper, con la RTX 3070) queda como mejora futura
seleccionable con otro `method`.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .types import BoardKeyframe, BoardResult


# --------------------------------------------------------------------------- #
# Parámetros por defecto (ajustables desde la CLI)
# --------------------------------------------------------------------------- #
DEFAULT_SAMPLE_FPS = 1.0        # frames muestreados por segundo de video
DEFAULT_MEDIAN_WINDOW = 7       # frames de la ventana para la mediana temporal
DEFAULT_MIN_GAP_SECONDS = 20.0  # separación mínima entre key-frames (evita duplicados)
DEFAULT_MAX_WIDTH = 1280        # redimensiona frames anchos para acelerar
AUTO_ROI_SAMPLES = 12            # frames ligeros usados para localizar la pizarra
BACKGROUND_SAMPLES = 32           # límite de frames para fondos globales


def _evenly_spaced_indices(count: int, limit: int) -> list[int]:
    """Índices uniformes sin materializar una copia de todo el video."""
    if count <= 0:
        return []
    if count <= limit:
        return list(range(count))
    return np.linspace(0, count - 1, limit, dtype=int).tolist()


def _board_candidates(frame: np.ndarray) -> list[tuple[float, tuple[int, int, int, int]]]:
    """Encuentra regiones grandes, claras y relativamente rectangulares.

    Es una heurística deliberadamente conservadora: no intenta reconocer texto,
    solo localizar la superficie persistente de una pizarra. Si no encuentra una
    región confiable, el pipeline conserva el encuadre completo como fallback.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    height, width = gray.shape
    percentile = float(np.percentile(gray, 55))
    # Una pizarra blanca suele ser más clara y menos saturada que pared/ladrillo.
    neutral_bright = (gray >= max(45.0, percentile)) & (hsv[:, :, 1] <= 145)
    mask = (neutral_bright.astype(np.uint8) * 255)
    close_size = max(9, int(round(min(width, height) / 28)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    frame_area = float(width * height)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        ratio = area / frame_area
        aspect = w / max(1.0, float(h))
        rectangularity = cv2.contourArea(contour) / max(1.0, area)
        if ratio < 0.10 or ratio > 0.88 or w < width * 0.30 or h < height * 0.14:
            continue
        if aspect < 1.15 or aspect > 7.0 or rectangularity < 0.30:
            continue
        local = gray[y:y + h, x:x + w]
        brightness = float(np.mean(local)) / 255.0 if local.size else 0.0
        score = 0.55 * min(ratio / 0.45, 1.0) + 0.35 * rectangularity + 0.10 * brightness
        candidates.append((score, (x, y, x + w, y + h)))
    return sorted(candidates, key=lambda item: item[0], reverse=True)[:4]


def _box_iou(a: tuple[float, float, float, float],
             b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, by1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(1e-6, area_a + area_b - inter)


def detect_board_roi(video_path: str | Path,
                     max_samples: int = AUTO_ROI_SAMPLES) -> tuple[float, float, float, float] | None:
    """Detecta automáticamente la superficie de la pizarra en coordenadas [0,1].

    Analiza pocos frames espaciados y agrupa candidatos persistentes. No carga
    el video completo. Devuelve ``None`` cuando la evidencia no es suficiente,
    en cuyo caso el llamador usa el frame completo en vez de inventar un ROI.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _evenly_spaced_indices(total, max_samples)
    observations: list[tuple[float, tuple[float, float, float, float]]] = []
    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = min(1.0, 960.0 / max(1, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
        for score, (x0, y0, x1, y1) in _board_candidates(frame):
            fh, fw = frame.shape[:2]
            observations.append((score, (x0 / fw, y0 / fh, x1 / fw, y1 / fh)))
    cap.release()
    if not observations:
        return None

    clusters: list[list[tuple[float, tuple[float, float, float, float]]]] = []
    for observation in sorted(observations, key=lambda item: item[0], reverse=True):
        for cluster in clusters:
            if _box_iou(observation[1], cluster[0][1]) >= 0.35:
                cluster.append(observation)
                break
        else:
            clusters.append([observation])
    clusters.sort(key=lambda c: (len(c), sum(x[0] for x in c) / len(c)), reverse=True)
    best = clusters[0]
    # Un único candidato puede ser una pared o una puerta; exigimos persistencia.
    if len(best) < 2 and len(observations) > 1:
        return None
    weights = np.array([max(0.05, item[0]) for item in best], dtype=np.float32)
    boxes = np.array([item[1] for item in best], dtype=np.float32)
    roi = (boxes * weights[:, None]).sum(axis=0) / weights.sum()
    # Margen pequeño para no cortar tinta cercana al borde de la pizarra.
    x0, y0, x1, y1 = roi.tolist()
    pad_x, pad_y = (x1 - x0) * 0.025, (y1 - y0) * 0.035
    return (max(0.0, x0 - pad_x), max(0.0, y0 - pad_y),
            min(1.0, x1 + pad_x), min(1.0, y1 + pad_y))


# --------------------------------------------------------------------------- #
# Etapa 1 — Muestreo de frames
# --------------------------------------------------------------------------- #
def _sample_frames(
    video_path: Path,
    sample_fps: float,
    max_width: int,
    progress_cb: Callable[[float], None] | None = None,
    roi: tuple[float, float, float, float] | None = None,
    person_segmenter: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[int], list[float], float, list[np.ndarray] | None]:
    """Devuelve (frames_gris, nums_frame, tiempos_seg, duracion_video).

    Muestrea el video a `sample_fps`. Trabajamos en escala de grises: para
    separar tinta de tablero el color no aporta y gasta memoria.

    `roi` = (x0, y0, x1, y1) en FRACCIONES [0..1] del ancho/alto. Recorta cada
    frame a la región del tablero antes de analizarlo. Es clave cuando el
    tablero ocupa solo una parte del encuadre (resto: pared, puerta, profesor):
    sin recorte, esas texturas contaminan la binarización.

    Si `person_segmenter` está activo, se ejecuta sobre el frame completo (para
    detectar al profesor aunque esté fuera del ROI) y devuelve una máscara
    alineada con cada frame recortado.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_seconds = (total_frames / native_fps) if total_frames else 0.0

    # Cada cuántos frames nativos tomamos uno.
    step = max(1, int(round(native_fps / max(0.1, sample_fps))))

    frames: list[np.ndarray] = []
    person_masks: list[np.ndarray] | None = [] if person_segmenter else None
    nums: list[int] = []
    times: list[float] = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            original_h, original_w = frame.shape[:2]
            person_mask = person_segmenter(frame) if person_segmenter else None

            # Recorte a la región del tablero (si se pidió), también sobre la máscara.
            if roi is not None:
                h, w = frame.shape[:2]
                x0 = int(max(0.0, min(1.0, roi[0])) * w)
                y0 = int(max(0.0, min(1.0, roi[1])) * h)
                x1 = int(max(0.0, min(1.0, roi[2])) * w)
                y1 = int(max(0.0, min(1.0, roi[3])) * h)
                if x1 > x0 and y1 > y0:
                    frame = frame[y0:y1, x0:x1]
                    if person_mask is not None:
                        person_mask = person_mask[y0:y1, x0:x1]
            # Redimensionar si es muy ancho (acelera sin perder legibilidad).
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                new_size = (max_width, int(h * scale))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
                if person_mask is not None:
                    person_mask = cv2.resize(
                        person_mask, new_size, interpolation=cv2.INTER_NEAREST,
                    )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            if person_masks is not None:
                person_masks.append(person_mask if person_mask is not None
                                    else np.zeros(gray.shape, dtype=np.uint8))
            nums.append(idx)
            times.append(idx / native_fps)
            if progress_cb and total_frames:
                progress_cb(idx / total_frames)
        idx += 1

    cap.release()
    return frames, nums, times, video_seconds, person_masks


# --------------------------------------------------------------------------- #
# Etapa 2 — Estabilización/alineación temporal
# --------------------------------------------------------------------------- #
def _stabilize_frames(
    frames: list[np.ndarray],
    person_masks: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray] | None, int]:
    """Alinea frames al frame central usando ECC (movimiento de traslación).

    La cámara puede desplazarse unos píxeles durante la clase. Sin alinear,
    la mediana temporal mezcla bordes, reflejos y escritura en posiciones
    distintas, creando fantasmas. Se usa solo tras el ROI para que ECC se
    concentre en la pizarra. La misma transformación se aplica a la máscara.
    Si ECC no converge para un frame, se conserva sin transformar.
    """
    n = len(frames)
    if n < 2:
        return frames, person_masks, 0
    reference = frames[n // 2].astype(np.float32) / 255.0
    reference = cv2.GaussianBlur(reference, (5, 5), 0)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        40,
        1e-4,
    )
    # Se modifica cada lista en el lugar: conservar otra lista alineada duplicaría
    # cientos de MB en videos largos justo cuando la memoria está más ajustada.
    aligned = frames
    aligned_masks = person_masks
    success_count = 0
    h, w = frames[0].shape[:2]

    for i, frame in enumerate(list(frames)):
        if i == n // 2:
            warp = np.eye(2, 3, dtype=np.float32)
            success = True
        else:
            moving = cv2.GaussianBlur(frame.astype(np.float32) / 255.0, (5, 5), 0)
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                _, warp = cv2.findTransformECC(
                    reference, moving, warp, cv2.MOTION_TRANSLATION,
                    criteria, None, 5,
                )
                success = True
            except cv2.error:
                success = False
        if success:
            aligned_frame = cv2.warpAffine(
                frame, warp, (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            if i != n // 2:
                success_count += 1
        else:
            aligned_frame = frame
        aligned[i] = aligned_frame

        if aligned_masks is not None:
            mask = aligned_masks[i]
            if success:
                mask = cv2.warpAffine(
                    mask, warp, (w, h),
                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255,
                )
            aligned_masks[i] = mask

    return aligned, aligned_masks, success_count


# --------------------------------------------------------------------------- #
# Etapa 3 — Reconstrucción del tablero sin el profesor (mediana temporal)
# --------------------------------------------------------------------------- #
def _visible_background(
    frames: list[np.ndarray],
    person_masks: list[np.ndarray] | None = None,
    max_samples: int = BACKGROUND_SAMPLES,
) -> np.ndarray:
    """Calcula un fondo usando como máximo `max_samples` frames.

    Una mediana exacta de miles de frames exige un arreglo enorme. Para el fondo
    solo necesitamos una referencia estable, por eso usamos frames espaciados:
    el uso de memoria queda acotado y no depende de la duración de la clase.
    """
    if not frames:
        return np.zeros((1, 1), np.uint8)
    indices = _evenly_spaced_indices(len(frames), max_samples)
    stack = np.stack([frames[i] for i in indices], axis=0)
    if person_masks is None:
        return np.median(stack, axis=0).astype(np.uint8)
    mask_stack = np.stack([person_masks[i] for i in indices], axis=0).astype(bool)
    values = stack.astype(np.float32)
    values[mask_stack] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(values, axis=0)
    return result.astype(np.float32)


def _clean_frame(
    frames: list[np.ndarray],
    index: int,
    window: int,
    person_masks: list[np.ndarray] | None = None,
    global_visible: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruye un solo frame, manteniendo en memoria solo su ventana."""
    half = max(1, window // 2)
    lo = max(0, index - half)
    hi = min(len(frames), index + half + 1)
    source = np.stack(frames[lo:hi], axis=0)
    if person_masks is None:
        return np.median(source, axis=0).astype(np.uint8)

    valid = ~np.stack(person_masks[lo:hi], axis=0).astype(bool)
    values = source.astype(np.float32)
    values[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_float = np.nanmedian(values, axis=0)
    fallback = np.median(source, axis=0).astype(np.float32)
    all_hidden = ~valid.any(axis=0)
    if global_visible is not None:
        replacement = np.where(np.isnan(global_visible), fallback, global_visible)
        median_float[all_hidden] = replacement[all_hidden]
        current_hidden = person_masks[index] > 0
        median_float[current_hidden] = replacement[current_hidden]
    else:
        median_float[all_hidden] = fallback[all_hidden]
    return np.nan_to_num(median_float, nan=0.0).astype(np.uint8)


def _clean_background(
    frames: list[np.ndarray],
    window: int,
    person_masks: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Reconstruye todos los frames sin crear una matriz con toda la clase."""
    if not frames:
        return []
    global_visible = (_visible_background(frames, person_masks)
                      if person_masks is not None else None)
    return [
        _clean_frame(frames, i, window, person_masks, global_visible)
        for i in range(len(frames))
    ]


def _global_background(frames: list[np.ndarray]) -> np.ndarray:
    """Referencia de tablero vacío con un número acotado de muestras."""
    return _visible_background(frames, None)


# --------------------------------------------------------------------------- #
# Etapa 3 — Binarización del contenido manuscrito
# --------------------------------------------------------------------------- #
def _binarize(
    clean: np.ndarray,
    background: np.ndarray | None = None,
    dark_ink: bool | None = None,
) -> np.ndarray:
    """Devuelve una imagen binaria (tinta = blanco 255, fondo = negro 0).

    `dark_ink=True` fuerza marcador oscuro sobre pizarra clara. Es necesario en
    grabaciones subexpuestas donde el whiteboard tiene media <128 y la heurística
    de brillo lo confundiría con un chalkboard.
    """
    blur = cv2.GaussianBlur(clean, (3, 3), 0)

    if background is not None:
        bg_blur = cv2.GaussianBlur(background, (3, 3), 0)
        # Diferencia absoluta: dónde el tablero cambió respecto al vacío.
        diff = cv2.absdiff(blur, bg_blur)
        # Umbral de Otsu sobre la diferencia: separa tinta de fondo automáticamente.
        _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mean_brightness = float(blur.mean())
        if dark_ink is True:
            thresh_type = cv2.THRESH_BINARY_INV
        elif dark_ink is False:
            thresh_type = cv2.THRESH_BINARY
        else:
            thresh_type = (cv2.THRESH_BINARY_INV if mean_brightness >= 128
                           else cv2.THRESH_BINARY)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type,
            blockSize=25, C=10,
        )

    # Limpieza morfológica: quita motas de 1-2 px (ruido), conserva trazos.
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary


def _ink_amount(binary: np.ndarray) -> int:
    """Cantidad de tinta: número de píxeles de contenido (blancos)."""
    return int(cv2.countNonZero(binary))


# --------------------------------------------------------------------------- #
# Etapas 4 y 5 — Curva de tinta y selección de key-frames
# --------------------------------------------------------------------------- #
def _select_keyframes(
    ink_curve: list[int],
    times: list[float],
    min_gap_seconds: float,
) -> list[int]:
    """Elige índices de key-frames: los máximos locales de contenido.

    Un key-frame marca un "tablero lleno" justo antes de un borrado o al final
    de un bloque de escritura. Buscamos picos de la curva de tinta separados por
    al menos `min_gap_seconds` para no repetir casi el mismo instante.
    """
    n = len(ink_curve)
    if n == 0:
        return []
    if n == 1:
        return [0]

    peak_max = max(ink_curve) or 1
    # Un "bloque de escritura" es un tramo con tinta por encima de un mínimo.
    # Separamos bloques por VALLES: caídas de tinta por debajo de una fracción
    # del máximo global (un borrado deja el tablero casi vacío). Dentro de cada
    # bloque, el key-frame es el instante de MÁXIMA tinta (tablero más lleno).
    valley_level = 0.40 * peak_max   # por debajo de esto, el tablero está "vacío"
    min_block_level = 0.10 * peak_max  # ignora bloques con casi nada escrito

    peaks: list[int] = []
    block_start = None
    for i in range(n):
        above = ink_curve[i] >= valley_level
        if above and block_start is None:
            block_start = i
        elif not above and block_start is not None:
            # Cerramos el bloque [block_start, i): su pico es el de más tinta.
            seg = range(block_start, i)
            best = max(seg, key=lambda j: ink_curve[j])
            if ink_curve[best] >= min_block_level:
                peaks.append(best)
            block_start = None
    if block_start is not None:
        seg = range(block_start, n)
        best = max(seg, key=lambda j: ink_curve[j])
        if ink_curve[best] >= min_block_level:
            peaks.append(best)

    if not peaks:  # curva plana sin valles claros: toma el máximo global
        peaks = [max(range(n), key=lambda j: ink_curve[j])]

    # Filtrar por separación temporal mínima, quedándonos con el de más tinta.
    peaks = sorted(set(peaks))
    filtered: list[int] = []
    for p in peaks:
        if not filtered:
            filtered.append(p)
            continue
        if times[p] - times[filtered[-1]] >= min_gap_seconds:
            filtered.append(p)
        else:
            # Demasiado cerca: conserva el de mayor contenido.
            if ink_curve[p] > ink_curve[filtered[-1]]:
                filtered[-1] = p
    return filtered


# --------------------------------------------------------------------------- #
# Orquestación completa
# --------------------------------------------------------------------------- #
def extract_board(
    video_path: str | Path,
    out_dir: str | Path,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    median_window: int = DEFAULT_MEDIAN_WINDOW,
    min_gap_seconds: float = DEFAULT_MIN_GAP_SECONDS,
    max_width: int = DEFAULT_MAX_WIDTH,
    roi: tuple[float, float, float, float] | None = None,
    person_segmenter: Callable[[np.ndarray], np.ndarray] | None = None,
    person_model: str = "",
    stabilize: bool = False,
    save_debug: bool = False,
    progress_cb: Callable[[float], None] | None = None,
) -> BoardResult:
    """Pipeline completo con ROI automático y memoria acotada.

    Cuando `roi` es ``None`` se localiza automáticamente una superficie de
    pizarra persistente. El ROI manual sigue disponible como override para
    grabaciones atípicas, pero ya no es necesario configurarlo en la interfaz.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if roi is None:
        effective_roi = detect_board_roi(video_path)
        roi_source = "automatic" if effective_roi is not None else "full_frame_fallback"
    else:
        effective_roi = roi
        roi_source = "manual"

    frames, nums, times, video_seconds, person_masks = _sample_frames(
        video_path, sample_fps, max_width, progress_cb, roi=effective_roi,
        person_segmenter=person_segmenter,
    )
    if not frames:
        raise RuntimeError("No se extrajo ningún frame del video.")

    frames_aligned = 0
    if stabilize:
        frames, person_masks, frames_aligned = _stabilize_frames(
            frames, person_masks,
        )

    # Solo se conserva una referencia global pequeña. Las ventanas locales se
    # crean y liberan una por una; no se apilan los miles de frames del video.
    global_visible = (_visible_background(frames, person_masks)
                      if person_masks is not None else None)
    background = None if person_masks is not None else _global_background(frames)

    # 1. Binarización/curva: conservar únicamente el número de píxeles, no las
    # imágenes intermedias completas.
    ink_curve: list[int] = []
    for i in range(len(frames)):
        clean = _clean_frame(frames, i, median_window, person_masks, global_visible)
        binary = _binarize(clean, background, dark_ink=(person_masks is not None))
        if person_masks is not None:
            binary[person_masks[i] > 0] = 0
        ink_curve.append(_ink_amount(binary))

    # Los extremos tienen una ventana incompleta y no son fiables para elegir
    # páginas; se excluyen solo de la curva, no del material exportable.
    edge = max(1, median_window // 2)
    if len(ink_curve) > 2 * edge:
        for j in range(edge):
            ink_curve[j] = -1
            ink_curve[-(j + 1)] = -1

    kf_indices = _select_keyframes(ink_curve, times, min_gap_seconds)

    # Recalcular únicamente los key-frames para guardar las imágenes finales.
    keyframes: list[BoardKeyframe] = []
    for order, i in enumerate(kf_indices):
        clean = _clean_frame(frames, i, median_window, person_masks, global_visible)
        binary = _binarize(clean, background, dark_ink=(person_masks is not None))
        if person_masks is not None:
            binary[person_masks[i] > 0] = 0
        clean_path = out_dir / f"keyframe_{order:02d}_clean.png"
        bin_path = out_dir / f"keyframe_{order:02d}_binary.png"
        cv2.imwrite(str(clean_path), clean)
        cv2.imwrite(str(bin_path), binary)

        raw_path = ""
        mask_path = ""
        if save_debug:
            rp = out_dir / f"keyframe_{order:02d}_raw.png"
            cv2.imwrite(str(rp), frames[i])
            raw_path = str(rp)
        if person_masks is not None:
            mp = out_dir / f"keyframe_{order:02d}_mask.png"
            cv2.imwrite(str(mp), person_masks[i])
            mask_path = str(mp)

        keyframes.append(BoardKeyframe(
            index=order,
            time=round(times[i], 2),
            frame_number=nums[i],
            ink_pixels=ink_curve[i],
            raw_image=raw_path,
            clean_image=str(clean_path),
            binary_image=str(bin_path),
            person_mask_image=mask_path,
        ))

    person_detected = 0
    person_coverage = 0.0
    if person_masks is not None:
        coverages = [float(np.mean(mask > 0)) for mask in person_masks]
        person_detected = sum(c > 0.001 for c in coverages)
        person_coverage = float(np.mean(coverages)) if coverages else 0.0

    result = BoardResult(
        method=("classic_opencv+person_segmentation" if person_masks is not None
                else "classic_opencv"),
        source_video=str(video_path),
        fps_sampled=sample_fps,
        frames_sampled=len(frames),
        video_seconds=video_seconds,
        processing_seconds=time.time() - t0,
        device=str(getattr(person_segmenter, "device", "cpu")),
        person_segmentation=person_masks is not None,
        person_model=person_model,
        person_frames_detected=person_detected,
        person_mean_coverage=person_coverage,
        stabilized=stabilize,
        frames_aligned=frames_aligned,
        roi=effective_roi,
        roi_source=roi_source,
        keyframes=keyframes,
    )

    import json
    (out_dir / "cuaderno.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
