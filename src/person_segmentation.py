"""Segmentación opcional del profesor para reconstrucción de pizarra.

Usa Ultralytics YOLO-seg solo cuando el usuario activa --person-segmentation.
La dependencia se carga de forma diferida para que la ruta clásica siga
funcionando sin YOLO instalado.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class PersonSegmenter:
    """Devuelve una máscara booleana de personas para un frame BGR."""

    def __init__(
        self,
        weights: str | Path = "yolo11n-seg.pt",
        confidence: float = 0.25,
        device: str | int | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "La segmentación neural requiere ultralytics. "
                "Instala la dependencia con: pip install ultralytics"
            ) from exc

        self.weights = str(weights)
        self.confidence = confidence
        self.model = YOLO(self.weights)
        if device is None:
            try:
                import torch
                device = 0 if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Segmenta personas; devuelve uint8 {0,255} del tamaño del frame."""
        result = self.model.predict(
            frame_bgr,
            classes=[0],  # COCO: person
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        if result.masks is None or result.boxes is None:
            return mask

        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for raw_mask, cls in zip(result.masks.data, classes):
            if cls != 0:
                continue
            small = (raw_mask.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            resized = cv2.resize(
                small, (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = np.maximum(mask, resized)
        # Margen pequeño para cubrir bordes, sombras y píxeles periféricos del
        # cuerpo; evita que el contorno del profesor se convierta en tinta.
        if np.any(mask):
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        return mask
