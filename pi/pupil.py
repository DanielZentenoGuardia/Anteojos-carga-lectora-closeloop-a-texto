"""Detección de pupila oscura en imágenes IR (OpenCV).

Con iluminación IR fuera del eje óptico la pupila es la región más oscura del ojo. Se umbraliza
por percentil, se limpia con morfología (elimina reflejos corneales) y se ajusta una elipse
al contorno más circular. La confianza combina circularidad y área plausible.
Basado en el enfoque de Kassner, Patera & Bulling (2014) doi:10.1145/2638728.2641695
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Pupila:
    x: float          # px de la imagen completa
    y: float
    diametro_px: float
    conf: float       # 0..1 (0 = no detectada: parpadeo o pérdida)


class DetectorPupila:
    def __init__(self, area_min=150, area_max=12000, percentil=4.0, margen_roi=90):
        self.area_min, self.area_max = area_min, area_max
        self.percentil = percentil
        self.margen = margen_roi
        self._ultima: Pupila | None = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def _roi(self, img):
        """Región de búsqueda alrededor de la última pupila (acelera a 250+ fps)."""
        h, w = img.shape[:2]
        if self._ultima is None or self._ultima.conf < 0.5:
            return 0, 0, w, h
        m = self.margen
        x0 = int(max(0, self._ultima.x - m)); y0 = int(max(0, self._ultima.y - m))
        x1 = int(min(w, self._ultima.x + m)); y1 = int(min(h, self._ultima.y + m))
        return x0, y0, x1, y1

    def detectar(self, gris: np.ndarray) -> Pupila:
        x0, y0, x1, y1 = self._roi(gris)
        roi = cv2.GaussianBlur(gris[y0:y1, x0:x1], (5, 5), 0)
        umbral = np.percentile(roi, self.percentil) + 8
        _, bin_ = cv2.threshold(roi, umbral, 255, cv2.THRESH_BINARY_INV)
        bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, self._kernel)
        bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, self._kernel)
        contornos, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mejor, mejor_conf = None, 0.0
        for c in contornos:
            area = cv2.contourArea(c)
            if not (self.area_min <= area <= self.area_max) or len(c) < 5:
                continue
            perim = cv2.arcLength(c, True)
            circ = 4 * np.pi * area / (perim * perim + 1e-9)
            if circ > mejor_conf:
                mejor, mejor_conf = c, circ
        if mejor is None or mejor_conf < 0.55:
            self._ultima = Pupila(np.nan, np.nan, np.nan, 0.0)
            return self._ultima
        (cx, cy), (a, b), _ = cv2.fitEllipse(mejor)
        self._ultima = Pupila(cx + x0, cy + y0, float(max(a, b)), float(min(1.0, mejor_conf)))
        return self._ultima
