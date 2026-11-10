"""Detección de fijaciones, sacadas y parpadeos en línea (algoritmo I-VT).

Referencia: Salvucci & Goldberg (2000), ETRA. doi:10.1145/355017.355028
La velocidad se calcula sobre la posición suavizada (media móvil de `ventana` muestras): a
250 Hz, un ruido de 0,1° por muestra equivale a >30 °/s y fragmentaría las fijaciones.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Muestra:
    t: float            # segundos (reloj del Raspberry Pi)
    x: float            # px CSS, coordenadas del documento
    y: float
    valida: bool
    pupila_mm: float = float("nan")


@dataclass
class Fijacion:
    t_ini: float
    t_fin: float
    x: float
    y: float
    n: int = 0

    @property
    def duracion_ms(self) -> float:
        return (self.t_fin - self.t_ini) * 1000.0


@dataclass
class Parpadeo:
    t_ini: float
    t_fin: float

    @property
    def duracion_ms(self) -> float:
        return (self.t_fin - self.t_ini) * 1000.0


@dataclass
class DetectorIVT:
    px_por_grado: float
    umbral_grados_s: float = 30.0
    fijacion_min_ms: float = 60.0
    parpadeo_min_ms: float = 50.0
    parpadeo_max_ms: float = 500.0
    ventana: int = 3
    _prev: tuple | None = None          # (t, x_suave, y_suave)
    _buf: deque = field(default_factory=lambda: deque(maxlen=3))
    _xs: list = field(default_factory=list)
    _ys: list = field(default_factory=list)
    _t_ini: float | None = None
    _t_ult: float | None = None
    _inv_ini: float | None = None
    _inv_ult: float | None = None

    def _cerrar_fijacion(self, eventos: list):
        if self._t_ini is not None and self._xs:
            f = Fijacion(self._t_ini, self._t_ult, float(np.median(self._xs)),
                         float(np.median(self._ys)), len(self._xs))
            if f.duracion_ms >= self.fijacion_min_ms:
                eventos.append(f)
        self._xs, self._ys, self._t_ini, self._t_ult = [], [], None, None

    def agregar(self, m: Muestra) -> list:
        """Procesa una muestra y devuelve los eventos que se cerraron con ella."""
        eventos: list = []
        if not m.valida:
            self._cerrar_fijacion(eventos)
            if self._inv_ini is None:
                self._inv_ini = m.t
            self._inv_ult = m.t
            self._prev = None
            self._buf.clear()
            return eventos

        if self._inv_ini is not None:
            dur = (m.t - self._inv_ini) * 1000.0
            if self.parpadeo_min_ms <= dur <= self.parpadeo_max_ms:
                eventos.append(Parpadeo(self._inv_ini, m.t))
            self._inv_ini = self._inv_ult = None

        if self._buf.maxlen != self.ventana:
            self._buf = deque(self._buf, maxlen=self.ventana)
        self._buf.append((m.x, m.y))
        xs, ys = np.mean(self._buf, axis=0)
        actual = (m.t, xs, ys)
        if self._prev is not None:
            dt = m.t - self._prev[0]
            if dt > 0:
                dist = np.hypot(xs - self._prev[1], ys - self._prev[2]) / self.px_por_grado
                if dist / dt > self.umbral_grados_s:
                    self._cerrar_fijacion(eventos)
                    self._prev = actual
                    return eventos
        if self._t_ini is None:
            self._t_ini = m.t
        self._t_ult = m.t
        self._xs.append(m.x)
        self._ys.append(m.y)
        self._prev = actual
        return eventos

    def vaciar(self) -> list:
        eventos: list = []
        self._cerrar_fijacion(eventos)
        return eventos
