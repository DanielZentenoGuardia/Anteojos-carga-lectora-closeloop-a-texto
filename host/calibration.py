"""Calibración: posición de pupila (cámara ocular) -> coordenadas de la cámara de escena.

Cadena completa de un anteojo montado en la cabeza:
    pupila (px cámara ocular) --polinomio--> escena (px cámara de escena) --H--> pantalla (px CSS)

La homografía H se recalcula continuamente con los marcadores ArUco de las esquinas de la
pantalla, por lo que la calibración tolera movimientos de cabeza. Sin marcadores (por ejemplo,
con mentonera o en el simulador) se usa H = identidad y el polinomio mapea directo a pantalla.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .geometry import aplicar


def _terminos(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([np.ones_like(x), x, y, x * y, x ** 2, y ** 2])


class CalibracionOjo:
    """Polinomio de 2.º orden por ojo (6 coeficientes por eje)."""

    def __init__(self):
        self.coef: np.ndarray | None = None

    def ajustar(self, pupila: np.ndarray, objetivo: np.ndarray) -> float:
        A = _terminos(np.asarray(pupila, float))
        self.coef, *_ = np.linalg.lstsq(A, np.asarray(objetivo, float), rcond=None)
        resid = A @ self.coef - objetivo
        return float(np.sqrt((resid ** 2).sum(axis=1)).mean())

    def predecir(self, pupila) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("Ojo sin calibrar")
        return (_terminos(np.atleast_2d(np.asarray(pupila, float))) @ self.coef)[0]


class Calibrador:
    """Recolecta muestras mientras se muestran 9 puntos y ajusta cada ojo."""

    ASENTAMIENTO_S = 0.5   # se ignoran las muestras iniciales mientras el ojo llega al punto
    VENTANA_S = 1.0        # duración útil por punto

    def __init__(self):
        self.ojos = {"L": CalibracionOjo(), "R": CalibracionOjo()}
        self.activa = False
        self._objetivo = None   # (idx, x_escena, y_escena, t_inicio)
        self._muestras = defaultdict(lambda: defaultdict(list))  # ojo -> idx -> [(px, py)]
        self._objetivos = {}
        self.error_px = None

    @property
    def lista(self) -> bool:
        return all(o.coef is not None for o in self.ojos.values())

    def iniciar(self):
        self.activa = True
        self._muestras.clear()
        self._objetivos.clear()

    def objetivo(self, idx: int, x_pant: float, y_pant: float, t: float, H_inv: np.ndarray | None):
        """Registra un nuevo punto de calibración (coordenadas de pantalla)."""
        xy = np.array([x_pant, y_pant])
        if H_inv is not None:
            xy = aplicar(H_inv, xy)
        self._objetivo = (idx, xy, t)
        self._objetivos[idx] = xy

    def muestra(self, ojo: str, px: float, py: float, conf: float, t: float):
        if not self.activa or self._objetivo is None or conf < 0.5:
            return
        idx, _, t0 = self._objetivo
        if self.ASENTAMIENTO_S <= t - t0 <= self.ASENTAMIENTO_S + self.VENTANA_S:
            self._muestras[ojo][idx].append((px, py))

    def finalizar(self) -> dict:
        self.activa = False
        errores = {}
        for ojo, cal in self.ojos.items():
            pup, obj = [], []
            for idx, lista in self._muestras[ojo].items():
                if len(lista) >= 5:
                    pup.append(np.median(np.asarray(lista), axis=0))
                    obj.append(self._objetivos[idx])
            if len(pup) >= 6:
                errores[ojo] = cal.ajustar(np.asarray(pup), np.asarray(obj))
            else:
                errores[ojo] = None
        return errores

    def identidad(self):
        """Calibración trivial (pupila == pantalla). Usada por el simulador."""
        for cal in self.ojos.values():
            cal.coef = np.zeros((6, 2))
            cal.coef[1, 0] = 1.0
            cal.coef[2, 1] = 1.0
