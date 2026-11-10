"""Asignación de cada fijación a una palabra en pantalla, con corrección de deriva vertical.

Con Arial 12 / interlineado 1,5 al 150 % de zoom, la separación entre líneas es ~1,1°,
cercana a la precisión de un eye-tracker low cost. Por eso se estima la deriva vertical
como la mediana del desvío entre las últimas fijaciones y el centro de la línea asignada.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class CajaPalabra:
    wid: str           # "<id_oracion>|<variante>|<indice>"
    sid: str
    texto: str
    orden: int         # posición global en el orden de lectura
    pagina: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


class MapaPalabras:
    def __init__(self, ventana_deriva: int = 15):
        self.cajas: dict[str, CajaPalabra] = {}
        self._lineas: list[tuple[float, list[CajaPalabra]]] = []
        self._desvios = deque(maxlen=ventana_deriva)
        self.deriva_y = 0.0
        self.paso_linea = 30.0

    def actualizar(self, cajas: list[CajaPalabra], reemplazar: bool = False):
        if reemplazar:
            self.cajas.clear()
        for c in cajas:
            self.cajas[c.wid] = c
        self._reconstruir()

    def eliminar_oracion(self, sid: str):
        self.cajas = {k: v for k, v in self.cajas.items() if v.sid != sid}
        self._reconstruir()

    def _reconstruir(self):
        filas: dict[int, list[CajaPalabra]] = {}
        for c in self.cajas.values():
            filas.setdefault(int(round(c.yc)), []).append(c)
        self._lineas = sorted((float(y), sorted(cs, key=lambda c: c.x0)) for y, cs in filas.items())
        if len(self._lineas) > 1:
            dif = np.diff([y for y, _ in self._lineas])
            dif = dif[dif > 1]
            if len(dif):
                self.paso_linea = float(np.median(dif))

    def asignar(self, x: float, y: float) -> CajaPalabra | None:
        if not self._lineas:
            return None
        yc = y - self.deriva_y
        centros = np.array([l[0] for l in self._lineas])
        i = int(np.argmin(np.abs(centros - yc)))
        if abs(centros[i] - yc) > 0.75 * self.paso_linea:
            return None
        linea = self._lineas[i][1]
        mejor, dmin = None, float("inf")
        for c in linea:
            d = 0.0 if c.x0 <= x <= c.x1 else min(abs(x - c.x0), abs(x - c.x1))
            if d < dmin:
                mejor, dmin = c, d
        ancho = (mejor.x1 - mejor.x0) if mejor else 0
        if mejor is None or dmin > max(ancho, 20.0):
            return None
        self._desvios.append(yc - centros[i])
        if len(self._desvios) >= 5:
            self.deriva_y += 0.5 * float(np.median(self._desvios))
            self._desvios.clear()
        return mejor
