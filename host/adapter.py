"""Política de adaptación del texto a lo largo de las ~100 páginas.

Tramo A (línea base)   : se muestra siempre el original; se aprenden las medias del lector.
Tramo B (exploración)  : original o variante al azar -> efectos causales por rasgo.
Tramo C (adaptación)   : se muestra la variante con menor carga predicha, dejando una
                         fracción de oraciones originales como control.
Regla fija: solo se decide sobre oraciones al menos `paginas_adelanto` por delante de la
página que se está leyendo; nunca se cambia lo que el lector tiene a la vista.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .profile import PerfilLector
from .text_model import Oracion


@dataclass
class Decision:
    sid: str
    variante: str
    tramo: str
    motivo: str
    control: bool = False


class PoliticaAdaptacion:
    def __init__(self, cfg: dict, perfil: PerfilLector, semilla: int | None = None):
        f = cfg["fases"]
        self.pag_base = f["paginas_linea_base"]
        self.pag_explorar = f["explorar_hasta_pagina"]
        self.adelanto = f["paginas_adelanto"]
        self.tasa_explorar = f["tasa_exploracion"]
        self.tasa_control = f["tasa_control"]
        self.prob_min = f["prob_min_recomendar"]
        self.mejora_min = f["mejora_min"]
        self.perfil = perfil
        self.rng = np.random.default_rng(semilla)
        self.decisiones: dict[str, Decision] = {}

    def tramo(self, pagina: int) -> str:
        if pagina < self.pag_base:
            return "A"
        if pagina < self.pag_explorar:
            return "B"
        return "C"

    def decidir(self, oracion: Oracion, pagina_oracion: int, pagina_lectura: int) -> Decision | None:
        """Devuelve la decisión (una sola vez por oración) o None si aún no corresponde."""
        if oracion.id in self.decisiones:
            return self.decisiones[oracion.id]
        if pagina_oracion < pagina_lectura + self.adelanto:
            return None  # demasiado cerca: no se toca
        tramo = self.tramo(pagina_oracion)
        alternativas = [v for v in oracion.variantes if v != "orig"]
        if tramo == "A" or not alternativas:
            d = Decision(oracion.id, "orig", tramo, "linea_base" if tramo == "A" else "sin_variantes")
        elif tramo == "B":
            if self.rng.random() < self.tasa_explorar:
                d = Decision(oracion.id, str(self.rng.choice(alternativas)), tramo, "exploracion")
            else:
                d = Decision(oracion.id, "orig", tramo, "exploracion", control=True)
        else:
            d = self._explotar(oracion, tramo)
        self.decisiones[oracion.id] = d
        return d

    def _explotar(self, oracion: Oracion, tramo: str) -> Decision:
        if self.rng.random() < self.tasa_control:
            return Decision(oracion.id, "orig", tramo, "control", control=True)
        orig = oracion.original.rasgos
        pred_orig = self.perfil.predecir(orig)
        mejor, mejor_pred = "orig", pred_orig
        for vid, v in oracion.variantes.items():
            if vid == "orig":
                continue
            p = self.perfil.predecir(v.rasgos)
            if p < mejor_pred:
                mejor, mejor_pred = vid, p
        if mejor == "orig" or pred_orig - mejor_pred < self.mejora_min:
            return Decision(oracion.id, "orig", tramo, "sin_mejora_suficiente")
        # Exigir evidencia: algún rasgo que la variante reduce debe tener P(b>0) >= prob_min
        v = oracion.variantes[mejor].rasgos
        reducidos = [n for n in self.perfil.nombres if v.get(n, 0) < orig.get(n, 0)]
        if not any(self.perfil.prob_positiva(n) >= self.prob_min for n in reducidos):
            return Decision(oracion.id, "orig", tramo, "evidencia_insuficiente")
        return Decision(oracion.id, mejor, tramo, f"carga_predicha {pred_orig:.2f}->{mejor_pred:.2f}")
