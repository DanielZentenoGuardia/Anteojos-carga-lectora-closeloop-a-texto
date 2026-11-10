"""Perfil de sensibilidad del lector: regresión lineal bayesiana en línea.

    L_s = b0 + sum_k b_k * x_sk + e,   b ~ N(media_poblacional, sd_prior^2),  e ~ N(0, ruido^2)

La media a priori representa el efecto poblacional; la posterior se aleja de ella en la
medida en que los datos de ESTE lector lo justifican (contracción bayesiana). Durante la
fase de exploración las variantes se asignan al azar, por lo que los coeficientes estiman
efectos causales de cada rasgo estructural sobre la carga de este lector.
"""
from __future__ import annotations

import json
import math

import numpy as np


class PerfilLector:
    def __init__(self, nombres: list[str], escala: dict, prior_media: dict,
                 prior_sd: float, ruido_sd: float):
        self.nombres = list(nombres)
        self.escala = dict(escala)
        self.ruido2 = ruido_sd ** 2
        k = len(self.nombres) + 1  # + intercepto
        self.mu0 = np.array([0.0] + [prior_media.get(n, 0.0) for n in self.nombres])
        self.precision = np.eye(k) / prior_sd ** 2
        self.precision[0, 0] = 1.0  # intercepto: prior N(0, 1)
        self.b = self.precision @ self.mu0
        self.n = 0

    @classmethod
    def desde_config(cls, cfg: dict) -> "PerfilLector":
        r = cfg["rasgos"]
        return cls(r["nombres"], r["escala"], r["prior_media"], r["prior_sd"], r["ruido_sd"])

    def vector(self, rasgos: dict) -> np.ndarray:
        return np.array([1.0] + [rasgos.get(n, 0.0) / self.escala.get(n, 1.0) for n in self.nombres])

    def actualizar(self, rasgos: dict, carga: float):
        if not np.isfinite(carga):
            return
        x = self.vector(rasgos)
        self.precision += np.outer(x, x) / self.ruido2
        self.b += x * carga / self.ruido2
        self.n += 1

    @property
    def media(self) -> np.ndarray:
        return np.linalg.solve(self.precision, self.b)

    @property
    def cov(self) -> np.ndarray:
        return np.linalg.inv(self.precision)

    def predecir(self, rasgos: dict, beta: np.ndarray | None = None) -> float:
        return float(self.vector(rasgos) @ (self.media if beta is None else beta))

    def muestrear(self, rng: np.random.Generator) -> np.ndarray:
        """Muestra de la posterior (para Thompson sampling)."""
        return rng.multivariate_normal(self.media, self.cov)

    def prob_positiva(self, nombre: str) -> float:
        i = self.nombres.index(nombre) + 1
        m, s = self.media[i], math.sqrt(self.cov[i, i])
        return 0.5 * (1 + math.erf(m / (s * math.sqrt(2))))

    def resumen(self) -> list[dict]:
        m, c = self.media, self.cov
        return [{"rasgo": n, "sensibilidad": round(float(m[i + 1]), 3),
                 "sd": round(math.sqrt(c[i + 1, i + 1]), 3),
                 "desvio_vs_poblacion": round(float(m[i + 1] - self.mu0[i + 1]), 3),
                 "prob_positiva": round(self.prob_positiva(n), 3)}
                for i, n in enumerate(self.nombres)]

    def guardar(self, ruta):
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump({"n_oraciones": self.n, "rasgos": self.resumen()}, f, ensure_ascii=False, indent=2)
