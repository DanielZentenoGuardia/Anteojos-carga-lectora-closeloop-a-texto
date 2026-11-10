"""Homografía cámara de escena -> pantalla, implementada solo con NumPy (DLT)."""
from __future__ import annotations

import numpy as np


def homografia(origen: np.ndarray, destino: np.ndarray) -> np.ndarray:
    """Estima H (3x3) tal que destino ~ H @ origen, con >= 4 pares de puntos.

    Usa la transformación lineal directa (DLT) con normalización de Hartley.
    """
    origen = np.asarray(origen, float)
    destino = np.asarray(destino, float)
    if len(origen) < 4 or origen.shape != destino.shape:
        raise ValueError("Se necesitan al menos 4 pares de puntos con la misma forma")

    def normalizar(p):
        c = p.mean(axis=0)
        d = np.sqrt(((p - c) ** 2).sum(axis=1)).mean() or 1.0
        s = np.sqrt(2) / d
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        ph = np.column_stack([p, np.ones(len(p))]) @ T.T
        return ph, T

    a, Ta = normalizar(origen)
    b, Tb = normalizar(destino)
    filas = []
    for (x, y, _), (u, v, _) in zip(a, b):
        filas.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        filas.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(filas))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ Hn @ Ta
    return H / H[2, 2]


def aplicar(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Aplica la homografía a puntos (N, 2) o (2,)."""
    pts = np.asarray(pts, float)
    una = pts.ndim == 1
    p = np.atleast_2d(pts)
    ph = np.column_stack([p, np.ones(len(p))]) @ H.T
    out = ph[:, :2] / ph[:, 2:3]
    return out[0] if una else out
