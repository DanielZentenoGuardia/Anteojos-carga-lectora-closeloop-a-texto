"""Carga de la configuración del sistema (config/config.json)."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_POR_DEFECTO = RAIZ / "config" / "config.json"


def _mezclar(base: dict, extra: dict) -> dict:
    """Mezcla recursiva: los valores de `extra` reemplazan a los de `base`."""
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _mezclar(out[k], v)
        else:
            out[k] = v
    return out


def cargar_config(ruta: str | Path | None = None, **sobrescribir) -> dict:
    """Lee la configuración y aplica sobrescrituras (útil para el simulador y los tests)."""
    with open(ruta or RUTA_POR_DEFECTO, encoding="utf-8") as f:
        cfg = json.load(f)
    return _mezclar(cfg, sobrescribir) if sobrescribir else cfg


def px_por_grado(cfg: dict) -> float:
    """Píxeles CSS que subtiende 1° de ángulo visual a la distancia configurada."""
    p = cfg["pantalla"]
    return p["distancia_mm"] * math.tan(math.radians(1.0)) * p["px_por_mm"]
