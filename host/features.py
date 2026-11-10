"""Rasgos léxicos de palabra y rasgos estructurales de oración."""
from __future__ import annotations

import re

try:  # Frecuencias reales en español (paquete opcional `wordfreq`)
    from wordfreq import zipf_frequency as _zipf

    def zipf(palabra: str) -> float:
        z = _zipf(palabra.lower(), "es")
        return z if z > 0 else 1.0
except ImportError:  # Aproximación si wordfreq no está instalado
    _FUNCIONALES = set(
        "el la los las un una unos unas de del al a en y o que se no con por para es su sus lo "
        "le les como más pero ya este esta ese esa muy sin sobre entre cuando también fue ha son".split()
    )

    def zipf(palabra: str) -> float:
        p = palabra.lower()
        if p in _FUNCIONALES:
            return 6.5
        return max(1.5, 6.0 - 0.35 * len(p))

_LIMPIAR = re.compile(r"[^\wáéíóúüñÁÉÍÓÚÜÑ]+")


def limpiar(palabra: str) -> str:
    return _LIMPIAR.sub("", palabra)


def palabras(texto: str) -> list[str]:
    return [p for p in texto.split() if limpiar(p)]


def rasgos_automaticos(texto: str) -> dict:
    """Rasgos calculables sin análisis sintáctico. Los sintácticos (pasiva, relativa de
    objeto, nominalizaciones, subordinadas) deben anotarse al preparar el texto
    (ver tools/anotar_texto.py) porque dependen de un parser y de revisión humana."""
    ps = [limpiar(p) for p in palabras(texto)]
    return {
        "largo_oracion": len(ps),
        "palabras_raras": sum(1 for p in ps if zipf(p) < 3.0),
        "nominalizaciones": sum(1 for p in ps if re.search(r"(ción|sión|miento|mientos|ciones)$", p.lower())),
    }
