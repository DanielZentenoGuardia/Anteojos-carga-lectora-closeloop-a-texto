"""Modelo del texto: capítulos, párrafos, oraciones y sus variantes pre-generadas.

Formato JSON esperado (ver data/texto_ejemplo.json):
{
  "titulo": "...",
  "capitulos": [{
    "id": "c1", "titulo": "...",
    "parrafos": [{"id": "c1p1", "oraciones": [{
        "id": "s1",
        "variantes": [
          {"id": "orig", "texto": "...", "rasgos": {...}},
          {"id": "dividir", "texto": "...", "transformaciones": ["dividir"], "rasgos": {...}}
        ]}]}],
    "preguntas": [{"pregunta": "...", "opciones": ["..."], "correcta": 0}]
  }]
}
Todas las variantes se escriben y revisan ANTES de la primera lectura: durante la sesión
el sistema solo elige cuál mostrar (no genera texto nuevo).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .features import rasgos_automaticos


@dataclass
class Variante:
    id: str
    texto: str
    rasgos: dict
    transformaciones: list = field(default_factory=list)
    surprisal: list | None = None


@dataclass
class Oracion:
    id: str
    capitulo: str
    orden: int
    variantes: dict[str, Variante]

    @property
    def original(self) -> Variante:
        return self.variantes["orig"]


@dataclass
class Texto:
    titulo: str
    oraciones: dict[str, Oracion]
    crudo: dict

    @classmethod
    def cargar(cls, ruta: str | Path, nombres_rasgos: list[str]) -> "Texto":
        with open(ruta, encoding="utf-8") as f:
            return cls.desde_dict(json.load(f), nombres_rasgos)

    @classmethod
    def desde_dict(cls, d: dict, nombres_rasgos: list[str]) -> "Texto":
        oraciones: dict[str, Oracion] = {}
        orden = 0
        for cap in d["capitulos"]:
            for par in cap["parrafos"]:
                for o in par["oraciones"]:
                    vs = {}
                    for v in o["variantes"]:
                        r = {k: 0.0 for k in nombres_rasgos}
                        r.update({k: v2 for k, v2 in rasgos_automaticos(v["texto"]).items() if k in r})
                        r.update(v.get("rasgos", {}))
                        vs[v["id"]] = Variante(v["id"], v["texto"], r,
                                               v.get("transformaciones", []), v.get("surprisal"))
                    if "orig" not in vs:
                        raise ValueError(f"La oración {o['id']} no tiene variante 'orig'")
                    oraciones[o["id"]] = Oracion(o["id"], cap["id"], orden, vs)
                    orden += 1
        return cls(d.get("titulo", ""), oraciones, d)
