"""Sesión de lectura: une las señales de los anteojos, la maqueta del texto y la adaptación.

Flujo por muestra:  pupila (Pi) -> calibración -> pantalla -> documento -> I-VT -> fijación
-> palabra -> P1..P4 -> carga por oración L_s -> perfil del lector -> decisiones de variantes.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adapter import PoliticaAdaptacion
from .calibration import Calibrador
from .config import px_por_grado
from .events import DetectorIVT, Fijacion, Muestra, Parpadeo
from .geometry import aplicar, homografia
from .metrics import AnalizadorLectura
from .profile import PerfilLector
from .text_model import Texto
from .words import CajaPalabra, MapaPalabras


@dataclass
class _MuestraOjo:
    t: float
    xy: np.ndarray | None
    mm: float
    valida: bool


class Sesion:
    def __init__(self, cfg: dict, texto: Texto, ruta_log: str | Path | None = None,
                 semilla: int | None = None):
        self.cfg = cfg
        self.texto = texto
        e = cfg["eventos"]
        self.ivt = DetectorIVT(px_por_grado(cfg), e["umbral_ivt_grados_s"], e["fijacion_min_ms"],
                               e["parpadeo_min_ms"], e["parpadeo_max_ms"])
        self.dt_bino = e["max_dt_binocular_ms"] / 1000.0
        self.mapa = MapaPalabras()
        self.analizador = AnalizadorLectura(cfg)
        self.perfil = PerfilLector.desde_config(cfg)
        self.politica = PoliticaAdaptacion(cfg, self.perfil, semilla)
        self.calibrador = Calibrador()
        self.H: np.ndarray | None = None          # escena -> pantalla
        self.marcadores_pantalla: dict[str, list] = {}
        self.scroll_y = 0.0
        self.pagina_lectura = 0
        self.pagina_oracion: dict[str, int] = {}
        self.oraciones_por_pagina: dict[int, set] = defaultdict(set)
        self.variante_mostrada: dict[str, str] = {sid: "orig" for sid in texto.oraciones}
        self.salida: list[dict] = []               # mensajes pendientes para el navegador
        self.resultados: list = []
        self._ult = {"L": None, "R": None}
        self._t_estado = 0.0
        self._t_mirada = 0.0
        self._log = open(ruta_log, "a", encoding="utf-8") if ruta_log else None

    # ------------------------------------------------------------------ registro
    def _registrar(self, tipo: str, **datos):
        if self._log:
            self._log.write(json.dumps({"tipo": tipo, "t_host": time.time(), **datos},
                                       ensure_ascii=False, default=float) + "\n")

    def cerrar(self):
        if self._log:
            self._registrar("perfil_final", perfil=self.perfil.resumen())
            self._log.close()
            self._log = None

    # ------------------------------------------------------- entradas del navegador
    def maqueta(self, palabras: list[dict], paginas_oracion: dict[str, int], reemplazar: bool = False):
        cajas = [CajaPalabra(p["wid"], p["sid"], p["texto"], int(p["orden"]), int(p["pagina"]),
                             p["x0"], p["y0"], p["x1"], p["y1"]) for p in palabras]
        self.mapa.actualizar(cajas, reemplazar=reemplazar)
        for sid, pag in paginas_oracion.items():
            viejo = self.pagina_oracion.get(sid)
            if viejo is not None:
                self.oraciones_por_pagina[viejo].discard(sid)
            self.pagina_oracion[sid] = int(pag)
            self.oraciones_por_pagina[int(pag)].add(sid)
        por_oracion = defaultdict(list)
        for c in cajas:
            por_oracion[c.sid].append(c)
        for sid, cs in por_oracion.items():
            variante = cs[0].wid.split("|")[1]
            ordenes = [c.orden for c in cs]
            self.analizador.registrar_oracion(sid, variante, min(ordenes), max(ordenes), len(cs))
        self._decidir()

    def scroll(self, y: float):
        self.scroll_y = float(y)

    def marcadores_en_pantalla(self, esquinas: dict[str, list]):
        self.marcadores_pantalla = {str(k): v for k, v in esquinas.items()}

    # ------------------------------------------------------------ entradas del Pi
    def marcadores_en_escena(self, esquinas: dict[str, list]):
        origen, destino = [], []
        for k, pts in esquinas.items():
            if str(k) in self.marcadores_pantalla:
                origen += pts
                destino += self.marcadores_pantalla[str(k)]
        if len(origen) >= 8:  # al menos 2 marcadores completos
            self.H = homografia(np.asarray(origen), np.asarray(destino))

    def muestra_ojo(self, ojo: str, t: float, px: float, py: float, mm: float, conf: float,
                    t_host: float | None = None):
        if self.calibrador.activa:
            self.calibrador.muestra(ojo, px, py, conf, time.monotonic() if t_host is None else t_host)
            return
        cal = self.calibrador.ojos[ojo]
        valida = conf >= 0.5 and cal.coef is not None
        xy = None
        if valida:
            esc = cal.predecir((px, py))
            pant = aplicar(self.H, esc) if self.H is not None else esc
            xy = np.array([pant[0], pant[1] + self.scroll_y])
        actual = _MuestraOjo(t, xy, mm if valida else float("nan"), valida)
        otro = "R" if ojo == "L" else "L"
        o = self._ult[otro]
        if o is not None and abs(o.t - t) <= self.dt_bino:
            self._ult[otro] = None
            self._procesar(self._combinar(o, actual))
            return
        self._ult[ojo] = actual
        if o is not None and t - o.t > 3 * self.dt_bino:
            self._ult[otro] = None
            self._procesar(Muestra(o.t, *(o.xy if o.valida else (np.nan, np.nan)), o.valida, o.mm))

    @staticmethod
    def _combinar(a: _MuestraOjo, b: _MuestraOjo) -> Muestra:
        validas = [m for m in (a, b) if m.valida]
        t = max(a.t, b.t)
        if not validas:
            return Muestra(t, np.nan, np.nan, False)
        xy = np.mean([m.xy for m in validas], axis=0)
        mm = float(np.nanmean([m.mm for m in validas]))
        return Muestra(t, float(xy[0]), float(xy[1]), True, mm)

    def ambiente(self, t: float, lux: float | None = None, **_):
        self._registrar("ambiente", t=t, lux=lux)

    # ------------------------------------------------------------ procesamiento
    def _procesar(self, m: Muestra):
        if m.valida and np.isfinite(m.pupila_mm):
            self.analizador.pupila(m.t, m.pupila_mm)
        for ev in self.ivt.agregar(m):
            if isinstance(ev, Parpadeo):
                self.analizador.parpadeo(ev)
                self._registrar("parpadeo", t0=ev.t_ini, t1=ev.t_fin)
            elif isinstance(ev, Fijacion):
                self._fijacion(ev)
        if m.valida and m.t - self._t_mirada >= 0.1:
            self._t_mirada = m.t
            self.salida.append({"tipo": "mirada", "x": m.x, "y": m.y - self.scroll_y})
        if m.t - self._t_estado >= 2.0:
            self._t_estado = m.t
            est = self.analizador.estado(m.t)
            self.salida.append({"tipo": "estado", **est, "pagina": self.pagina_lectura})

    def _fijacion(self, f: Fijacion):
        caja = self.mapa.asignar(f.x, f.y)
        self._registrar("fijacion", t0=f.t_ini, dur=f.duracion_ms, x=f.x, y=f.y,
                        wid=caja.wid if caja else None)
        for r in self.analizador.fijacion(f, caja):
            self.resultados.append(r)
            if not self.analizador.en_linea_base:
                rasgos = self.texto.oraciones[r.sid].variantes[r.variante].rasgos
                self.perfil.actualizar(rasgos, r.carga)
            d = self.politica.decisiones.get(r.sid)
            self._registrar("oracion", sid=r.sid, variante=r.variante, carga=r.carga, z=r.z,
                            p1=r.p1_residuo_ms, p2r=r.p2_tasa_regresion, p2g=r.p2_gopast_ms_palabra,
                            p3=r.p3_ipa_hz, tramo=d.tramo if d else "A",
                            control=d.control if d else False)
        if caja is not None and caja.pagina > self.pagina_lectura:
            self.pagina_lectura = caja.pagina
            if self.analizador.en_linea_base and self.pagina_lectura >= self.politica.pag_base:
                self.analizador.congelar_linea_base()
                self._registrar("linea_base_congelada", pagina=self.pagina_lectura)
            self._decidir()

    def _decidir(self):
        """Decide variantes para las oraciones de las próximas páginas (nunca la actual)."""
        cambios = {}
        p0 = self.pagina_lectura + self.politica.adelanto
        for pag in range(p0, p0 + 2):
            for sid in sorted(self.oraciones_por_pagina.get(pag, ()),
                              key=lambda s: self.texto.oraciones[s].orden):
                d = self.politica.decidir(self.texto.oraciones[sid], pag, self.pagina_lectura)
                if d and d.variante != self.variante_mostrada.get(sid):
                    self.variante_mostrada[sid] = d.variante
                    cambios[sid] = d.variante
                if d:
                    self._registrar("decision", sid=sid, variante=d.variante, tramo=d.tramo,
                                    motivo=d.motivo, control=d.control)
        if cambios:
            self.salida.append({"tipo": "variantes", "decisiones": cambios})

    # ---------------------------------------------------------------- calibración
    def calibracion_inicio(self):
        self.calibrador.iniciar()

    def calibracion_punto(self, idx: int, x: float, y: float, t_host: float | None = None):
        H_inv = np.linalg.inv(self.H) if self.H is not None else None
        self.calibrador.objetivo(idx, x, y, time.monotonic() if t_host is None else t_host, H_inv)

    def calibracion_fin(self) -> dict:
        errores = self.calibrador.finalizar()
        ppd = px_por_grado(self.cfg)
        res = {o: (round(e / ppd, 2) if e is not None else None) for o, e in errores.items()}
        self._registrar("calibracion", error_grados=res)
        self.salida.append({"tipo": "calibracion", "error_grados": res})
        return res

    def respuesta(self, capitulo: str, pregunta: int, eleccion: int, correcta: bool):
        self._registrar("comprension", capitulo=capitulo, pregunta=pregunta,
                        eleccion=eleccion, correcta=correcta)
