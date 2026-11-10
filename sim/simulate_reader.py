"""Simulador de un lector con anteojos: prueba todo el análisis sin hardware.

Genera un texto sintético de N páginas con variantes, un lector con sensibilidades
"verdaderas" conocidas, y muestras oculares binoculares (mirada, pupila, parpadeos).
Las alimenta a la misma clase `Sesion` que usa el servidor real y, al final, compara el
perfil aprendido con el verdadero.

Uso:
    python -m sim.simulate_reader --paginas 40 --hz 120 --semilla 1
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from host.config import cargar_config
from host.features import limpiar, zipf
from host.session import Sesion
from host.text_model import Texto

FRECUENTES = ("el la de que y en los se del las un por con no una su para es al lo como más "
              "pero sus le ya o fue este ha sí porque esta son entre cuando muy sin sobre "
              "también me hasta hay donde quien desde todo nos durante todos uno les ni contra "
              "otros ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él "
              "tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo "
              "nosotros tiempo vida mundo casa forma parte caso día ciudad país trabajo").split()
RARAS = ("epistemología hermenéutica paradigmático idiosincrasia inextricable pusilánime "
         "conmensurable heterodoxo ontológico preeminencia subrepticio apofántico "
         "inconmensurabilidad perlocutivo deíctico anafórico isomorfismo teleológico").split()

# Sensibilidades "verdaderas" del lector simulado (en unidades de carga por unidad de rasgo escalado)
VERDAD = {"largo_oracion": 0.8, "palabras_raras": 0.3, "pasiva": 0.0,
          "relativa_objeto": 1.2, "nominalizaciones": 0.0, "subordinadas": 0.5}


# ------------------------------------------------------------------ texto sintético
def generar_texto(paginas: int, rng: np.random.Generator) -> dict:
    n_oraciones = int(paginas * 350 / 19)  # ~350 palabras por página, ~19 palabras por oración
    capitulos, parrafos, oraciones = [], [], []
    for i in range(n_oraciones):
        n = int(np.clip(rng.normal(19, 7), 6, 40))
        ws = [str(rng.choice(RARAS)) if rng.random() < 0.06 else str(rng.choice(FRECUENTES))
              for _ in range(n)]
        r = {"largo_oracion": n, "palabras_raras": sum(zipf(w) < 3 for w in ws),
             "pasiva": int(rng.random() < 0.2), "relativa_objeto": int(rng.random() < 0.15),
             "nominalizaciones": int(rng.poisson(0.5)), "subordinadas": int(rng.poisson(0.6))}
        variantes = [{"id": "orig", "texto": " ".join(ws) + ".", "rasgos": r}]
        if n > 20:
            corte = n // 2
            rr = dict(r, largo_oracion=corte, subordinadas=max(0, r["subordinadas"] - 1))
            variantes.append({"id": "dividir", "texto": " ".join(ws[:corte]) + ". " + " ".join(ws[corte:]) + ".",
                              "transformaciones": ["dividir"], "rasgos": rr})
        if r["relativa_objeto"]:
            variantes.append({"id": "svo", "texto": " ".join(ws[::-1]) + ".",
                              "transformaciones": ["relativa_a_svo"], "rasgos": dict(r, relativa_objeto=0)})
        if r["pasiva"]:
            variantes.append({"id": "activa", "texto": " ".join(ws) + ".",
                              "transformaciones": ["pasiva_a_activa"], "rasgos": dict(r, pasiva=0)})
        oraciones.append({"id": f"s{i}", "variantes": variantes})
        if len(oraciones) == 5:
            parrafos.append({"id": f"p{len(parrafos)}", "oraciones": oraciones})
            oraciones = []
        if len(parrafos) == 12:
            capitulos.append({"id": f"c{len(capitulos)}", "titulo": f"Capítulo {len(capitulos) + 1}",
                              "parrafos": parrafos})
            parrafos = []
    if oraciones:
        parrafos.append({"id": f"p{len(parrafos)}", "oraciones": oraciones})
    if parrafos:
        capitulos.append({"id": f"c{len(capitulos)}", "titulo": "Final", "parrafos": parrafos})
    return {"titulo": "Texto sintético", "capitulos": capitulos}


# ------------------------------------------------------------------- maquetación
def maquetar(texto: Texto, variantes: dict, cfg: dict):
    """Replica la maqueta del navegador: Arial 12 / 1,5 / márgenes normales, con zoom."""
    z, mm = cfg["pantalla"]["zoom"], cfg["pantalla"]["px_por_mm"]
    ancho = 150 * mm * z                 # 21 cm - 2 x 3 cm de margen
    alto_pagina = 247 * mm * z           # 29,7 cm - 2 x 2,5 cm
    paso = 12 * 1.5 * 1.15 * (96 / 72) * z  # interlineado 1,5 de Arial 12 (px CSS)
    char = 12 * 0.5 * (96 / 72) * z      # ancho medio de carácter de Arial 12
    x, y = 0.0, paso / 2
    cajas, paginas = [], {}
    for o in sorted(texto.oraciones.values(), key=lambda o: o.orden):
        v = variantes[o.id]
        ws = o.variantes[v].texto.split()
        paginas[o.id] = int(y // alto_pagina)
        for i, w in enumerate(ws):
            ancho_w = len(w) * char
            if x + ancho_w > ancho:
                x, y = 0.0, y + paso
            cajas.append({"wid": f"{o.id}|{v}|{i}", "sid": o.id, "texto": w, "orden": o.orden * 1000 + i,
                          "pagina": int(y // alto_pagina), "x0": x, "y0": y - paso * 0.35,
                          "x1": x + ancho_w, "y1": y + paso * 0.35})
            x += ancho_w + char
    return cajas, paginas


# -------------------------------------------------------------- lector simulado
class LectorSimulado:
    def __init__(self, cfg: dict, hz: int, rng: np.random.Generator):
        self.rng, self.dt = rng, 1.0 / hz
        self.t = 0.0
        self.ppd = cfg["pantalla"]["distancia_mm"] * math.tan(math.radians(1)) * cfg["pantalla"]["px_por_mm"]
        self.esc = cfg["rasgos"]["escala"]
        self.pupila = 4.0
        self.prox_parpadeo = rng.exponential(4.0)

    def carga(self, rasgos: dict) -> float:
        return sum(VERDAD[k] * rasgos[k] / self.esc[k] for k in VERDAD) - 1.2

    def _muestras(self, sesion: Sesion, x: float, y: float, dur_s: float, carga: float, deambula: bool):
        """Emite muestras binoculares de una fijación (con ruido, pupila y parpadeos)."""
        tasa = 0.5 + 0.6 * max(carga, 0.0)            # Hz de dilataciones rápidas (IPA)
        tasa_parp = (0.5 if deambula else 0.25)        # parpadeos por segundo
        n = max(1, int(dur_s / self.dt))
        for _ in range(n):
            self.t += self.dt
            if self.rng.random() < tasa * self.dt:
                self.pupila += self.rng.normal(0.08, 0.02)
            self.pupila += (4.0 - self.pupila) * 0.05 + self.rng.normal(0, 0.002)
            parpadeo = False
            self.prox_parpadeo -= self.dt
            if self.prox_parpadeo <= 0:
                parpadeo = self.prox_parpadeo > -0.15
                if not parpadeo:
                    self.prox_parpadeo = self.rng.exponential(1 / tasa_parp)
            ruido = self.rng.normal(0, 0.15 * self.ppd, 2)
            for ojo in ("L", "R"):
                conf = 0.0 if parpadeo else 0.95
                sesion.muestra_ojo(ojo, self.t, x + ruido[0], y + ruido[1], self.pupila, conf, t_host=self.t)

    def sacada(self, sesion: Sesion, x0, y0, x1, y1):
        for k in range(1, 4):
            self.t += self.dt
            xs, ys = x0 + (x1 - x0) * k / 3, y0 + (y1 - y0) * k / 3
            for ojo in ("L", "R"):
                sesion.muestra_ojo(ojo, self.t, xs, ys, self.pupila, 0.95, t_host=self.t)

    def leer_oracion(self, sesion: Sesion, cajas: list[dict], rasgos: dict, pos: list):
        carga = self.carga(rasgos) + self.rng.normal(0, 0.3)
        deambula = self.rng.random() < 0.02
        i = 0
        while i < len(cajas):
            c = cajas[i]
            w = limpiar(c["texto"])
            if len(w) <= 2 and self.rng.random() < 0.4:
                i += 1
                continue
            if deambula:
                dur = 260 + self.rng.normal(0, 40)
            else:
                dur = 180 + 9 * len(w) - 14 * (zipf(w) - 4) + 30 * max(carga, -1)
                dur *= math.exp(self.rng.normal(0, 0.2))
            x, y = (c["x0"] + c["x1"]) / 2, (c["y0"] + c["y1"]) / 2
            self.sacada(sesion, pos[0], pos[1], x, y)
            self._muestras(sesion, x, y, max(dur, 80) / 1000, carga, deambula)
            pos[:] = [x, y]
            if i > 1 and not deambula and self.rng.random() < 0.05 + 0.06 * max(carga, 0):
                j = max(0, i - int(self.rng.integers(1, 5)))
                cj = cajas[j]
                xj, yj = (cj["x0"] + cj["x1"]) / 2, (cj["y0"] + cj["y1"]) / 2
                self.sacada(sesion, x, y, xj, yj)
                self._muestras(sesion, xj, yj, 0.18, carga, deambula)
                pos[:] = [xj, yj]
            i += 1


def simular(paginas: int = 40, hz: int = 120, semilla: int = 1, verbose: bool = True) -> dict:
    rng = np.random.default_rng(semilla)
    base = max(2, round(paginas * 0.08))
    explorar = max(base + 2, round(paginas * 0.3))
    cfg = cargar_config(fases={"paginas_linea_base": base, "explorar_hasta_pagina": explorar})
    texto = Texto.desde_dict(generar_texto(paginas, rng), cfg["rasgos"]["nombres"])
    sesion = Sesion(cfg, texto, semilla=semilla)
    sesion.calibrador.identidad()
    variantes = {sid: "orig" for sid in texto.oraciones}

    cajas, pags = maquetar(texto, variantes, cfg)
    sesion.maqueta(cajas, pags, reemplazar=True)
    lector = LectorSimulado(cfg, hz, rng)
    pos = [0.0, 0.0]
    t0 = time.time()
    for o in sorted(texto.oraciones.values(), key=lambda o: o.orden):
        cambios = {}
        for msg in sesion.salida:
            if msg["tipo"] == "variantes":  # (se ignoran "estado" y "mirada")
                cambios.update(msg["decisiones"])
        sesion.salida.clear()
        if cambios:
            variantes.update(cambios)
            cajas, pags = maquetar(texto, variantes, cfg)
            sesion.maqueta(cajas, pags, reemplazar=True)
        mias = [c for c in cajas if c["sid"] == o.id]
        lector.leer_oracion(sesion, mias, o.variantes[variantes[o.id]].rasgos, pos)

    res = sesion.resultados
    dec = sesion.politica.decisiones
    tramo_c = [r for r in res if dec.get(r.sid) and dec[r.sid].tramo == "C"]
    adapt = [r.carga for r in tramo_c if r.variante != "orig"]
    ctrl = [r.carga for r in tramo_c if dec[r.sid].control]
    salida = {"oraciones_analizadas": len(res), "perfil": sesion.perfil.resumen(),
              "carga_adaptadas": float(np.mean(adapt)) if adapt else None,
              "carga_controles": float(np.mean(ctrl)) if ctrl else None,
              "n_adaptadas": len(adapt), "n_controles": len(ctrl), "segundos": time.time() - t0}
    if verbose:
        print(f"\nTexto sintético: {paginas} páginas, {len(texto.oraciones)} oraciones; "
              f"tramos A<{base} B<{explorar} C>={explorar}")
        print(f"Oraciones analizadas: {len(res)}  (simulación: {salida['segundos']:.0f} s)\n")
        print(f"{'rasgo':<18}{'verdad':>8}{'aprendido':>11}{'sd':>7}{'P(>0)':>8}")
        for p in salida["perfil"]:
            print(f"{p['rasgo']:<18}{VERDAD[p['rasgo']]:>8.2f}{p['sensibilidad']:>11.2f}"
                  f"{p['sd']:>7.2f}{p['prob_positiva']:>8.2f}")
        if adapt and ctrl:
            print(f"\nTramo C: carga media adaptadas = {salida['carga_adaptadas']:.2f} (n={len(adapt)}) "
                  f"vs controles = {salida['carga_controles']:.2f} (n={len(ctrl)})")
    return salida


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paginas", type=int, default=40)
    ap.add_argument("--hz", type=int, default=120)
    ap.add_argument("--semilla", type=int, default=1)
    a = ap.parse_args()
    simular(a.paginas, a.hz, a.semilla)
