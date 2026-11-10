"""Los cuatro parámetros oculares de carga cognitiva y el índice de carga por oración.

P1  Duración de fijación residual (gaze duration observada - predicha por largo, frecuencia
    y surprisal). Smith & Levy (2013) doi:10.1016/j.cognition.2013.02.013;
    Shain et al. (2024) doi:10.1073/pnas.2307876121
    + pendiente duración~predictibilidad: se aplana al desacoplarse la atención.
    Reichle, Reineberg & Schooler (2010) doi:10.1177/0956797610378686
P2  Regresiones y go-past time. Rayner (1998) doi:10.1037/0033-2909.124.3.372;
    Staub (2010) doi:10.1016/j.cognition.2010.04.002
P3  Index of Pupillary Activity (IPA). Duchowski et al. (2018) doi:10.1145/3173574.3173856
    (aquí con ondícula de Haar; el original usa sym16).
P4  Tasa de parpadeo. Smilek, Carriere & Cheyne (2010) doi:10.1177/0956797610368063
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .events import Fijacion, Parpadeo
from .features import limpiar, zipf
from .words import CajaPalabra


# ----------------------------------------------------------------- P3: IPA
def ipa(diametro: np.ndarray, duracion_s: float) -> float:
    """Frecuencia (Hz) de cambios pupilares rápidos: máximos de módulo de los coeficientes
    de detalle (nivel 2, Haar) que superan el umbral universal sigma*sqrt(2 ln n)."""
    d = np.asarray(diametro, float)
    if duracion_s <= 0 or d.size < 16:
        return float("nan")
    malos = ~np.isfinite(d)
    if malos.all():
        return float("nan")
    if malos.any():
        idx = np.arange(d.size)
        d[malos] = np.interp(idx[malos], idx[~malos], d[~malos])
    a = d
    det = None
    for _ in range(2):
        if a.size % 2:
            a = a[:-1]
        par, impar = a[0::2], a[1::2]
        det = (par - impar) / math.sqrt(2)
        a = (par + impar) / math.sqrt(2)
    c = np.abs(det)
    if c.size < 3:
        return float("nan")
    sigma = np.median(np.abs(det - np.median(det))) / 0.6745
    umbral = sigma * math.sqrt(2 * math.log(c.size))
    maximos = (c[1:-1] > c[:-2]) & (c[1:-1] >= c[2:]) & (c[1:-1] > umbral)
    return float(maximos.sum() / duracion_s)


# --------------------------------------------------------- estadística base
@dataclass
class Base:
    """Media y desviación de una métrica durante la línea base del lector."""
    valores: list = field(default_factory=list)
    congelada: bool = False
    media: float = 0.0
    sd: float = 1.0

    def agregar(self, v: float):
        if self.congelada or not np.isfinite(v):
            return
        self.valores.append(v)
        if len(self.valores) >= 3:
            self.media = float(np.mean(self.valores))
            self.sd = max(float(np.std(self.valores, ddof=1)), 1e-6)

    def z(self, v: float) -> float:
        if not np.isfinite(v) or len(self.valores) < 3:
            return 0.0
        return float(np.clip((v - self.media) / self.sd, -5, 5))


# ------------------------------------------------------------ P1: léxico
class ModeloLexico:
    """gaze duration ~ intercepto + largo + zipf + surprisal (ajustado por lector)."""

    NOMBRES = ("intercepto", "largo", "zipf", "surprisal")

    def __init__(self, por_defecto: dict, lambda_ridge: float = 50.0):
        self.beta0 = np.array([por_defecto[k] for k in self.NOMBRES], float)
        self.beta = self.beta0.copy()
        self.lam = lambda_ridge
        self._X, self._y = [], []

    @staticmethod
    def vector(texto: str, surprisal: float | None) -> np.ndarray:
        p = limpiar(texto)
        return np.array([1.0, len(p), zipf(p), surprisal if surprisal is not None else 0.0])

    def agregar(self, x: np.ndarray, gd: float):
        self._X.append(x)
        self._y.append(gd)

    def ajustar(self):
        if len(self._y) < 30:
            return
        X, y = np.asarray(self._X), np.asarray(self._y)
        A = X.T @ X + self.lam * np.eye(4)
        self.beta = np.linalg.solve(A, X.T @ y + self.lam * self.beta0)

    def predecir(self, x: np.ndarray) -> float:
        return float(x @ self.beta)


# ------------------------------------------------------- registros de oración
@dataclass
class RegistroOracion:
    sid: str
    variante: str
    orden_min: int
    orden_max: int
    n_palabras: int
    t_entrada: float | None = None
    t_salida: float | None = None
    gopast_ms: float = 0.0
    sacadas: int = 0
    regresiones: int = 0
    residuos: list = field(default_factory=list)
    cerrada: bool = False


@dataclass
class ResultadoOracion:
    sid: str
    variante: str
    t_salida: float
    p1_residuo_ms: float
    p2_tasa_regresion: float
    p2_gopast_ms_palabra: float
    p3_ipa_hz: float
    z: dict
    carga: float           # índice L_s (combinación ponderada de z)


class AnalizadorLectura:
    """Recibe fijaciones asignadas a palabras, parpadeos y pupila; produce L_s por oración."""

    def __init__(self, cfg: dict):
        c = cfg["carga"]
        self.pesos = c["pesos"]
        self.ventana_ipa = c["ventana_ipa_s"]
        self.ventana_parp = c["ventana_parpadeo_s"]
        self.lexico = ModeloLexico(cfg["lexico_por_defecto"])
        self.bases = {k: Base() for k in ("p1", "p2_regresiones", "p2_gopast", "p3_ipa",
                                           "pendiente", "parpadeo")}
        self.oraciones: dict[str, RegistroOracion] = {}
        self._pupila = deque()                   # (t, mm), últimos ~30 s
        self._parpadeos = deque()                # t de cada parpadeo
        self._pend = deque(maxlen=c["ventana_pendiente_palabras"])  # (predictor, gd)
        self._ult_caja: CajaPalabra | None = None
        self._max_orden = -1
        self._cur_wid = None
        self._cur_fp = False
        self._cur_dur = 0.0
        self._cur_caja: CajaPalabra | None = None
        self._vistas: set[str] = set()
        self.en_linea_base = True
        self.ultima_carga = 0.0

    # ------------------------------------------------------------- registro
    def registrar_oracion(self, sid, variante, orden_min, orden_max, n_palabras):
        r = self.oraciones.get(sid)
        if r is None or (r.t_entrada is None and r.variante != variante):
            self.oraciones[sid] = RegistroOracion(sid, variante, orden_min, orden_max, n_palabras)
        else:
            r.orden_min, r.orden_max, r.n_palabras = orden_min, orden_max, n_palabras

    def pupila(self, t: float, mm: float):
        self._pupila.append((t, mm))
        while self._pupila and t - self._pupila[0][0] > 30:
            self._pupila.popleft()

    def parpadeo(self, p: Parpadeo):
        self._parpadeos.append(p.t_ini)
        while self._parpadeos and p.t_ini - self._parpadeos[0] > self.ventana_parp:
            self._parpadeos.popleft()

    def tasa_parpadeo(self, t: float) -> float:
        n = sum(1 for tb in self._parpadeos if t - tb <= self.ventana_parp)
        return n * 60.0 / self.ventana_parp

    # ------------------------------------------------------------ fijaciones
    def _cerrar_palabra(self):
        if self._cur_caja is not None and self._cur_fp:
            caja = self._cur_caja
            x = ModeloLexico.vector(caja.texto, None)
            gd = self._cur_dur
            if self.en_linea_base:
                self.lexico.agregar(x, gd)
            res = gd - self.lexico.predecir(x)
            reg = self.oraciones.get(caja.sid)
            if reg is not None:
                reg.residuos.append(res)
            self._pend.append((x[2], gd))

    def fijacion(self, f: Fijacion, caja: CajaPalabra | None) -> list[ResultadoOracion]:
        dur = f.duracion_ms
        resultados: list[ResultadoOracion] = []
        abiertas = [r for r in self.oraciones.values() if r.t_entrada is not None and not r.cerrada]
        if caja is None:
            for r in abiertas:
                r.gopast_ms += dur
            return resultados

        o = caja.orden
        # P1: primera pasada (gaze duration)
        if caja.wid == self._cur_wid:
            self._cur_dur += dur
        else:
            self._cerrar_palabra()
            self._cur_wid, self._cur_caja, self._cur_dur = caja.wid, caja, dur
            self._cur_fp = caja.wid not in self._vistas and o > self._max_orden
            self._vistas.add(caja.wid)

        # P2: sacadas y regresiones (atribuidas a la oración de origen)
        if self._ult_caja is not None and self._ult_caja.wid != caja.wid:
            origen = self.oraciones.get(self._ult_caja.sid)
            if origen is not None and not origen.cerrada:
                origen.sacadas += 1
                if o < self._ult_caja.orden:
                    origen.regresiones += 1
        self._ult_caja = caja

        # Go-past: desde la entrada a la oración hasta la primera fijación a su derecha
        reg = self.oraciones.get(caja.sid)
        if reg is not None and reg.t_entrada is None and o >= self._max_orden:
            reg.t_entrada = f.t_ini
        for r in list(abiertas) + ([reg] if reg is not None and reg not in abiertas
                                   and reg.t_entrada is not None else []):
            if r.cerrada:
                continue
            if o > r.orden_max:
                r.t_salida = f.t_ini
                r.cerrada = True
                resultados.append(self._cerrar_oracion(r))
            else:
                r.gopast_ms += dur
        self._max_orden = max(self._max_orden, o)
        return resultados

    def _cerrar_oracion(self, r: RegistroOracion) -> ResultadoOracion:
        p1 = float(np.mean(r.residuos)) if r.residuos else float("nan")
        p2r = r.regresiones / max(r.sacadas, 1)
        p2g = r.gopast_ms / max(r.n_palabras, 1)
        t1 = r.t_salida
        t0 = min(r.t_entrada, t1 - self.ventana_ipa)
        d = np.array([mm for t, mm in self._pupila if t0 <= t <= t1])
        p3 = ipa(d, t1 - t0)
        crudos = {"p1": p1, "p2_regresiones": p2r, "p2_gopast": p2g, "p3_ipa": p3}
        if self.en_linea_base:
            for k, v in crudos.items():
                self.bases[k].agregar(v)
        z = {k: self.bases[k].z(v) for k, v in crudos.items()}
        carga = sum(self.pesos[k] * z[k] for k in self.pesos)
        self.ultima_carga = carga
        return ResultadoOracion(r.sid, r.variante, t1, p1, p2r, p2g, p3, z, carga)

    # ----------------------------------------------------------------- estado
    def pendiente(self) -> float:
        """Pendiente de la duración sobre la frecuencia (ms por unidad Zipf; normalmente < 0).
        Si se acerca a 0, la lectura dejó de responder al texto (desacoplamiento)."""
        if len(self._pend) < 20:
            return float("nan")
        x, y = np.array(self._pend).T
        if np.std(x) < 1e-6:
            return float("nan")
        return float(np.polyfit(x, y, 1)[0])

    def estado(self, t: float) -> dict:
        pend = self.pendiente()
        parp = self.tasa_parpadeo(t)
        if self.en_linea_base:
            self.bases["pendiente"].agregar(pend)
            self.bases["parpadeo"].agregar(parp)
        # La pendiente es negativa: aplanarse = z positivo respecto de la base
        z_pend = self.bases["pendiente"].z(pend)
        z_parp = self.bases["parpadeo"].z(parp)
        if z_pend > 1.0 and z_parp > 0.5:
            nombre = "desacoplamiento"
        elif self.ultima_carga > 1.0:
            nombre = "sobrecarga"
        else:
            nombre = "optima"
        return {"estado": nombre, "carga": round(self.ultima_carga, 3),
                "z_pendiente": round(z_pend, 2), "z_parpadeo": round(z_parp, 2),
                "parpadeos_min": round(parp, 1)}

    def congelar_linea_base(self):
        self.lexico.ajustar()
        for b in self.bases.values():
            b.congelada = True
        self.en_linea_base = False
