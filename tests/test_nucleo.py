"""Tests del núcleo de análisis (no requieren hardware). Ejecutar: pytest -q"""
import numpy as np
import pytest

from host.adapter import PoliticaAdaptacion
from host.calibration import CalibracionOjo
from host.config import cargar_config
from host.events import DetectorIVT, Fijacion, Muestra, Parpadeo
from host.geometry import aplicar, homografia
from host.metrics import ipa
from host.profile import PerfilLector
from host.text_model import Texto
from host.words import CajaPalabra, MapaPalabras


@pytest.fixture
def cfg():
    return cargar_config()


def test_ivt_detecta_fijaciones_y_parpadeo():
    det = DetectorIVT(px_por_grado=36.0, umbral_grados_s=30, fijacion_min_ms=60)
    ev, t = [], 0.0
    for _ in range(50):                       # fijación 1: 200 ms en (100, 100)
        t += 0.004
        ev += det.agregar(Muestra(t, 100, 100, True))
    for k in range(1, 6):                     # sacada
        t += 0.004
        ev += det.agregar(Muestra(t, 100 + 40 * k, 100, True))
    for _ in range(50):                       # fijación 2
        t += 0.004
        ev += det.agregar(Muestra(t, 300, 100, True))
    for _ in range(30):                       # parpadeo de 120 ms
        t += 0.004
        ev += det.agregar(Muestra(t, np.nan, np.nan, False))
    t += 0.004
    ev += det.agregar(Muestra(t, 300, 100, True))
    ev += det.vaciar()
    fij = [e for e in ev if isinstance(e, Fijacion)]
    parp = [e for e in ev if isinstance(e, Parpadeo)]
    assert len(fij) == 2
    assert fij[0].x == pytest.approx(100) and fij[1].x == pytest.approx(300)
    assert 150 < fij[0].duracion_ms < 220
    assert len(parp) == 1 and 100 < parp[0].duracion_ms < 160


def test_ipa_aumenta_con_dilataciones_rapidas():
    rng = np.random.default_rng(0)
    fs, dur = 250, 8.0
    n = int(fs * dur)
    tranquila = 4 + 0.002 * rng.standard_normal(n)
    activa = tranquila.copy()
    for i in rng.choice(n - 5, 40, replace=False):
        activa[i:] += 0.05
    assert ipa(activa, dur) > ipa(tranquila, dur)


def test_homografia_recupera_transformacion():
    H = np.array([[1.2, 0.1, 30], [-0.05, 0.9, 12], [1e-4, 2e-4, 1]])
    src = np.array([[0, 0], [640, 0], [640, 480], [0, 480], [320, 240], [100, 400]], float)
    dst = aplicar(H, src)
    H2 = homografia(src, dst)
    assert np.allclose(aplicar(H2, src), dst, atol=1e-6)


def test_calibracion_polinomial():
    rng = np.random.default_rng(1)
    pup = rng.uniform(100, 500, (9, 2))
    obj = np.column_stack([3 * pup[:, 0] + 0.001 * pup[:, 0] ** 2, 2 * pup[:, 1] - 50])
    cal = CalibracionOjo()
    err = cal.ajustar(pup, obj)
    assert err < 1e-6
    assert np.allclose(cal.predecir(pup[0]), obj[0])


def test_mapa_palabras_corrige_deriva():
    cajas = []
    for linea in range(5):
        for i in range(6):
            cajas.append(CajaPalabra(f"s{linea}|orig|{i}", f"s{linea}", "palabra", linea * 1000 + i, 0,
                                     i * 100, linea * 40 + 10, i * 100 + 80, linea * 40 + 30))
    mapa = MapaPalabras()
    mapa.actualizar(cajas)
    aciertos = 0
    for k in range(30):  # mirada con deriva vertical sistemática de +14 px
        linea, i = k % 5, k % 6
        c = mapa.asignar(i * 100 + 40, linea * 40 + 20 + 14)
        aciertos += c is not None and c.wid == f"s{linea}|orig|{i}"
    assert aciertos >= 25
    assert mapa.deriva_y > 5


def test_perfil_recupera_sensibilidades(cfg):
    perfil = PerfilLector.desde_config(cfg)
    rng = np.random.default_rng(2)
    verdad = {"largo_oracion": 1.0, "relativa_objeto": 1.5, "pasiva": 0.0}
    for _ in range(600):
        r = {"largo_oracion": rng.integers(5, 40), "relativa_objeto": rng.integers(0, 2),
             "pasiva": rng.integers(0, 2), "palabras_raras": 0, "nominalizaciones": 0, "subordinadas": 0}
        y = sum(verdad[k] * r[k] / cfg["rasgos"]["escala"][k] for k in verdad) + rng.normal(0, 1)
        perfil.actualizar(r, y)
    res = {p["rasgo"]: p for p in perfil.resumen()}
    assert res["largo_oracion"]["sensibilidad"] == pytest.approx(1.0, abs=0.2)
    assert res["relativa_objeto"]["sensibilidad"] == pytest.approx(1.5, abs=0.3)
    assert res["relativa_objeto"]["prob_positiva"] > 0.99
    assert abs(res["pasiva"]["sensibilidad"]) < 0.3


def _texto(cfg):
    d = {"capitulos": [{"id": "c1", "parrafos": [{"id": "p", "oraciones": [
        {"id": f"s{i}", "variantes": [
            {"id": "orig", "texto": "una oración larga " * 10, "rasgos": {"largo_oracion": 30}},
            {"id": "dividir", "texto": "una oración. larga " * 10, "rasgos": {"largo_oracion": 15}}]}
        for i in range(400)]}]}]}
    return Texto.desde_dict(d, cfg["rasgos"]["nombres"])


def test_politica_respeta_adelanto_y_tramos(cfg):
    texto = _texto(cfg)
    pol = PoliticaAdaptacion(cfg, PerfilLector.desde_config(cfg), semilla=0)
    o = texto.oraciones["s0"]
    assert pol.decidir(o, pagina_oracion=5, pagina_lectura=5) is None      # página actual: no se toca
    assert pol.decidir(o, pagina_oracion=6, pagina_lectura=5).variante == "orig"  # tramo A
    tramo_b = [pol.decidir(texto.oraciones[f"s{i}"], 10, 9) for i in range(1, 201)]
    frac = np.mean([d.variante != "orig" for d in tramo_b])
    assert 0.35 < frac < 0.65                                               # exploración ~50 %


def test_politica_adapta_con_evidencia_y_deja_controles(cfg):
    texto = _texto(cfg)
    perfil = PerfilLector.desde_config(cfg)
    for _ in range(300):  # el lector carga mucho con oraciones largas
        perfil.actualizar({"largo_oracion": 30}, 2.0)
        perfil.actualizar({"largo_oracion": 15}, 0.0)
    pol = PoliticaAdaptacion(cfg, perfil, semilla=0)
    ds = [pol.decidir(texto.oraciones[f"s{i}"], 40, 38) for i in range(200, 400)]
    adaptadas = np.mean([d.variante == "dividir" for d in ds])
    controles = np.mean([d.control for d in ds])
    assert adaptadas > 0.7
    assert 0.08 < controles < 0.3


def test_ivt_robusto_al_ruido_a_250hz():
    """0,1° de ruido por muestra a 250 Hz no debe fragmentar una fijación de 300 ms."""
    rng = np.random.default_rng(5)
    det = DetectorIVT(px_por_grado=36.0)
    ev, t = [], 0.0
    for _ in range(75):
        t += 0.004
        dx, dy = rng.normal(0, 0.1 * 36, 2)
        ev += det.agregar(Muestra(t, 200 + dx, 200 + dy, True))
    ev += det.vaciar()
    fij = [e for e in ev if isinstance(e, Fijacion)]
    assert len(fij) == 1 and fij[0].duracion_ms > 250
