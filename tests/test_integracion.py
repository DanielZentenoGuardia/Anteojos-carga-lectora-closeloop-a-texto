"""Pruebas de integración: texto de ejemplo, simulación corta y servidor."""
import json

import pytest

from host.config import RAIZ, cargar_config
from host.text_model import Texto
from sim.simulate_reader import simular


def test_texto_ejemplo_carga():
    cfg = cargar_config()
    t = Texto.cargar(RAIZ / "data" / "texto_ejemplo.json", cfg["rasgos"]["nombres"])
    assert len(t.oraciones) == 9
    assert t.oraciones["s3"].variantes["activa"].rasgos["pasiva"] == 0
    assert t.oraciones["s1"].original.rasgos["largo_oracion"] > 30


def test_simulacion_corta_aprende_direccion_correcta():
    r = simular(paginas=10, hz=80, semilla=3, verbose=False)
    perfil = {p["rasgo"]: p for p in r["perfil"]}
    assert r["oraciones_analizadas"] > 150
    assert perfil["largo_oracion"]["prob_positiva"] > 0.9
    assert perfil["relativa_objeto"]["sensibilidad"] > perfil["pasiva"]["sensibilidad"]


def test_servidor_entrega_texto_y_perfil():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from host.server import crear_app
    cfg = cargar_config()
    app = crear_app(str(RAIZ / "data" / "texto_ejemplo.json"))
    with TestClient(app) as c:
        r = c.get("/api/texto")
        assert r.status_code == 200 and r.json()["zoom"] == cfg["pantalla"]["zoom"]
        assert c.get("/api/perfil").json()["n_oraciones"] == 0
        with c.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"tipo": "scroll", "y": 120}))
