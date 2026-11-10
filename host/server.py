"""Servidor del computador: recibe los datos del Pi (UDP) y atiende la app de lectura (WebSocket).

Uso:
    python -m host.server --texto data/texto_ejemplo.json
y abrir http://localhost:8000 en el navegador (pantalla completa).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import RAIZ, cargar_config
from .session import Sesion
from .text_model import Texto

WEB = Path(__file__).parent / "web"


class ProtocoloPi(asyncio.DatagramProtocol):
    """Mensajes JSON del Raspberry Pi (ver pi/main.py)."""

    def __init__(self, sesion: Sesion):
        self.sesion = sesion

    def datagram_received(self, data: bytes, addr):
        t_host = time.monotonic()
        for linea in data.decode("utf-8", "ignore").splitlines():
            try:
                m = json.loads(linea)
            except json.JSONDecodeError:
                continue
            tipo = m.get("tipo")
            if tipo == "ojo":
                self.sesion.muestra_ojo(m["ojo"], m["t"], m["x"], m["y"], m["mm"], m["conf"], t_host)
            elif tipo == "marcadores":
                self.sesion.marcadores_en_escena(m["esquinas"])
            elif tipo == "ambiente":
                self.sesion.ambiente(m["t"], m.get("lux"))


def crear_app(ruta_texto: str, ruta_config: str | None = None) -> FastAPI:
    cfg = cargar_config(ruta_config)
    texto = Texto.cargar(ruta_texto, cfg["rasgos"]["nombres"])
    carpeta = RAIZ / "data" / "sesiones"
    carpeta.mkdir(parents=True, exist_ok=True)
    marca = time.strftime("%Y%m%d-%H%M%S")
    sesion = Sesion(cfg, texto, ruta_log=carpeta / f"sesion-{marca}.jsonl")
    clientes: set[WebSocket] = set()

    async def despachar():
        while True:
            await asyncio.sleep(0.05)
            if not sesion.salida or not clientes:
                continue
            mensajes, sesion.salida[:] = list(sesion.salida), []
            for ws in list(clientes):
                for m in mensajes:
                    try:
                        await ws.send_json(m)
                    except Exception:
                        clientes.discard(ws)

    @asynccontextmanager
    async def ciclo(app: FastAPI):
        loop = asyncio.get_running_loop()
        transporte, _ = await loop.create_datagram_endpoint(
            lambda: ProtocoloPi(sesion), local_addr=("0.0.0.0", cfg["red"]["udp_port"]))
        tarea = asyncio.create_task(despachar())
        yield
        tarea.cancel()
        transporte.close()
        sesion.perfil.guardar(carpeta / f"perfil-{marca}.json")
        sesion.cerrar()

    app = FastAPI(title="Anteojos de carga lectora", lifespan=ciclo)
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    async def inicio():
        return FileResponse(WEB / "index.html")

    @app.get("/api/texto")
    async def api_texto():
        return JSONResponse({"texto": texto.crudo, "zoom": cfg["pantalla"]["zoom"],
                             "adelanto": cfg["fases"]["paginas_adelanto"]})

    @app.get("/api/perfil")
    async def api_perfil():
        return {"n_oraciones": sesion.perfil.n, "rasgos": sesion.perfil.resumen()}

    @app.websocket("/ws")
    async def ws_lectura(ws: WebSocket):
        await ws.accept()
        clientes.add(ws)
        try:
            while True:
                m = await ws.receive_json()
                tipo = m.get("tipo")
                if tipo == "maqueta":
                    sesion.maqueta(m["palabras"], m["paginas_oracion"], m.get("reemplazar", False))
                elif tipo == "scroll":
                    sesion.scroll(m["y"])
                elif tipo == "marcadores":
                    sesion.marcadores_en_pantalla(m["esquinas"])
                elif tipo == "calib_inicio":
                    sesion.calibracion_inicio()
                elif tipo == "calib_punto":
                    sesion.calibracion_punto(m["idx"], m["x"], m["y"])
                elif tipo == "calib_fin":
                    sesion.calibracion_fin()
                elif tipo == "respuesta":
                    sesion.respuesta(m["capitulo"], m["pregunta"], m["eleccion"], m["correcta"])
        except WebSocketDisconnect:
            clientes.discard(ws)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texto", default=str(RAIZ / "data" / "texto_ejemplo.json"))
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = cargar_config(a.config)
    uvicorn.run(crear_app(a.texto, a.config), host="0.0.0.0", port=cfg["red"]["http_port"])


if __name__ == "__main__":
    main()
