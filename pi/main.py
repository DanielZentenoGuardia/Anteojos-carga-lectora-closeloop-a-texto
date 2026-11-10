"""Programa del Raspberry Pi: detecta pupilas a alta frecuencia y envía datos livianos al computador.

Se envían por UDP líneas JSON (pocos KB/s); nunca video crudo (≈1,2 Gbit/s a 300 fps):
  {"tipo":"ojo","ojo":"L","t":...,"x":...,"y":...,"mm":...,"conf":...}
  {"tipo":"marcadores","t":...,"esquinas":{"0":[[x,y]x4],...}}
  {"tipo":"ambiente","t":...,"lux":...,"acc_g":[...],"gir_dps":[...]}

Uso (en el Pi):  python -m pi.main --host 192.168.1.50
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time

from host.config import cargar_config

from .cameras import CamaraEscena, CamaraOjo
from .pupil import DetectorPupila
from .sensors import MPU6050, VEML7700, Salidas, abrir_bus


class Emisor:
    def __init__(self, host: str, puerto: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (host, puerto)
        self._lock = threading.Lock()

    def enviar(self, msg: dict):
        with self._lock:
            self.sock.sendto((json.dumps(msg, separators=(",", ":")) + "\n").encode(), self.dest)


def hilo_ojo(nombre: str, indice: int, cfg: dict, emisor: Emisor, parar: threading.Event):
    p = cfg["pi"]
    cam = CamaraOjo(indice, p["tamano"], p["fps_ojos"], p["exposicion_us"])
    det = DetectorPupila()
    cam.iniciar()
    n, t_ref = 0, time.monotonic()
    try:
        while not parar.is_set():
            t, img = cam.leer()
            pu = det.detectar(img)
            emisor.enviar({"tipo": "ojo", "ojo": nombre, "t": t, "x": pu.x, "y": pu.y,
                           "mm": pu.diametro_px * p["mm_por_px_pupila"], "conf": pu.conf})
            n += 1
            if time.monotonic() - t_ref >= 5:
                print(f"[{nombre}] {n / (time.monotonic() - t_ref):.0f} fps")
                n, t_ref = 0, time.monotonic()
    finally:
        cam.detener()


def hilo_escena(cfg: dict, emisor: Emisor, parar: threading.Event):
    esc = CamaraEscena(cfg["pi"]["camara_escena"], cfg["pi"]["diccionario_aruco"])
    while not parar.is_set():
        m = esc.marcadores()
        if m:
            emisor.enviar({"tipo": "marcadores", "t": time.monotonic(), "esquinas": m})


def hilo_ambiente(emisor: Emisor, parar: threading.Event):
    bus = abrir_bus()
    luz, imu = VEML7700(bus), MPU6050(bus)
    while not parar.is_set():
        emisor.enviar({"tipo": "ambiente", "t": time.monotonic(), "lux": luz.lux(), **imu.leer()})
        time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", help="IP del computador (por defecto, la de config.json)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = cargar_config(a.config)
    emisor = Emisor(a.host or cfg["pi"]["host_ip"], cfg["red"]["udp_port"])
    salidas = Salidas(cfg["pi"])
    salidas.encender()
    parar = threading.Event()
    hilos = [threading.Thread(target=hilo_ojo, args=("L", 0, cfg, emisor, parar), daemon=True),
             threading.Thread(target=hilo_ojo, args=("R", 1, cfg, emisor, parar), daemon=True),
             threading.Thread(target=hilo_escena, args=(cfg, emisor, parar), daemon=True),
             threading.Thread(target=hilo_ambiente, args=(emisor, parar), daemon=True)]
    for h in hilos:
        h.start()
    salidas.vibrar()
    print("Enviando a", emisor.dest, "- botón o Ctrl+C para terminar")
    try:
        while not salidas.boton.is_pressed:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        parar.set()
        salidas.apagar()


if __name__ == "__main__":
    main()
