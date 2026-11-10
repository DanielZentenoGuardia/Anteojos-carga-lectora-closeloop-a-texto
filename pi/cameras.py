"""Cámaras: 2 x OV9281 (ojos, CSI0/CSI1) y 1 x USB (escena, marcadores ArUco).

IMPORTANTE: verificar en el dispositivo. El modo 640x400 del OV9281 en el Pi 5 se reporta
a ~310 fps con el driver estándar (rpicam-hello --list-cameras); hay reportes de módulos
genéricos no detectados. Probar una cámara antes de comprar la segunda.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:  # permite importar el módulo fuera del Pi
    Picamera2 = None


class CamaraOjo:
    def __init__(self, indice: int, tamano=(640, 400), fps=250, exposicion_us=2000):
        if Picamera2 is None:
            raise RuntimeError("picamera2 no disponible: ejecutar en el Raspberry Pi")
        self.cam = Picamera2(indice)
        periodo = int(1e6 / fps)
        conf = self.cam.create_video_configuration(
            main={"size": tuple(tamano), "format": "YUV420"},
            controls={"FrameDurationLimits": (periodo, periodo), "ExposureTime": exposicion_us,
                      "AnalogueGain": 4.0, "AeEnable": False},
            buffer_count=6)
        self.cam.configure(conf)
        self.alto = tamano[1]

    def iniciar(self):
        self.cam.start()

    def leer(self) -> tuple[float, np.ndarray]:
        """Devuelve (t en s, imagen en grises). t = marca de tiempo del sensor si existe."""
        req = self.cam.capture_request()
        try:
            img = req.make_array("main")[: self.alto]      # plano Y = escala de grises
            meta = req.get_metadata()
            t = meta.get("SensorTimestamp", time.monotonic_ns()) / 1e9
        finally:
            req.release()
        return t, img

    def detener(self):
        self.cam.stop()


class CamaraEscena:
    """Cámara USB (UVC) que ve la pantalla; detecta los 4 marcadores ArUco de las esquinas."""

    def __init__(self, indice=0, diccionario="DICT_4X4_50"):
        self.cap = cv2.VideoCapture(indice)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, diccionario))
        if hasattr(cv2.aruco, "ArucoDetector"):            # OpenCV >= 4.7
            self._det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
            self._detectar = self._det.detectMarkers
        else:                                              # OpenCV < 4.7
            params = cv2.aruco.DetectorParameters_create()
            self._detectar = lambda img: cv2.aruco.detectMarkers(img, dic, parameters=params)

    def marcadores(self) -> dict[str, list] | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        esquinas, ids, _ = self._detectar(gris)
        if ids is None:
            return {}
        return {str(int(i)): c.reshape(4, 2).tolist() for i, c in zip(ids.flatten(), esquinas)}
