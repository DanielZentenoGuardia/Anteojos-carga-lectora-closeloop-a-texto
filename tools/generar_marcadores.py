"""Genera los 4 marcadores ArUco (DICT_4X4_50, ids 0-3) que la app muestra en las esquinas.

Requiere OpenCV:  pip install opencv-python
Uso:              python tools/generar_marcadores.py
"""
from pathlib import Path

import cv2

DESTINO = Path(__file__).resolve().parents[1] / "host" / "web" / "marcadores"


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i in range(4):
        if hasattr(cv2.aruco, "generateImageMarker"):
            img = cv2.aruco.generateImageMarker(dic, i, 240)
        else:
            img = cv2.aruco.drawMarker(dic, i, 240)
        cv2.imwrite(str(DESTINO / f"aruco_{i}.png"), img)
    print("Marcadores guardados en", DESTINO)


if __name__ == "__main__":
    main()
