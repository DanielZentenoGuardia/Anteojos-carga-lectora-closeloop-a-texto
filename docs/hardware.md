# Hardware

Precios de referencia en USD (noviembre de 2026). Varían a diario; no incluyen envío ni IVA.

## Lista de materiales

| Bloque | Componente | Costo mínimo | Versión estándar |
|---|---|---|---|
| Montura | 2 × cámara OV9281 global shutter, sin filtro IR (640×400 a ~300 fps en Pi 5) | $26–31 c/u | $26–44 c/u |
| | 16 × LED IR 850 nm SMD + MOSFET (PWM) | ≈ $2 | $12–16 |
| | 2 × filtro pasabanda 850 nm (8–10 mm) | $2–10 | ≈ $24 |
| | Cámara de escena USB (solo detecta los marcadores ArUco) | $5–17 | $35–47 |
| | VEML7700 (luz ambiente, I2C) | $0,5–1 | $5–7 |
| | MPU-6050 (IMU, I2C) | $0,3–3 | $2–8 |
| | LED de privacidad + botón + motor de vibración 1027 | ≈ $1 | ≈ $3 |
| | Montura impresa en 3D | $9–16 | ≈ $10 |
| Bolsillo | Raspberry Pi 5 de 1 GB | $45 | $45 |
| | Disipador activo | ≈ $1 | $5–7 |
| | microSD 64 GB | ≈ $13 | $16–32 |
| | Power bank 10 000 mAh USB-C PD (5 V / 3 A estables) | $8–12 | $18–46 |
| | Cables de cámara 22→15 pines (×2) | $2–14 | $5–10 |
| **Total** | | **≈ $150–200** | **≈ $240–350** |

No incluido: PCB para los LEDs (rígida pequeña, pocos dólares; flexible, bastante más).

## Conexiones del Raspberry Pi 5

| Señal | Componente | Pin físico (GPIO) |
|---|---|---|
| CAM0 / CAM1 (22 pines) | OV9281 ojo izquierdo / derecho | conectores CSI |
| USB 3 | Cámara de escena | puerto USB |
| 3V3 | VEML7700, MPU-6050 | 1 |
| GND | sensores, MOSFET, LED, botón | 6 |
| SDA (GPIO2) / SCL (GPIO3) | bus I2C (0x10 luz, 0x68 IMU) | 3 / 5 |
| GPIO18 (PWM0) | compuerta del MOSFET de los LEDs IR | 12 |
| GPIO23 | transistor del motor háptico | 16 |
| GPIO24 | LED de privacidad (con resistencia de 330 Ω) | 18 |
| GPIO25 | botón (a GND, pull-up interno) | 22 |

## Seguridad ocular
Los LEDs IR deben operar con corriente limitada y pulsados. Antes de cualquier uso con
personas, mida la irradiancia en la posición del ojo y verifique que queda bajo los límites
de exposición de IEC 62471.

## Riesgos conocidos
- Hay reportes de módulos OV9281 genéricos que el Pi 5 no detecta: pruebe una unidad primero.
- Procesar dos cámaras a ~300 fps exige recortar la imagen alrededor del ojo (ROI), como hace
  `pi/pupil.py`.
- El SFH 4050 (OSRAM) emite a 860 nm: si lo usa, confirme que cae dentro de la banda del filtro.
