"""Sensores I2C (bus 1: GPIO2 = SDA, GPIO3 = SCL) y salidas GPIO.

VEML7700 (luz ambiente, dirección 0x10) y MPU-6050 (IMU, dirección 0x68).
"""
from __future__ import annotations

import struct

try:
    from smbus2 import SMBus
except ImportError:
    SMBus = None


class VEML7700:
    DIR = 0x10

    def __init__(self, bus: "SMBus"):
        self.bus = bus
        # Registro 0x00 (ALS_CONF): ganancia x1, integración 100 ms, encendido
        self.bus.write_i2c_block_data(self.DIR, 0x00, [0x00, 0x00])

    def lux(self) -> float:
        lo, hi = self.bus.read_i2c_block_data(self.DIR, 0x04, 2)
        return ((hi << 8) | lo) * 0.0576   # resolución con ganancia x1 e integración 100 ms


class MPU6050:
    DIR = 0x68

    def __init__(self, bus: "SMBus"):
        self.bus = bus
        self.bus.write_byte_data(self.DIR, 0x6B, 0x00)   # salir de reposo

    def leer(self) -> dict:
        d = self.bus.read_i2c_block_data(self.DIR, 0x3B, 14)
        ax, ay, az, _temp, gx, gy, gz = struct.unpack(">7h", bytes(d))
        return {"acc_g": [ax / 16384, ay / 16384, az / 16384],      # rango ±2 g
                "gir_dps": [gx / 131, gy / 131, gz / 131]}          # rango ±250 °/s


class Salidas:
    """LEDs IR (PWM), LED de privacidad, motor háptico y botón."""

    def __init__(self, cfg_pi: dict):
        from gpiozero import LED, Button, PWMLED, OutputDevice
        self.ir = PWMLED(cfg_pi["gpio_leds_ir"], frequency=1000)
        self.privacidad = LED(cfg_pi["gpio_led_privacidad"])
        self.haptico = OutputDevice(cfg_pi["gpio_haptico"])
        self.boton = Button(cfg_pi["gpio_boton"], pull_up=True)
        self.ciclo = cfg_pi["ciclo_leds_ir"]

    def encender(self):
        # SEGURIDAD: medir la irradiancia en el ojo y mantenerla bajo IEC 62471 antes de usar.
        self.ir.value = self.ciclo
        self.privacidad.on()

    def apagar(self):
        self.ir.off()
        self.privacidad.off()

    def vibrar(self, segundos=0.2):
        self.haptico.blink(on_time=segundos, n=1, background=True)


def abrir_bus():
    if SMBus is None:
        raise RuntimeError("smbus2 no disponible: ejecutar en el Raspberry Pi")
    return SMBus(1)
