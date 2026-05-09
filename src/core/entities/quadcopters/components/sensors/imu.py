import numpy as np
import pybullet as p  # type: ignore
from typing import Dict

from core.entities.quadcopters.components.sensors.interfaces.sensor_interface import Sensor
from PyFlyt.core.drones.quadx import QuadX

"""
Ciclo de pegar o pybullet, que seria atualizar e depois pegar os dados
"""


class InertialMeasurementUnit(Sensor):
    def __init__(self, quadx: QuadX):
        self.quadx = quadx
        self.reset()
        self.update_data()

    def reset(self):
        self.position: np.ndarray = np.zeros(3, dtype=np.float32)
        self.quaternions: np.ndarray = np.zeros(4, dtype=np.float32)
        self.attitude: np.ndarray = np.zeros(3, dtype=np.float32)

        self.velocity: np.ndarray = np.zeros(3, dtype=np.float32)
        self.angular_rate: np.ndarray = np.zeros(3, dtype=np.float32)

    def update_data(self):
        POSITION = 3
        EULER = 1
        VELOCITY = 2
        ANGULAR_RATE = 0

        # Update state from PyFlyt / PyBullet
        self.quadx.update_state()
        state = self.quadx.state

        # Copy directly from simulator (no randomness, no modification)
        self.position[:] = state[POSITION]
        self.attitude[:] = state[EULER]
        self.velocity[:] = state[VELOCITY]
        self.angular_rate[:] = state[ANGULAR_RATE]

        # Compute quaternion safely from Euler angles
        euler = state[EULER]
        quat = p.getQuaternionFromEuler(euler)

        # Safety: avoid zero / invalid quaternion (prevents downstream crashes)
        if quat is None or len(quat) != 4:
            self.quaternions[:] = np.array([0, 0, 0, 1], dtype=np.float32)
        else:
            self.quaternions[:] = np.asarray(quat, dtype=np.float32)

    def read_data(self) -> Dict:
        return {
            "position": self.position,
            "attitude": self.attitude,
            "quaternion": self.quaternions,
            "velocity": self.velocity,
            "angular_rate": self.angular_rate,
        }
