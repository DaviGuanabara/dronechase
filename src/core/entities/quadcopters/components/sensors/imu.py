import numpy as np
from typing import Tuple, List, Dict
import pybullet as p

from core.entities.quadcopters.components.sensors.sensor_interface import Sensor
from PyFlyt.core.drones.quadx import QuadX

"""
Ciclo de pegar o pybullet, que seria atualizar e depois pegar os dados
"""


class InertialMeasurementUnit(Sensor):
    def __init__(self, quadx:QuadX):
        self.quadx = quadx
        self.reset()
        self.update_data()

    def reset(self):
        self.position: np.ndarray = np.zeros(3)
        self.quaternions: np.ndarray = np.zeros(4)
        self.attitude: np.ndarray = np.zeros(3)

        self.velocity: np.ndarray = np.zeros(3)
        self.angular_rate: np.ndarray = np.zeros(3)

    def update_data(self):
        POSITION = 3
        EULER = 1
        VELOCITY = 2
        ANGULAR_RATE = 0
        
        self.quadx.update_state()
        state = self.quadx.state

        self.position = np.array(state[POSITION])
        self.attitude = np.array(state[EULER])
        self.quaternions = np.array(p.getQuaternionFromEuler(state[EULER]))
        
        self.velocity = np.array(state[VELOCITY])
        self.angular_rate = np.array(state[ANGULAR_RATE])


    def read_data(
        self,
    ) -> Dict:
        position = self.position
        attitude = self.attitude
        quaternion = self.quaternions
        velocity = self.velocity
        angular_rate = self.angular_rate

        return {
            "position": position,
            "attitude": attitude,
            "quaternion": quaternion,
            "velocity": velocity,
            "angular_rate": angular_rate,
        }
