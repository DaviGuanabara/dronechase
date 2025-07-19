from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from typing import Dict
from core.entities.quadcopters.components.sensors.interfaces.sensor_interface import Sensor
from enum import Enum, auto
import numpy as np
import random

class BaseLidar(Sensor):
    
    def buffer_lidar_data(self, message: Dict, publisher_id: int):
        pass

    def buffer_inertial_data(self, message: Dict, publisher_id: int):
        pass

    def buffer_step_broadcast(self, message: Dict, publisher_id: int):
        pass


class LiDARBufferManager:
    def __init__(self, current_step=0, max_buffer_size: int = 10):

        self.buffer: Dict[int, List[Optional[Dict]]] = {}

        self.max_buffer_size = max_buffer_size
        #self.buffer: List[Optional[Dict]] = [None] * max_buffer_size

        self.current_step = current_step

    def update_step(self, step: int) -> None:
        self.current_step = step


    def _add_publisher(self, publisher_id: int) -> None:
        if publisher_id not in self.buffer:
            self.buffer[publisher_id] = [None] * self.max_buffer_size

    def _buffer_latest_only(self, message: Dict, publisher_id: int) -> None:
        self.buffer[publisher_id][0] = message


    def _buffer_temporal_message(self, message: Dict, publisher_id: int, step: int) -> None:
        delta_step = self.current_step - step
        if delta_step >= self.max_buffer_size:
            return  #avoid overflow
        self.buffer[publisher_id][delta_step] = message



    def buffer_message(self, message: Dict, publisher_id: int, step: int = -1) -> None:
        self._add_publisher(publisher_id)

        if step < 0:
            self._buffer_latest_only(message, publisher_id)
        elif step > self.current_step:
            raise ValueError(
                "Step must be less than or equal to the current step.")
        else:
            self._buffer_temporal_message(message, publisher_id, step)


    
    def clear_buffer(self) -> None:
        self.buffer = {}
        

    
    def clear_buffer_data(self, publisher_id: int) -> None:
        self.buffer.pop(publisher_id, None)

        
    def get_data(self) -> List[Dict]:
        return [entry for sublist in self.buffer.values() for entry in sublist if entry is not None]
    
    # to keep LIDAR compatible with the old interface

    def get_all_data(self) -> Dict[int, Dict]:
        return {
            k: next(item for item in v if item is not None)
            for k, v in self.buffer.items()
            if any(item is not None for item in v)
        }


    def sample_data(self, n: int) -> List[Dict]:
        valid_data = self.get_data()
        return random.sample(valid_data, min(n, len(valid_data)))



class Channels(Enum):
    DISTANCE_CHANNEL = 0
    FLAG_CHANNEL = 1


class CoordinateConverter:
    @staticmethod
    def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
        radius, theta, phi = spherical
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.array([x, y, z])

    @staticmethod
    def cartesian_to_spherical(cartesian: np.ndarray) -> np.ndarray:
        x, y, z = cartesian
        radius = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / radius)
        phi = np.arctan2(y, x)
        return np.array([radius, theta, phi])
