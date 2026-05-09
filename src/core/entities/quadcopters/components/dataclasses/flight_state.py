from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
from typing import Union, List, Dict, Any

from enum import Enum, auto


class FlightStateDataType(Enum):
    INERTIAL = auto()
    LIDAR = auto()
    COMBINED = auto()


INERTIAL_KEYS = ("position", "velocity", "attitude",
                 "quaternion", "angular_rate")
LIDAR_KEYS = ("lidar", "sphere", "features",
              "stacked_spheres", "validity_mask")


class FlightStateManager:
    def __init__(self):
        # Initialize a dictionary to store sensor readings
        self.reset()

    def reset(self):
        """Reset the flight state to default values."""
        self.data = {}

    def update_data(self, sensor_data: dict):
        """Update the flight state with new sensor data."""
        # Avoid dict resizing patterns by assigning explicitly
        for k, v in sensor_data.items():
            self.data[k] = v

    def get_data(self, key: Union[str, List[str], None] = None, copy: bool = False) -> Dict[str, Any]:
        """Retrieve specific sensor readings or all data.

        By default returns references (no copy) for performance.
        Set copy=True to return shallow copies of arrays.
        """
        if not key:
            if copy:
                return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in self.data.items()}
            return self.data

        if isinstance(key, str):
            val = self.data.get(key)
            if copy and val is not None and hasattr(val, "copy"):
                val = val.copy()
            return {key: val}

        if isinstance(key, list):
            if copy:
                result = {}
                for k in key:
                    val = self.data.get(k)
                    if val is not None and hasattr(val, "copy"):
                        val = val.copy()
                    result[k] = val
                return result
            return {k: self.data.get(k) for k in key}

        return {}

    def get_inertial_data(self) -> Dict[str, Any]:
        """Retrieve data related to the inertial measurement unit (IMU)."""
        data = {k: self.data.get(k) for k in INERTIAL_KEYS}
        return data if any(v is not None for v in data.values()) else {key: None for key in INERTIAL_KEYS}

    def get_lidar_data(self) -> Dict[str, Any]:
        """Retrieve data related to the lidar."""
        data = {k: self.data.get(k) for k in LIDAR_KEYS}
        return data if any(v is not None for v in data.values()) else {key: None for key in LIDAR_KEYS}

    def get_data_by_type(
        self, flight_state_data_type: FlightStateDataType
    ) -> Dict[str, Any]:
        """Retrieve data related to the lidar."""
        if flight_state_data_type == FlightStateDataType.INERTIAL:
            return self.get_inertial_data()
        elif flight_state_data_type == FlightStateDataType.LIDAR:
            return self.get_lidar_data()
        elif flight_state_data_type == FlightStateDataType.COMBINED:
            return self.get_data()
        else:
            return {}

    # Additional states can be added as needed, such as:
    # - Accelerations (linear and angular)
    # - Control surface deflections (aileron, elevator, rudder, etc.)
    # - Engine settings or thrust levels
    # - etc.
