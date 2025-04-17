from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
from typing import Union, List, Dict, Any

from enum import Enum, auto
from copy import deepcopy


class FlightStateDataType(Enum):
    INERTIAL = auto()
    LIDAR = auto()
    COMBINED = auto()


class FlightStateManager:
    def __init__(self):
        # Initialize a dictionary to store sensor readings
        self.reset()

    def reset(self):
        """Reset the flight state to default values."""
        self.data = {}

    def update_data(self, sensor_data: dict):
        """Update the flight state with new sensor data."""
        self.data.update(sensor_data)

    def get_data(self, key: Union[str, List[str], None] = None) -> Dict[str, Any]:
        """Retrieve specific sensor readings or all data."""

        # If key is None or an empty string, return a deep copy of the entire data.
        if not key:
            return deepcopy(self.data)

        # If key is a string, check if it exists and then return a deep copy of the corresponding value.
        if isinstance(key, str):
            return {
                key: deepcopy(self.data.get(key))
            }  # This will return the value for the key if it exists, or None if it doesn't

        # If key is a list of strings, return a deep copy of the dictionary of corresponding values for existing keys.
        return (
            {k: deepcopy(self.data.get(k)) for k in key}
            if isinstance(key, list)
            else {}
        )

    def get_inertial_data(self) -> Dict[str, Any]:
        """Retrieve data related to the inertial measurement unit (IMU)."""
        inertial_keys = [
            "position",
            "velocity",
            "attitude",
            "quaternion",
            "angular_rate",
        ]
        return self.get_data(inertial_keys) or {key: None for key in inertial_keys}

    def get_lidar_data(self) -> Dict[str, Any]:
        """Retrieve data related to the lidar."""
        lidar_keys = ["lidar"]
        return self.get_data(lidar_keys) or {key: None for key in lidar_keys}

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
