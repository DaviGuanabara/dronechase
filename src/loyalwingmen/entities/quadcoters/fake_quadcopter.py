from .quadcopter import Quadcopter, FlightStateManager, EntityType
from typing import Optional, Dict
import numpy as np


class FakeQuadcopter(Quadcopter):
    """
    Making this class to facilitate the testing of other modules that require the quadcopter class.
    """

    def __init__(
        self,
        id: int,
        quadcopter_type: EntityType = EntityType.QUADCOPTER,
        inertial_data: Dict[str, np.ndarray] = {"position": np.array([0, 0, 0])},
    ):
        self.id = id
        self.current_command: np.ndarray = np.array([0, 0, 0, 0])
        self.quadcopter_type = quadcopter_type

        self.flight_state_manager: FlightStateManager = FlightStateManager()
        self.insert_inertial_data(inertial_data)

        # TODO: search for a better way and insert it into Qaudcopter
        print(
            "[Fake Quadcopter] TODO: formation_position in Quadcopter is not implemented"
        )
        self.formation_position = inertial_data["position"]
        self.gun_available = True

        self.munition = 1

    def insert_inertial_data(self, inertial_data: dict):
        """
        Keys expected:
        "position", "velocity", "attitude", "quaternion", "angular_rate",

        """
        self.flight_state_manager.update_data(inertial_data)

    @property
    def inertial_data(self) -> dict:
        return self.flight_state_manager.get_inertial_data()

    def drive(self, command: np.ndarray):
        self.current_command = command

    # TODO: i should verify this kind of things in the quadcopter class

    @property
    def is_gun_available(self):
        return self.gun_available

    @property
    def is_munition_available(self):
        return self.munition > 0

    def insert_gun_availability(self, availability: bool):
        self.gun_available = availability

    def insert_gun_munition(self, munition: int):
        if munition == 0:
            self.gun_available = False
        self.munition = munition
