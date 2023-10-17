import numpy as np
import pybullet as p

from .components.base.quadcopter import (
    Quadcopter,
    QuadcopterType,
    FlightStateManager,
    QuadcopterSpecs,
    OperationalConstraints,
    EnvironmentParameters,
    DroneModel,
    CommandType,
)
from dataclasses import dataclass, field, fields


class LoyalWingman(Quadcopter):
    def __init__(
        self,
        id: int,
        model: DroneModel,
        droneSpecs: QuadcopterSpecs,
        operationalConstraints: OperationalConstraints,
        environment_parameters: EnvironmentParameters,
        quadcopter_name: str,
        command_type: CommandType,
    ):
        super().__init__(
            id,
            model,
            droneSpecs,
            operationalConstraints,
            environment_parameters,
            QuadcopterType.LOYALWINGMAN,
            quadcopter_name,
            command_type,
        )
        self.quadcopter_name = quadcopter_name
