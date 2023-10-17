import numpy as np
import pybullet as p

from .components.base.quadcopter import (
    Quadcopter,
    FlightStateManager,
    EnvironmentParameters,
    QuadcopterType,
    OperationalConstraints,
    QuadcopterSpecs,
    DroneModel,
    CommandType,
)

from enum import Enum


class LoiteringMunitionBehavior(Enum):
    FROZEN = 1
    STRAIGHT_LINE = 2
    CIRCLE = 3


class LoiteringMunition(Quadcopter):
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
        use_direct_velocity = True
        super().__init__(
            id,
            model,
            droneSpecs,
            operationalConstraints,
            environment_parameters,
            QuadcopterType.LOITERINGMUNITION,
            quadcopter_name,
            command_type,
        )
        self.quadcopter_name: str = quadcopter_name

        self.set_behavior(LoiteringMunitionBehavior.FROZEN)

    def _frozen(self, flight_state: FlightStateManager):
        return np.array([0, 0, 0, 0])

    def _straight_line(
        self, flight_state: FlightStateManager, direction, intensity
    ) -> np.ndarray:
        return np.array([*direction, intensity])

    def _circle(
        self,
        flight_state: FlightStateManager,
        radius=1,
        origin: np.ndarray = np.array([0, 0, 0]),
        velocity: np.ndarray = np.array([0, 0, 0]),
    ):
        timeset_period = self.environment_parameters.timestep
        aggregate_physics_steps = self.environment_parameters.aggregate_physics_steps

        inertial_data = flight_state.get_data("inertial")
        quad_position: np.ndarray = inertial_data.get("position", np.zeros(3))
        # quad_position = flight_state.get_data("position")
        # quad_position = quad_position if quad_position is not None else np.zeros(3)
        quad_velocity = velocity

        # Calculate center of circle
        distance_to_origin = np.linalg.norm(quad_position)
        distance_to_origin = distance_to_origin if distance_to_origin != 0 else 1

        center_direction = (origin - 1 * quad_position) / distance_to_origin
        center = quad_position + center_direction * radius

        # Calculate Aceleration:
        distance_to_center = np.linalg.norm(center - quad_position)
        distance_to_center = distance_to_center if distance_to_center != 0 else 1

        quad_velocity_intensity = np.linalg.norm(quad_velocity)
        aceleration_direction = (center - quad_position) / distance_to_center
        aceleration_intensity = quad_velocity_intensity**2 / radius
        aceleration = aceleration_direction * aceleration_intensity

        # Calculate new velocity
        period = timeset_period * aggregate_physics_steps
        new_velocity = quad_velocity + aceleration * period

        # Calculate New Command
        new_intensity = np.linalg.norm(new_velocity)
        new_intensity = new_intensity if new_intensity != 0 else 1
        new_direction = new_velocity / np.linalg.norm(new_intensity)

        return np.append(new_direction, new_intensity)

    def set_behavior(self, behavior: LoiteringMunitionBehavior):
        velocity_vector = self._random_direction_and_intensity()
        direction = velocity_vector[:3]
        intensity = velocity_vector[3]

        if behavior == LoiteringMunitionBehavior.FROZEN:
            self.behavior_function = lambda flight_state: self._frozen(flight_state)

        elif behavior == LoiteringMunitionBehavior.STRAIGHT_LINE:
            self.behavior_function = lambda flight_state: self._straight_line(
                flight_state, direction, intensity
            )

        elif behavior == LoiteringMunitionBehavior.CIRCLE:
            radius = 1
            origin = np.array([0, 0, 0])

            self.behavior_function = lambda flight_state: self._circle(
                flight_state, radius, origin, velocity=intensity * direction
            )

        self.current_behavior = behavior

    def _random_direction_and_intensity(self) -> np.ndarray:
        direction = np.random.uniform(-1, 1, 3)
        intensity = np.random.uniform(0.05, 0.5)

        return np.append(direction, intensity)

    def drive_via_behavior(self):
        flight_state_manager = self.flight_state_manager
        command = self.behavior_function(flight_state_manager)

        self.drive(command)
        # self.show_name()
