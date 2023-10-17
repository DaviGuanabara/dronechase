from typing import Dict
import numpy as np
from ...quadcoters.components.dataclasses.operational_constraints import (
    OperationalConstraints,
)
from .environment_parameters import EnvironmentParameters


# TODO: CHANGE THE NAME NORMALIZE_FLIGHT_STATE TO NORMALIZE_INERTIAL_DATA
def normalize_inertial_data(
    flight_state: Dict,
    operational_constraints: OperationalConstraints,
    dome_radius: float,
) -> Dict:
    """Normalize the flight state of the loyal wingman."""
    position = flight_state.get("position", np.zeros(3))
    velocity = flight_state.get("velocity", np.zeros(3))
    attitude = flight_state.get("attitude", np.zeros(3))
    angular_rate = flight_state.get("angular_rate", np.zeros(3))

    acceleration = flight_state.get("acceleration", np.zeros(3))
    angular_acceleration = flight_state.get("angular_acceleration", np.zeros(3))

    norm_position = normalize_position(position, dome_radius)
    norm_attitude = normalize_attitude(attitude)

    norm_velocity = normalize_velocity(velocity, operational_constraints.speed_limit)
    norm_angular_rate = normalize_angular_rate(
        angular_rate, operational_constraints.angular_speed_limit
    )

    norm_acceleration = normalize_acceleration(
        acceleration, operational_constraints.acceleration_limit
    )
    norm_angular_acceleration = normalize_angular_acceleration(
        angular_acceleration, operational_constraints.angular_acceleration_limit
    )

    return {
        "position": norm_position,
        "velocity": norm_velocity,
        "attitude": norm_attitude,
        "angular_rate": norm_angular_rate,
        "acceleration": norm_acceleration,
        "angular_acceleration": norm_angular_acceleration,
    }


def normalize_acceleration(acceleration: np.ndarray, max_acceleration: float):
    if max_acceleration == 0:
        print("max_acceleration is 0")
        return acceleration

    acceleration /= max_acceleration
    return np.clip(
        acceleration,
        -1,
        1,
        dtype=np.float32,
    )


def normalize_angular_acceleration(
    angular_acceleration: np.ndarray, max_angular_acceleration: float
):
    if max_angular_acceleration == 0:
        print("max_angular_acceleration is 0")
        return angular_acceleration

    angular_acceleration /= max_angular_acceleration
    return np.clip(
        angular_acceleration,
        -1,
        1,
        dtype=np.float32,
    )


def normalize_position(position: np.ndarray, dome_radius: float):
    if dome_radius == 0:
        print("dome_radius is 0")
        return position

    position /= dome_radius
    return np.clip(
        position,
        -1,
        1,
        dtype=np.float32,
    )


def normalize_velocity(velocity: np.ndarray, max_velocity: float):
    if max_velocity == 0:
        print("max_velocity is 0")
        return velocity

    velocity /= max_velocity
    return np.clip(
        velocity,
        -1,
        1,
        dtype=np.float32,
    )


def normalize_attitude(attitude: np.ndarray, max_attitude: float = np.pi):
    if max_attitude == 0:
        print("max_attitude is 0")
        return attitude

    attitude /= max_attitude
    return np.clip(
        attitude,
        -1,
        1,
        dtype=np.float32,
    )


def normalize_angular_rate(angular_rate: np.ndarray, max_angular_rate: float):
    if max_angular_rate == 0:
        print("max_angular_rate is 0")
        return angular_rate

    angular_rate /= max_angular_rate
    return np.clip(
        angular_rate,
        -1,
        1,
        dtype=np.float32,
    )
