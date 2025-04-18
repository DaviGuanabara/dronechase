from typing import Dict
import numpy as np


# TODO: CHANGE THE NAME NORMALIZE_FLIGHT_STATE TO NORMALIZE_INERTIAL_DATA
def normalize_inertial_data(
    inertial_data: Dict,
    max_speed: float,
    dome_radius: float,
) -> Dict:
    # (2 * pi rad / rev)
    angular_max_speed = 2 * np.pi

    position = inertial_data.get("position", np.zeros(3))
    velocity = inertial_data.get("velocity", np.zeros(3))
    attitude = inertial_data.get("attitude", np.zeros(3))
    angular_rate = inertial_data.get("angular_rate", np.zeros(3))

    norm_position = normalize_position(position, dome_radius)
    norm_attitude = normalize_attitude(attitude)

    norm_velocity = normalize_velocity(velocity, max_speed)
    norm_angular_rate = normalize_angular_rate(angular_rate, angular_max_speed)

    return {
        "position": norm_position,
        "velocity": norm_velocity,
        "attitude": norm_attitude,
        "angular_rate": norm_angular_rate,
    }


def normalize_acceleration(acceleration: np.ndarray, max_acceleration: float):
    if max_acceleration == 0:
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
        return angular_rate

    angular_rate /= max_angular_rate
    return np.clip(
        angular_rate,
        -1,
        1,
        dtype=np.float32,
    )
