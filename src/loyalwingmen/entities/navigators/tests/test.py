from enum import Enum, auto
from typing import Optional
import random

# import pytest

# from .geometry_utils import GeometryUtils
import numpy as np

from loyalwingmen.environments.level4.components.entities_management.offsets_handler import (
    OffsetHandler,
    Offsets,
)

from loyalwingmen.environments.level4.components.entities_management.navigators.loitering_munition_navigator import (
    LoiteringMunitionNavigator,
    State,
)

# ===============================================================================
# Gen Offset
# ===============================================================================


def gen_test_offset(
    # dome_radius: float,
    lw_positions: np.ndarray,
    lm_positions: np.ndarray,
) -> Offsets:
    num_lw = lw_positions.shape[0]
    num_lm = lm_positions.shape[0]
    # Generate unique IDs for LW and LM
    lw_ids = list(range(0, num_lw))  # random.sample(range(1, 11), num_lw)
    lm_ids = list(
        range(num_lw, num_lw + num_lm)
    )  # random.sample(range(11, 21), num_lm)

    # Generate random positions within the dome
    # if lw_positions is None:
    #    lw_positions = np.array([random_position(dome_radius) for _ in range(num_lw)])

    # if lm_positions is None:
    #    lm_positions = np.array([random_position(dome_radius) for _ in range(num_lm)])

    # Generate random velocities for LWs
    lw_velocities = np.array([random_velocity() for _ in range(num_lw)])

    # Compute distances and directions
    distances, directions = compute_distances_and_directions(lw_positions, lm_positions)

    return Offsets(
        distances=distances,
        directions=directions,
        pursuer_ids=lw_ids,
        invader_ids=lm_ids,
        pursuer_positions=lw_positions,
        invader_positions=lm_positions,
        pursuer_velocities=lw_velocities,
    )


def random_position(dome_radius: float) -> np.ndarray:
    while True:
        position = np.random.uniform(-dome_radius, dome_radius, 3)
        if 0 < np.linalg.norm(position) < dome_radius and position[2] > 0:
            return position


def random_velocity() -> np.ndarray:
    return np.random.uniform(-1, 1, 3)


def compute_distances_and_directions(pursuer_positions, invader_positions) -> tuple:
    relative_positions = (
        pursuer_positions[:, np.newaxis, :] - invader_positions[np.newaxis, :, :]
    )
    distances = np.linalg.norm(relative_positions, axis=-1)
    mask = distances[:, :, np.newaxis] != 0

    # Ensure the output array is of a floating-point type
    directions = np.zeros_like(relative_positions, dtype=np.float64)

    np.divide(
        relative_positions, distances[:, :, np.newaxis], out=directions, where=mask
    )
    return distances, directions


# ===============================================================================
# Point between two points
# ===============================================================================


def point_between_two_points(point1, point2, distance):
    # point1 and point2 are np.array([x, y, z])
    # distance is a float
    # returns a point between point1 and point2 that is distance away from point1
    # if distance is greater than the distance between point1 and point2, returns point2
    vector = point2 - point1
    vector = vector / np.linalg.norm(vector)
    return point1 + vector * distance


# Additional imports and utility functions as necessary


def test_navigator_collide_with_wingman():
    lm_position = np.array([0, 0, 10])
    lw_1_position = point_between_two_points(lm_position, np.array([0, 0, 0]), 5)
    lw_2_position = np.array([0, 1, 8])

    # Generate test offsets
    path_to_origin_obstructed_offset_test = gen_test_offset(
        np.array([lw_1_position, lw_2_position]), np.array([lm_position])
    )

    # Initialize OffsetHandler and Navigator
    offset_handler = OffsetHandler(10)
    offset_handler.insert_offset(path_to_origin_obstructed_offset_test)

    navigator = LoiteringMunitionNavigator()

    # Run the test
    command = navigator.get_command(offset_handler, 2)

    # Assertions
    assert (
        command == np.array([0, 1, -2])
    ).all() and navigator.state == State.COLLIDE_WITH_WINGMAN, (
        "Unexpected command output"
    )
    print(command, navigator.state)
    # assert navigator.state == State.COLLIDE_WITH_WINGMAN, "Unexpected navigator state"


def test_navigator_collide_with_building():
    lm_position = np.array([0, 0, 10])
    lw_1_position = np.array(
        [0, 0, -10]
    )  # point_between_two_points(lm_position, np.array([0, 0, 0]), 5)
    lw_2_position = np.array([0, 2, 9])

    # Generate test offsets
    path_to_origin_obstructed_offset_test = gen_test_offset(
        np.array([lw_1_position, lw_2_position]), np.array([lm_position])
    )

    # Initialize OffsetHandler and Navigator
    offset_handler = OffsetHandler(10)
    offset_handler.insert_offset(path_to_origin_obstructed_offset_test)

    navigator = LoiteringMunitionNavigator()

    # Run the test
    command = navigator.get_command(offset_handler, 2)

    # Assertions
    print(command, navigator.state)
    assert (
        command == np.array([0.0, 0.0, -9.0], dtype=np.float32)
    ).all() and navigator.state == State.COLLIDE_WITH_BUILDING, (
        "Unexpected command output"
    )
    print(command, navigator.state)
    # assert navigator.state == State.COLLIDE_WITH_WINGMAN, "Unexpected navigator state"


test_navigator_collide_with_building()
