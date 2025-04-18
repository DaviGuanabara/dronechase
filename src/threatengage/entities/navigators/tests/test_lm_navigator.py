from enum import Enum, auto
from typing import Optional, Tuple, List
import random

# import pytest

# from .geometry_utils import GeometryUtils
import numpy as np

from threatengage.entities.quadcoters.fake_quadcopter import (
    FakeQuadcopter,
    Quadcopter,
    EntityType,
)
from threatengage.environments.level4.components.entities_management.offsets_handler import (
    OffsetHandler,
    Offsets,
)

from threatengage.entities.navigators.loitering_munition_navigator import (
    KamikazeNavigator,
    CollideWithBuildingState,
    CollideWithWingmanState,
    WaitState,
    State,
)

# from loyalwingmen.entities.quadcoters.fake_quadcopter import Quadcopter, FakeQuadcopter


# ===============================================================================
# Gen Offset
# ===============================================================================


def gen_offset_from_fakes(
    lws: List[FakeQuadcopter], lms: List[FakeQuadcopter]
) -> Offsets:
    lw_ids = [lw.id for lw in lws]
    lm_ids = [lm.id for lm in lms]

    lw_positions = np.array([lw.inertial_data["position"] for lw in lws])
    lm_positions = np.array([lm.inertial_data["position"] for lm in lms])

    lw_velocities = np.array([lw.inertial_data["velocity"] for lw in lws])

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


def fake_drone_creator(ids, positions, velocities, quadcopter_type) -> list:
    return [
        FakeQuadcopter(
            id=id,
            quadcopter_type=quadcopter_type,
            inertial_data={"position": pos, "velocity": vel},
        )
        for id, pos, vel in zip(ids, positions, velocities)
    ]


def create_fake_drones(
    lw_positions: np.ndarray, lm_positions: np.ndarray
) -> Tuple[List[FakeQuadcopter], List[FakeQuadcopter]]:
    """
    Create lists of fake drones for loyal wingmen and loitering munitions.

    Parameters:
    lw_positions (np.ndarray): Positions of loyal wingmen drones.
    lm_positions (np.ndarray): Positions of loitering munitions drones.

    Returns:
    Tuple[List[FakeQuadcopter], List[FakeQuadcopter]]: Lists of fake drones for both categories.
    """
    lw_ids = np.arange(lw_positions.shape[0])
    lm_ids = np.arange(
        lw_positions.shape[0], lw_positions.shape[0] + lm_positions.shape[0]
    )

    lw_velocities = np.random.uniform(-1, 1, (lw_positions.shape[0], 3))
    lm_velocities = np.random.uniform(-1, 1, (lm_positions.shape[0], 3))

    lws = fake_drone_creator(
        lw_ids, lw_positions, lw_velocities, EntityType.LOYALWINGMAN
    )
    lms = fake_drone_creator(
        lm_ids, lm_positions, lm_velocities, EntityType.LOITERINGMUNITION
    )

    return lws, lms


def test_navigator_collide_with_wingman():
    building_position = np.array([0, 0, 1])
    # Make fake Kamikaze
    lm_positions = np.array([[0, 0, 10]])
    lw_positions = np.array([[0, 5, 5], [0, 1, 5]])

    lws, lms = create_fake_drones(lw_positions, lm_positions)

    # Initialize OffsetHandler and Navigator
    offset_handler = OffsetHandler(10)
    offsets = gen_offset_from_fakes(lws, lms)
    offset_handler.insert_offset(offsets)

    navigator = KamikazeNavigator(building_position)
    kamikaze = lms[0]
    navigator.update(kamikaze, offset_handler)

    # Run the test
    assert isinstance(
        navigator.fetch_state(kamikaze), CollideWithWingmanState
    ), f"State {type(navigator.fetch_state(kamikaze)).__name__} is not {CollideWithWingmanState}"


def test_navigator_collide_with_building():
    building_position = np.array([0, 0, 1])
    # Make fake Kamikaze
    lm_positions = np.array([[0, 0, 10]])
    lw_positions = np.array([[0, 5, 5], [0, 10, 5]])

    lws, lms = create_fake_drones(lw_positions, lm_positions)

    # Initialize OffsetHandler and Navigator
    offset_handler = OffsetHandler(10)
    offsets = gen_offset_from_fakes(lws, lms)
    offset_handler.insert_offset(offsets)

    navigator = KamikazeNavigator(building_position)
    kamikaze = lms[0]
    navigator.update(kamikaze, offset_handler)

    # Run the test
    assert isinstance(
        navigator.fetch_state(kamikaze), CollideWithBuildingState
    ), f"State {type(navigator.fetch_state(kamikaze)).__name__} is not {CollideWithBuildingState}"
