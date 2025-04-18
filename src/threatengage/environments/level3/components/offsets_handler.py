import numpy as np
from dataclasses import dataclass
from .quadcopter_manager import QuadcopterManager
from threatengage.entities.quadcoters.quadcopter import Quadcopter
from typing import Dict, List


@dataclass(frozen=True)
class Offsets:
    distances: np.ndarray
    directions: np.ndarray
    pursuer_ids: list
    invader_ids: list
    pursuer_positions: np.ndarray  # Array of pursuer positions (pursuers x 3)
    invader_positions: np.ndarray  # Array of invader positions (invaders x 3)
    pursuer_velocities: np.ndarray


class OffsetHandler:
    """Class to handle offsets for the environment."""

    def __init__(
        self,
        quadcopter_manager: QuadcopterManager,
        dome_radius: float,
    ):
        """Initialize the offset handler."""
        self.quadcopter_manager = quadcopter_manager
        self.dome_radius = dome_radius

        self.current_moving_average = MovingAverage(10)
        self.last_moving_average = MovingAverage(10)

        # self.current_offsets: Offsets = self.calculate_invader_offsets_from_pursuers()
        # self.last_offsets: Offsets = self.filter_last_offsets(self.current_offsets, self.last_offsets)

    # ===========================================================================
    # Flow of the offsets
    # ===========================================================================

    def on_episode_start(self):
        # print("Offserts handler = on_episode_start")
        self.current_offsets: Offsets = self.calculate_invader_offsets_from_pursuers()
        self.last_offsets: Offsets = self.current_offsets

        self.current_moving_average = MovingAverage(10)
        self.last_moving_average = MovingAverage(10)
        # print("offsets handler = started: ")

    def on_middle_step(self):
        self.update_offsets()

    def on_end_step(self):
        self.update_last_offsets()

    # ===========================================================================
    # Update offsets
    # ===========================================================================

    def update_offsets(self):
        # last offsets has to be filtered, otherwise we get wrong distances
        # this filter aims to remove the invaders that are not present in the current offsets

        self.current_offsets: Offsets = self.calculate_invader_offsets_from_pursuers()
        self.last_offsets: Offsets = self.filter_last_offsets(
            self.current_offsets, self.last_offsets
        )

    def update_last_offsets(self):
        self.last_offsets: Offsets = self.current_offsets

    def calculate_invader_offsets_from_pursuers(self):
        # Get invaders and pursuers positions as arrays
        invaders: "list[Quadcopter]" = self.quadcopter_manager.get_armed_invaders()
        pursuers: "list[Quadcopter]" = self.quadcopter_manager.get_armed_pursuers()

        invaders_positions, invader_ids = self.compute_positions_and_ids(invaders)
        pursuers_positions, pursuer_ids = self.compute_positions_and_ids(pursuers)
        pursuers_velocities = np.array(
            [pursuer.inertial_data["velocity"] for pursuer in pursuers]
        )  # Shape (n_pursuers, 3)

        distances, directions = self.compute_distances_and_directions(
            pursuers_positions, invaders_positions
        )

        return Offsets(
            distances,
            directions,
            pursuer_ids,
            invader_ids,
            pursuers_positions,
            invaders_positions,
            pursuers_velocities,
        )

    def filter_last_offsets(
        self, current_offsets: Offsets, last_offsets: Offsets
    ) -> Offsets:
        # Filter last_offsets to keep only invaders present in current_offsets
        common_invader_ids = set(last_offsets.invader_ids) & set(
            current_offsets.invader_ids
        )
        indices_to_keep = [
            last_offsets.invader_ids.index(invader_id)
            for invader_id in common_invader_ids
        ]

        distances = last_offsets.distances[:, indices_to_keep]
        directions = last_offsets.directions[:, indices_to_keep]
        pursuer_ids = [
            last_offsets.pursuer_ids[0]
        ]  # since pursuer_ids seems to have always one element
        invader_ids = [last_offsets.invader_ids[i] for i in indices_to_keep]
        pursuer_positions = (
            last_offsets.pursuer_positions
        )  # seems to be a 2D array with one row, so keep as is
        invader_positions = last_offsets.invader_positions[indices_to_keep]
        pursuer_velocities = (
            last_offsets.pursuer_velocities
        )  # seems to be a 2D array with one row, so keep as is

        return Offsets(
            distances,
            directions,
            pursuer_ids,
            invader_ids,
            pursuer_positions,
            invader_positions,
            pursuer_velocities,
        )

    def compute_positions_and_ids(self, quadcopters):
        positions = np.array(
            [quadcopter.inertial_data["position"] for quadcopter in quadcopters]
        )
        ids = [quadcopter.id for quadcopter in quadcopters]
        return positions, ids

    """
    def compute_distances_and_directions(self, pursuers_positions, invaders_positions):
        
        relative_positions = pursuers_positions[:, np.newaxis, :] - invaders_positions[np.newaxis, :, :]
        distances = np.linalg.norm(relative_positions, axis=-1)
        mask=distances[:, :, np.newaxis] != 0
        directions = np.divide(relative_positions, distances[:, :, np.newaxis], out=np.zeros_like(relative_positions), where=mask)
        return distances, directions
    """

    def compute_distances_and_directions(self, pursuers_positions, invaders_positions):
        relative_positions = (
            pursuers_positions[:, np.newaxis, :] - invaders_positions[np.newaxis, :, :]
        )

        distances = np.linalg.norm(relative_positions, axis=-1)
        mask = distances[:, :, np.newaxis] != 0
        directions = np.divide(
            relative_positions,
            distances[:, :, np.newaxis],
            out=np.zeros_like(relative_positions),
            where=mask,
        )

        return distances, directions

    # ===========================================================================
    # Helpers
    # ===========================================================================

    """
    This code were developed with an assumption that there is only one pursuer. But, it is not the case anymore.
    It will be two or more pursuers. The first one is the RL Agent, and the others are the support pursuers, running behavior tree. 
    As the time is short, I will make only small changes, enough to run the new requirements.
    """

    def compute_closest_rl_agent_distance(self, distances) -> np.ndarray:
        # print("distances:", distances[0])
        return np.min(distances[0], axis=0)

    @property
    def current_pursuer_velocities(self):
        return self.current_offsets.pursuer_velocities

    @property
    def current_closest_pursuer_to_invader_distance(self) -> float:
        """se tiver mais de um pursuer, essa lógica quebra"""
        # print(
        #    "closest pursuers distance:",
        #    self.compute_closest_rl_agent_distance(self.current_offsets.distances),
        # )
        return np.sum(
            self.compute_closest_rl_agent_distance(self.current_offsets.distances)
        )

    @property
    def last_closest_pursuer_to_invader_distance(self) -> int:
        """se tiver mais de um pursuer, essa lógica quebra"""
        return np.sum(
            self.compute_closest_rl_agent_distance(self.last_offsets.distances)
        )

    # def identify_captured_invaders(self) -> list:
    #    capture_mask = self.current_offsets.distances < self.CAPTURE_DISTANCE
    #    is_captured_by_any_pursuer = np.any(capture_mask, axis=0)
    #    return [
    #        invader_id
    #        for idx, invader_id in enumerate(self.current_offsets.invader_ids)
    #        if is_captured_by_any_pursuer[idx]
    #    ]

    def identify_invaders_in_range(
        self, range_threshold: float
    ) -> Dict[int, List[int]]:
        """
        Identify invaders within a specified range of each pursuer and return a dictionary.
        Each key is a pursuer ID, and the value is a list of invader IDs within range, sorted by distance.
        """
        range_mask = self.current_offsets.distances < range_threshold
        invaders_in_range = {}

        for pursuer_index, pursuer_id in enumerate(self.current_offsets.pursuer_ids):
            invaders_indices = np.where(range_mask[pursuer_index])[0]
            distances = self.current_offsets.distances[pursuer_index,
                                                    invaders_indices]

            if ids := [
                self.current_offsets.invader_ids[i]
                for i, _ in sorted(zip(invaders_indices, distances), key=lambda x: x[1])
            ]:
                invaders_in_range[pursuer_id] = ids

        return invaders_in_range


    def identify_invaders_in_origin(self, threshold_range=0.2) -> list:
        invaders_in_origin_bool_array = (
            np.linalg.norm(self.current_offsets.invader_positions, axis=1)
            < threshold_range
        )
        return list(
            np.array(self.current_offsets.invader_ids)[invaders_in_origin_bool_array]
        )

    def identify_pursuer_outside_dome(self) -> list:
        pursuers_outside_dome_bool_array = (
            np.linalg.norm(self.current_offsets.pursuer_positions, axis=1)
            > self.dome_radius
        )
        return list(
            np.array(self.current_offsets.pursuer_ids)[pursuers_outside_dome_bool_array]
        )

    def identify_invader_outside_dome(self) -> list:
        invaders_outside_dome_bool_array = (
            np.linalg.norm(self.current_offsets.invader_positions, axis=1)
            > self.dome_radius
        )
        return list(
            np.array(self.current_offsets.invader_ids)[invaders_outside_dome_bool_array]
        )

    def identify_pursuer_touched_ground(self) -> list:
        """
        Identify pursuers that have touched the ground.
        Returns a list of pursuer IDs that are at ground level (x=0).
        """
        pursuer_at_ground_bool_array = (
            self.current_offsets.pursuer_positions[:, 2] < 0.01
        )
        return list(
            np.array(self.current_offsets.pursuer_ids)[pursuer_at_ground_bool_array]
        )

    def identify_invader_touched_ground(self) -> list:
        """
        Identify invaders that have touched the ground.
        Returns a list of invader IDs that are at ground level (x=0).
        """
        invader_at_ground_bool_array = (
            self.current_offsets.invader_positions[:, 2] < 0.01
        )
        return list(
            np.array(self.current_offsets.invader_ids)[invader_at_ground_bool_array]
        )


class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []

    def add_value(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get_average(self):
        return sum(self.values) / len(self.values) if self.values else 0
