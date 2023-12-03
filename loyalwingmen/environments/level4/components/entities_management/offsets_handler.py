import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

from .entities_manager import Quadcopter


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
        dome_radius: float,
        entities_manager=None,
    ):
        """Initialize the offset handler."""
        self.entities_manager = entities_manager
        self.dome_radius = dome_radius

        # self.current_offsets: Offsets = self.calculate_invader_offsets_from_pursuers()
        # self.last_offsets: Offsets = self.filter_last_offsets(self.current_offsets, self.last_offsets)

    # ===========================================================================
    # Flow of the offsets
    # ===========================================================================

    def on_episode_start(self):
        self.current_offsets: Offsets = self.calculate_invader_offsets_from_pursuers()
        self.last_offsets: Offsets = self.current_offsets

    def on_middle_step(self):
        self.update_offsets()

    def on_end_step(self):
        self.update_last_offsets()

    def insert_offset(self, offset: Offsets):
        self.current_offsets = offset
        self.last_offsets = offset

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
        if self.entities_manager is None:
            print("Entities manager is None")
            raise ValueError("Entities manager is None")

        invaders: "list[Quadcopter]" = self.entities_manager.get_armed_invaders()
        pursuers: "list[Quadcopter]" = self.entities_manager.get_armed_pursuers()

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

    @property
    def current_pursuer_velocities(self):
        return self.current_offsets.pursuer_velocities

    @property
    def current_closest_pursuer_to_invader_distance(self) -> int:
        return np.sum(
            self.compute_closest_pursuer_distance(self.current_offsets.distances)
        )

    @property
    def last_closest_pursuer_to_invader_distance(self) -> int:
        return np.sum(
            self.compute_closest_pursuer_distance(self.last_offsets.distances)
        )

    def compute_closest_pursuer_distance(self, distances) -> np.ndarray:
        return np.min(distances, axis=0)

    # def identify_captured_invaders(self) -> list:
    #    capture_mask = self.current_offsets.distances < self.CAPTURE_DISTANCE
    #    is_captured_by_any_pursuer = np.any(capture_mask, axis=0)
    #    return [
    #        invader_id
    #        for idx, invader_id in enumerate(self.current_offsets.invader_ids)
    #        if is_captured_by_any_pursuer[idx]
    #    ]

    def identify_closest_pursuer(self, invader_id: int) -> int:  # Dict[int, float]:
        """
        Identify the closest pursuer to a given invader and return their ID and distance.

        Args:
            invader_id (int): ID of the invader.

        Returns:
            Dict[int, float]: A dictionary with the closest pursuer's ID as the key and the distance as the value.
        """
        try:
            invader_index = self.current_offsets.invader_ids.index(invader_id)
        except ValueError:
            # Invader ID not found in the current offsets
            return -1  # {}

        # Extract distances to this invader from all pursuers
        distances_to_invader = self.current_offsets.distances[:, invader_index]

        # Find the minimum distance and corresponding pursuer index
        min_distance = np.min(distances_to_invader)
        closest_pursuer_index = np.argmin(distances_to_invader)

        # Get the ID of the closest pursuer
        closest_pursuer_id = self.current_offsets.pursuer_ids[closest_pursuer_index]

        return closest_pursuer_id  # {closest_pursuer_id: min_distance}

    def identify_closest_invader(self, pursuer_id: int) -> int:
        """
        Identify the closest invader to a given pursuer and return their ID.

        Args:
            pursuer_id (int): ID of the pursuer.

        Returns:
            int: The ID of the closest invader.
        """
        try:
            pursuer_index = self.current_offsets.pursuer_ids.index(pursuer_id)
        except ValueError:
            # Pursuer ID not found in the current offsets
            return -1

        # Extract distances to this pursuer from all invaders
        distances_to_pursuer = self.current_offsets.distances[pursuer_index, :]

        # Find the minimum distance and corresponding invader index
        closest_invader_index = np.argmin(distances_to_pursuer)

        # Get the ID of the closest invader
        closest_invader_id = self.current_offsets.invader_ids[closest_invader_index]

        return closest_invader_id

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
            # Máscara para invasores dentro do alcance deste perseguidor específico
            invaders_indices = np.where(range_mask[pursuer_index])[0]

            # Calcula as distâncias para invasores dentro do alcance
            distances = self.current_offsets.distances[pursuer_index, invaders_indices]

            # Associa invasores com suas respectivas distâncias e ordena
            invaders_distances = zip(invaders_indices, distances)
            sorted_invaders = sorted(invaders_distances, key=lambda x: x[1])

            # Salva os IDs dos invasores ordenados pela distância no dicionário
            ids = [self.current_offsets.invader_ids[i] for i, _ in sorted_invaders]
            if len(ids) > 0:
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

    # ===========================================================================
    #
    # ===========================================================================

    def get_invader_position(self, id: int) -> np.ndarray:
        """Get the position of the entity with the specified ID."""

        invader_index = self.current_offsets.invader_ids.index(id)
        return self.current_offsets.invader_positions[invader_index]

    def get_pursuer_position(self, id: int) -> np.ndarray:
        """Get the position of the entity with the specified ID."""

        pursuer_index = self.current_offsets.pursuer_ids.index(id)
        return self.current_offsets.pursuer_positions[pursuer_index]

    def get_purusers_positions(self) -> np.ndarray:
        return self.current_offsets.pursuer_positions
