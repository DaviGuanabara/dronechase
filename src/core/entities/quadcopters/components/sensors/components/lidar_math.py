import itertools

import pybullet as p  # type: ignore

import numpy as np
from typing import List, Optional, Tuple, Union

from core.dataclasses.angle_grid import LIDARSpec
from core.dataclasses.perception_snapshot import PerceptionSnapshot
from core.entities.entity_type import EntityType
from core.enums.channel_index import LidarChannels


class LidarMath:
    def __init__(self, lidar_spec: LIDARSpec):
        self.spec = lidar_spec
        self.max_radius = float(lidar_spec.max_radius)
        self.inv_max_radius = 1.0 / self.max_radius if self.max_radius > 0.0 else 0.0
        self.theta_range = float(
            lidar_spec.theta_final_radian - lidar_spec.theta_initial_radian)
        self.phi_range = float(
            lidar_spec.phi_final_radian - lidar_spec.phi_initial_radian)
        self.theta_scale = lidar_spec.n_theta_points / self.theta_range
        self.phi_scale = lidar_spec.n_phi_points / self.phi_range

    @staticmethod
    def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
        radius, theta, phi = spherical
        sin_theta = np.sin(theta)
        out = np.empty(3, dtype=np.float32)
        out[0] = radius * sin_theta * np.cos(phi)
        out[1] = radius * sin_theta * np.sin(phi)
        out[2] = radius * np.cos(theta)
        return out

    @staticmethod
    def cartesian_to_spherical(cartesian: np.ndarray) -> np.ndarray:
        x, y, z = cartesian
        radius = float(np.sqrt(x * x + y * y + z * z))
        out = np.empty(3, dtype=np.float32)
        if radius == 0.0:
            out[:] = 0.0
            return out

        out[0] = radius
        out[1] = np.arccos(np.clip(z / radius, -1.0, 1.0))
        out[2] = np.arctan2(y, x)
        return out

    @staticmethod
    def _invert_quaternion(quaternion: np.ndarray) -> np.ndarray:
        """
        Returns the normalized inverse of a quaternion (x, y, z, w).
        """
        quaternion = np.asarray(quaternion, dtype=np.float32)
        norm_sq = float(np.dot(quaternion, quaternion))
        if norm_sq == 0.0:
            # Some optimized IMU paths keep quaternion as a zero placeholder.
            # Treat it as identity rotation to keep LiDAR fusion robust.
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        inv = np.empty(4, dtype=np.float32)
        inv[0] = -quaternion[0] / norm_sq
        inv[1] = -quaternion[1] / norm_sq
        inv[2] = -quaternion[2] / norm_sq
        inv[3] = quaternion[3] / norm_sq
        return inv

    @staticmethod
    def _rotation_matrix_from_quaternion(quaternion: np.ndarray, transpose: bool = False) -> np.ndarray:
        quaternion = np.asarray(quaternion, dtype=np.float32)
        if float(np.dot(quaternion, quaternion)) == 0.0:
            quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        matrix = np.asarray(
            p.getMatrixFromQuaternion(quaternion),
            dtype=np.float32,
        ).reshape(3, 3)

        return matrix.T if transpose else matrix

    def reframe(
        self,
        local_vector: np.ndarray,
        neighbor_position: Optional[np.ndarray], neighbor_quaternion: Optional[np.ndarray],
        own_position: Optional[np.ndarray], own_quaternion: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Transforms a point from a neighbor's local LiDAR frame into the own_drone's local frame.
        Handles None positions and quaternions gracefully by returning None.

        Returns:
            A 3D point in the own_drone's local frame, or None if inputs are invalid.
        """
        if (
            local_vector is None or
            neighbor_position is None or own_position is None or
            neighbor_quaternion is None or own_quaternion is None
        ):
            return None

        local_vector = np.asarray(local_vector, dtype=np.float32)
        neighbor_position = np.asarray(neighbor_position, dtype=np.float32)
        own_position = np.asarray(own_position, dtype=np.float32)
        neighbor_quaternion = np.asarray(neighbor_quaternion, dtype=np.float32)
        own_quaternion = np.asarray(own_quaternion, dtype=np.float32)

        if float(np.dot(neighbor_quaternion, neighbor_quaternion)) == 0.0:
            neighbor_quaternion = np.array(
                [0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if float(np.dot(own_quaternion, own_quaternion)) == 0.0:
            own_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        neighbor_rotation = self._rotation_matrix_from_quaternion(
            neighbor_quaternion)
        own_inverse_rotation = self._rotation_matrix_from_quaternion(
            own_quaternion, transpose=True)

        global_position = neighbor_position + neighbor_rotation @ local_vector
        relative_to_own = global_position - own_position
        own_local_vector = own_inverse_rotation @ relative_to_own

        return own_local_vector.astype(np.float32, copy=False)

    # TODO: OVERFLOW AND UNDERFLOW HANDLING
    # @staticmethod
    # def index_from_radian(radian: float, min_angle: float, max_angle: float, n_points: int) -> int:
    #    return round((radian - min_angle) / (max_angle - min_angle) * n_points) % n_points

    @staticmethod
    def index_from_radian(radian: float, min_angle: float, max_angle: float, n_points: int) -> int:
        index = int((radian - min_angle) / (max_angle - min_angle) * n_points)
        if index < 0:
            return 0
        if index >= n_points:
            return n_points - 1
        return index

    # @staticmethod
    # def radian_from_index(index: int, min_angle: float, max_angle: float, n_points: int) -> float:
    #    return (index / n_points) * (max_angle - min_angle) + min_angle

    def theta_index_from_radian(self, radian: float) -> int:
        index = int((radian - self.spec.theta_initial_radian)
                    * self.theta_scale)
        if index < 0:
            return 0
        if index >= self.spec.n_theta_points:
            return self.spec.n_theta_points - 1
        return index

    @staticmethod
    def radian_from_index(index: int, min_angle: float, max_angle: float, n_points: int) -> float:
        return ((index + 0.5) / n_points) * (max_angle - min_angle) + min_angle

    def theta_radian_from_index(self, index: int) -> float:
        return self.radian_from_index(index, self.spec.theta_initial_radian, self.spec.theta_final_radian, self.spec.n_theta_points)

    def phi_radian_from_index(self, index: int) -> float:
        return self.radian_from_index(index, self.spec.phi_initial_radian, self.spec.phi_final_radian, self.spec.n_phi_points)

    def phi_index_from_radian(self, radian: float) -> int:
        index = int((radian - self.spec.phi_initial_radian) * self.phi_scale)
        if index < 0:
            return 0
        if index >= self.spec.n_phi_points:
            return self.spec.n_phi_points - 1
        return index

    # def theta_radian_from_index(self, index: int) -> float:
    #    return self.index_from_radian(index, self.spec.theta_initial_radian, self.spec.theta_final_radian, self.spec.n_theta_points)

    # def phi_radian_from_index(self, index: int) -> float:
    #   return self.index_from_radian(index, self.spec.phi_initial_radian, self.spec.phi_final_radian, self.spec.n_phi_points)

    def normalize_distance(self, distance: float) -> float:
        normalized = distance * self.inv_max_radius
        if normalized < 0.0:
            return 0.0
        if normalized > 1.0:
            return 1.0
        return float(normalized)

    def normalize_spherical(self, spherical: np.ndarray) -> Tuple[float, int, int]:
        """Converte coordenada esférica contínua em valores normalizados e discretizados."""

        distance, theta, phi = spherical
        norm_distance = self.normalize_distance(distance)
        return norm_distance, theta, phi

    def extract_features(self, temporal_sphere: Optional[np.ndarray]) -> np.ndarray:
        """
        Extracts features from a temporal sphere with time, distance, and entity type.
        Returns columns: [norm_distance, theta_rad, phi_rad, entity_type, relative_step]
        """
        if temporal_sphere is None:
            return np.empty((0, 5), dtype=np.float32)

        distance_channel = temporal_sphere[LidarChannels.distance.value]
        theta_indices, phi_indices = np.where(distance_channel < 1.0)
        n_features = theta_indices.size
        if n_features == 0:
            return np.empty((0, 5), dtype=np.float32)

        features = np.empty((n_features, 5), dtype=np.float32)
        features[:, 0] = distance_channel[theta_indices, phi_indices]
        features[:, 1] = ((theta_indices + 0.5) / self.spec.n_theta_points) * \
            self.theta_range + self.spec.theta_initial_radian
        features[:, 2] = ((phi_indices + 0.5) / self.spec.n_phi_points) * \
            self.phi_range + self.spec.phi_initial_radian
        features[:, 3] = temporal_sphere[LidarChannels.flag.value][theta_indices, phi_indices]
        features[:, 4] = temporal_sphere[LidarChannels.time.value][theta_indices, phi_indices]
        return features

    # ----------------------------
    # Feature Transformation
    # ----------------------------

    def _rotate_by_quaternion(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """
        Rotates a 3D position vector using a quaternion in PyBullet format.
        """
        rotation_matrix = self._rotation_matrix_from_quaternion(quaternion)
        return rotation_matrix @ np.asarray(position, dtype=np.float32)

    @staticmethod
    def _is_self_echo(publisher_id, agent_id):
        return publisher_id == agent_id

    def transform_features(
        self,
        neighbor: PerceptionSnapshot,
        own: PerceptionSnapshot
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Transforms features from a neighbor's frame into the own's local frame.
        Each input feature is:
            [normalized_r (0-1), theta (rad), phi (rad), entity_type, relative_step]

        The transformation:
        1. Denormalizes r
        2. Converts spherical to global Cartesian using neighbor position
        3. Transforms to own's frame (inverse translation)
        4. Converts back to spherical
        5. Normalizes r again to [0-1] for compatibility with LiDAR sphere

        Returns:
            List of features in own frame: [normalized_r, theta (rad), phi (rad), entity_type, relative_step (0-1)]

        Note:
        - Most probable feature delta_step info is outdated, as it is can be kept saved for long time ago.
        |-> It is recommended to ignore it and use directly from snapshot (that is updated on retrieval).
        """

        if neighbor.lidar_features is None:
            return []

        if neighbor.position is None or neighbor.quaternion is None or own.position is None or own.quaternion is None:
            return []

        neighbor_position = np.asarray(neighbor.position, dtype=np.float32)
        own_position = np.asarray(own.position, dtype=np.float32)
        neighbor_rotation = self._rotation_matrix_from_quaternion(
            neighbor.quaternion)
        own_inverse_rotation = self._rotation_matrix_from_quaternion(
            own.quaternion, transpose=True)

        transformed_features = []
        for feature in neighbor.lidar_features:
            # TODO: É UMA TUPLA, NÃO OBJETO FEATURES. ENTÃO EU TENHO QUE
            # ADICIONAR O ID NESSA TUPLA, NA QUARTA POSIÇÃO.
            # TODO: ESQUECER ESSE RELATIVE STEP, PQ ELE VAI ESTAR DESATUALIZADO.
            r, theta, phi, entity_type, _, publisher_id = feature
            if publisher_id == own.publisher_id:
                # print(
                #    "[DEBUG] [LIDAR MATH] [transform_features] READING OF ITSELF REMOVED")
                continue  # Skip synthetic echo of self from neighbor

            # if LidarMath._is_self_echo(publisher_id, own.publisher_id):
            #    print("READING OF ITSELF REMOVED")
            #    continue  # Skip synthetic echo of self from neighbor

            R = r * self.max_radius
            # Convert to cartesian (spherical to cartesian)
            cartesian = self.spherical_to_cartesian(
                np.array((R, theta, phi), dtype=np.float32)
            )

            global_position = neighbor_position + neighbor_rotation @ cartesian
            relative_to_own = global_position - own_position
            reframed = own_inverse_rotation @ relative_to_own

            # Convert back to spherical
            spherical = self.cartesian_to_spherical(reframed)
            normalized_spherical = self.normalize_spherical(spherical)

            # Keep temporal info
            # ONCE THE SNAPSHOT is garthered, its normalized delta is updated
            # therefore, considering no advance in step, it should be ok to use.
            transformed_features.append(
                [*normalized_spherical, entity_type, neighbor.normalized_delta])

        return transformed_features

    def decide_feature_overwrite(self, current_distance, feature_distance, invert_criteria: bool):
        """
        returns TRUE if the feature should overwrite the current one
        """
        if invert_criteria:
            # Inverted criteria: farther is better
            return feature_distance > current_distance if current_distance < 1 else True
        else:
            # Normal criteria: closer is better
            return feature_distance < current_distance

    def add_features(self, sphere: np.ndarray, features: Union[np.ndarray, List[Tuple[float, float, float, float, float, int]]], invert_prioritization_criteria: bool = False) -> Tuple[np.ndarray, List[Tuple[float, float, float, float, float, int]]]:
        """
        Adds features to the LiDAR sphere.
        Each feature: [r, theta, phi, entity_type, delta_step, publisher_id]
        """
        kept_features = {}
        distance_channel = sphere[LidarChannels.distance.value]
        flag_channel = sphere[LidarChannels.flag.value]
        time_channel = sphere[LidarChannels.time.value]

        for feature in features:
            feature_distance = feature[0]
            theta = feature[1]
            phi = feature[2]
            entity_type = feature[3]
            delta_step = feature[4]

            theta_idx = self.theta_index_from_radian(theta)
            phi_idx = self.phi_index_from_radian(phi)

            current_distance = distance_channel[theta_idx, phi_idx]

            if self.decide_feature_overwrite(current_distance, feature_distance, invert_prioritization_criteria):
                distance_channel[theta_idx, phi_idx] = feature_distance
                flag_channel[theta_idx, phi_idx] = entity_type.value / 5 if isinstance(
                    entity_type, EntityType) else entity_type
                time_channel[theta_idx, phi_idx] = delta_step
                kept_features[(theta_idx, phi_idx)] = feature

        return sphere, list(kept_features.values())

    def create_neighbor_sphere_transformed(self, transformed_features: Union[np.ndarray, List]) -> np.ndarray:
        """
        Here, we are creating a neighbor sphere that will feed the owns drone lidar from its point of view
        but we have a problem. during transformation, some features can be closer to the own drone than others
        if we use the same prioritization, closer better, we will lose valuable information (given that closer tend to be
        seen by drones own lidar) that comes from neighbors lidar.
        here is the catch, we need to invert the logic when adding the features HERE, and only HERE.

        """
        new_sphere = self.spec.empty_sphere()
        new_sphere, _ = self.add_features(
            new_sphere, transformed_features, invert_prioritization_criteria=True)
        return new_sphere

    def neighbor_sphere_from_new_frame(self, neighbor: PerceptionSnapshot, own: PerceptionSnapshot) -> Optional[np.ndarray]:
        """
        Generates a LiDAR sphere from the neighbor's memory, transformed into the own's local frame.
        Assumes:
        - `neighbor.sphere` contains temporally indexed observations.
        - Temporal offset has already been converted to a relative frame by `neighbor.build_temporal_sphere(own.step)`.

        Returns:
        - A new sphere aligned to the own's spatial frame with the neighbor’s observations embedded.
        """

        if neighbor.lidar_features is None or neighbor.position is None:
            return None

        if own.position is None:
            return None

        # print(f"[DEBUG] Reframing neighbor {neighbor.publisher_id}")
        # print(f"    Neighbor position: {neighbor.position}, quaternion: {neighbor.quaternion}")
        # print(f"    Own position:    {own.position}, quaternion:    {own.quaternion}")

        transformed_features = self.transform_features(neighbor, own)

        # time_channel = sphere_agent.shape[0]  # Assume next available channel
        # OK
        return self.create_neighbor_sphere_transformed(transformed_features)
