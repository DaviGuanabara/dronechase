import itertools

import numpy as np
from typing import List, Optional, Tuple, Union

from core.dataclasses.angle_grid import LIDARSpec
from core.dataclasses.perception_snapshot import PerceptionSnapshot
from core.enums.channel_index import LidarChannels


class LidarMath:
    def __init__(self, lidar_spec: LIDARSpec):
        self.spec = lidar_spec

    @staticmethod
    def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
        radius, theta, phi = spherical
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.array([x, y, z])

    @staticmethod
    def cartesian_to_spherical(cartesian: np.ndarray) -> np.ndarray:
        x, y, z = cartesian
        radius = np.sqrt(x**2 + y**2 + z**2)
        if radius == 0:
            return np.array([0.0, 0.0, 0.0])
        theta = np.arccos(z / radius)
        phi = np.arctan2(y, x)
        return np.array([radius, theta, phi])

    @staticmethod
    def normalize_angle(angle: float, initial: float, final: float, n_points: int) -> int:
        """Converte ângulo contínuo em ponto discreto com wrap-around."""
        return round((angle - initial) / (final - initial) * n_points) % n_points
    

    @staticmethod
    def reframe(cartesian: np.ndarray, frame: np.ndarray, new_frame: np.ndarray) -> np.ndarray:
        return cartesian + frame - new_frame

    @staticmethod
    def index_from_radian(radian: float, min_angle: float, max_angle: float, n_points: int) -> int:
        return round((radian - min_angle) / (max_angle - min_angle) * n_points) % n_points

    @staticmethod
    def radian_from_index(index: int, min_angle: float, max_angle: float, n_points: int) -> float:
        return (index / n_points) * (max_angle - min_angle) + min_angle
    
    def theta_index_from_radian(self, radian: float) -> int:
        return self.index_from_radian(radian, self.spec.theta_initial_radian, self.spec.theta_final_radian, self.spec.n_theta_points)

    def phi_index_from_radian(self, radian: float) -> int:
        return self.index_from_radian(radian, self.spec.phi_initial_radian, self.spec.phi_final_radian, self.spec.n_phi_points)

    def theta_radian_from_index(self, index: int) -> float:
        return self.index_from_radian(index, self.spec.theta_initial_radian, self.spec.theta_final_radian, self.spec.n_theta_points)

    def phi_radian_from_index(self, index: int) -> float:
        return self.index_from_radian(index, self.spec.phi_initial_radian, self.spec.phi_final_radian, self.spec.n_phi_points)

    def normalize_theta(self, theta: float) -> int:
        return self.normalize_angle(theta, self.spec.theta_initial_radian, self.spec.theta_final_radian, self.spec.n_theta_points)


    def normalize_phi(self, phi: float) -> int:
        return self.normalize_angle(phi, self.spec.phi_initial_radian, self.spec.phi_final_radian, self.spec.n_phi_points)


    def normalize_distance(self, distance: float) -> float:
        return np.clip(distance / self.spec.max_radius, 0, 1)


    def normalize_spherical(self,spherical: np.ndarray) -> Tuple[float, int, int]:
        """Converte coordenada esférica contínua em valores normalizados e discretizados."""

        distance, theta, phi = spherical
        norm_distance = self.normalize_distance(distance)
        theta_point = self.normalize_theta(theta)
        phi_point = self.normalize_phi(phi)
        return norm_distance, theta_point, phi_point




    def extract_features(self, temporal_sphere: Optional[np.ndarray]) -> np.ndarray:
        """
        Extracts features from a temporal sphere with time, distance, and entity type.
        Each feature: [norm_distance, theta_rad, phi_rad, entity_type, step]

        returns a list of [normalized distance (r), theta (in radians), phi (in radians), entity type, relative_step]
        """

        if temporal_sphere is None:
            return np.array([])

        features = []
        for theta_idx, phi_idx in itertools.product(
            range(self.spec.n_theta_points), range(self.spec.n_phi_points)
        ):
            norm_dist = temporal_sphere[LidarChannels.distance.value][theta_idx][phi_idx]

            if norm_dist < 1:
                theta = self.theta_radian_from_index(theta_idx)
                phi = self.phi_radian_from_index(phi_idx)
                relative_step = temporal_sphere[LidarChannels.time.value][theta_idx][phi_idx]
                entity_type = temporal_sphere[LidarChannels.flag.value][theta_idx][phi_idx]
                features.append([norm_dist, theta, phi, entity_type, relative_step])

        return np.array(features)


    # ----------------------------
    # Feature Transformation
    # ----------------------------

    def transform_features(
        self,
        neighbour_features: np.ndarray,
        neighbour_position: np.ndarray,
        agent_position: np.ndarray
    ) -> List[Tuple[float, float, float, float, float]]:

        """
        Transforms features from a neighbor's frame into the agent's local frame.
        Each input feature is:
            [normalized_r (0-1), theta (rad), phi (rad), entity_type, relative_step]

        The transformation:
        1. Denormalizes r
        2. Converts spherical to global Cartesian using neighbor position
        3. Transforms to agent's frame (inverse translation)
        4. Converts back to spherical
        5. Normalizes r again to [0-1] for compatibility with LiDAR sphere

        Returns:
            List of features in agent frame: [normalized_r, theta (rad), phi (rad), entity_type, relative_step (0-1)]
        """

        transformed_features = []
        for feature in neighbour_features:
            r, theta, phi, entity_type, relative_step = feature
            R = r * self.spec.max_radius
            # Convert to cartesian (spherical to cartesian)
            cartesian = self.spherical_to_cartesian(spherical=np.array((R, theta, phi)))

            # Reframe neighbor to agent
            cartesian_reframed = self.reframe(cartesian, neighbour_position, agent_position)

            # Convert back to spherical
            spherical_agent = self.cartesian_to_spherical(cartesian_reframed)
            R_from_agent = spherical_agent[0]
            R = self.normalize_distance(R_from_agent)
            
            # Keep temporal info
            transformed_features.append([R, *spherical_agent[1:], entity_type, relative_step])

        return transformed_features


    def add_features(self, sphere: np.ndarray, features: Union[np.ndarray, List[Tuple[float, float, float, float, float]]]) -> np.ndarray:
        """
        Adds features to the LiDAR sphere, preserving the most recent observations.
        
        Each feature: [r, theta, phi, entity_type, delta_step]
        - `delta_step` is a relative temporal offset: 0 means freshest (most recent).
        - If current cell has `delta_step == 0`, it is the freshest and must be preserved.
        """

        for feature in features:
            r, theta, phi, entity_type, delta_step = feature

            theta_idx = self.theta_index_from_radian(theta)
            phi_idx = self.phi_index_from_radian(phi)

            current_time = sphere[LidarChannels.time.value][theta_idx][phi_idx]

            if current_time == 0:
                continue  # Don't overwrite freshest data

            if delta_step < current_time:
                sphere[LidarChannels.distance.value][theta_idx][phi_idx] = r
                sphere[LidarChannels.flag.value][theta_idx][phi_idx] = entity_type
                sphere[LidarChannels.time.value][theta_idx][phi_idx] = delta_step

        return sphere




    def create_sphere(self, transformed_features: np.ndarray) -> np.ndarray:
        
        new_sphere = self.spec.empty_sphere()
        new_sphere = self.add_features(new_sphere, transformed_features)
        return new_sphere


    def neighbor_sphere_from_new_frame(self, neighbor: PerceptionSnapshot, agent: PerceptionSnapshot) -> Optional[np.ndarray]:
        """
        Generates a LiDAR sphere from the neighbor's memory, transformed into the agent's local frame.
        Assumes:
        - `neighbor.sphere` contains temporally indexed observations.
        - Temporal offset has already been converted to a relative frame by `neighbor.build_temporal_sphere(agent.step)`.

        Returns:
        - A new sphere aligned to the agent's spatial frame with the neighbor’s observations embedded.
        """

        if not neighbor.sphere or not neighbor.position: 
            return None
        
        if not agent.position: 
            return None

        temporal_sphere = neighbor.build_temporal_sphere(agent.step)
        features_extracted = self.extract_features(temporal_sphere)

        #Ok - transform features to agent's frame
        transformed_features = self.transform_features(
            features_extracted,
            neighbor.position,
            agent.position)

        #time_channel = sphere_agent.shape[0]  # Assume next available channel
        #OK
        return self.create_sphere(np.asarray(transformed_features))

