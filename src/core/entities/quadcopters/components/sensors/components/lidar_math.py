import itertools

import numpy as np
from typing import List, Optional, Tuple, Union

from core.dataclasses.angle_grid import LIDARSpec
from core.dataclasses.perception_snapshot import PerceptionSnapshot
from core.entities.entity_type import EntityType
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
        #theta = np.arccos(z / radius)
        theta = np.arccos(np.clip(z / radius, -1.0, 1.0))

        phi = np.arctan2(y, x)
        return np.array([radius, theta, phi])
    

    @staticmethod
    def _invert_quaternion(quaternion: np.ndarray) -> np.ndarray:
        """
        Returns the normalized inverse of a quaternion (x, y, z, w).
        """
        norm_sq = np.dot(quaternion, quaternion)
        if norm_sq == 0:
            raise ValueError("Cannot invert zero-length quaternion.")
        
        # Conjugate: [x, y, z, w] → [-x, -y, -z, w]
        conjugate = np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])
        
        # Inverse = conjugate / norm²
        return conjugate / norm_sq


    def reframe(
        self,
        local_vector: np.ndarray,
        neighbor_position: Optional[np.ndarray], neighbor_quaternion: Optional[np.ndarray],
        agent_position: Optional[np.ndarray], agent_quaternion: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Transforms a point from a neighbor's local LiDAR frame into the agent's local frame.
        Handles None positions and quaternions gracefully by returning None.

        Returns:
            A 3D point in the agent's local frame, or None if inputs are invalid.
        """
        import pybullet as p

        if (
            local_vector is None or
            neighbor_position is None or agent_position is None or
            neighbor_quaternion is None or agent_quaternion is None
        ):
            return None

        global_vector = p.rotateVector(neighbor_quaternion, local_vector)
        global_position = global_vector + neighbor_position

        relative_to_agent = global_position - agent_position
        agent_q_inverted = self._invert_quaternion(agent_quaternion)

        agent_local_vector = p.rotateVector(agent_q_inverted, relative_to_agent)

        return np.array(agent_local_vector)




    #TODO: OVERFLOW AND UNDERFLOW HANDLING
    #@staticmethod
    #def index_from_radian(radian: float, min_angle: float, max_angle: float, n_points: int) -> int:
    #    return round((radian - min_angle) / (max_angle - min_angle) * n_points) % n_points

    @staticmethod
    def index_from_radian(radian: float, min_angle: float, max_angle: float, n_points: int) -> int:
        index = int((radian - min_angle) / (max_angle - min_angle) * n_points)
        return np.clip(index, 0, n_points - 1)


    #@staticmethod
    #def radian_from_index(index: int, min_angle: float, max_angle: float, n_points: int) -> float:
    #    return (index / n_points) * (max_angle - min_angle) + min_angle
    
    def theta_index_from_radian(self, radian: float) -> int:
        return self.index_from_radian(radian, self.spec.theta_initial_radian, self.spec.theta_final_radian, self.spec.n_theta_points)

    @staticmethod
    def radian_from_index(index: int, min_angle: float, max_angle: float, n_points: int) -> float:
        return ((index + 0.5) / n_points) * (max_angle - min_angle) + min_angle


    def theta_radian_from_index(self, index: int) -> float:
        return self.radian_from_index(index, self.spec.theta_initial_radian, self.spec.theta_final_radian, self.spec.n_theta_points)


    def phi_radian_from_index(self, index: int) -> float:
        return self.radian_from_index(index, self.spec.phi_initial_radian, self.spec.phi_final_radian, self.spec.n_phi_points)


    def phi_index_from_radian(self, radian: float) -> int:
        return self.index_from_radian(radian, self.spec.phi_initial_radian, self.spec.phi_final_radian, self.spec.n_phi_points)

    #def theta_radian_from_index(self, index: int) -> float:
    #    return self.index_from_radian(index, self.spec.theta_initial_radian, self.spec.theta_final_radian, self.spec.n_theta_points)

    #def phi_radian_from_index(self, index: int) -> float:
    #   return self.index_from_radian(index, self.spec.phi_initial_radian, self.spec.phi_final_radian, self.spec.n_phi_points)

    def normalize_distance(self, distance: float) -> float:
        return np.clip(distance / self.spec.max_radius, 0, 1)


    def normalize_spherical(self,spherical: np.ndarray) -> Tuple[float, int, int]:
        """Converte coordenada esférica contínua em valores normalizados e discretizados."""

        distance, theta, phi = spherical
        norm_distance = self.normalize_distance(distance)
        return norm_distance, theta, phi




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

    def _rotate_by_quaternion(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """
        Rotates a 3D position vector using a quaternion (w, x, y, z) in the PyBullet format.
        This is done via conversion to rotation matrix.
        """
        import pybullet as p  # Safe to keep this inside the function if used only here
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        return rotation_matrix @ position


    def transform_features(
        self,
        neighbor: PerceptionSnapshot,
        agent: PerceptionSnapshot
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

        if neighbor.lidar_features is None:
            return []

        transformed_features = []
        for feature in neighbor.lidar_features:
            r, theta, phi, entity_type, relative_step = feature
            R = r * self.spec.max_radius
            # Convert to cartesian (spherical to cartesian)
            cartesian = self.spherical_to_cartesian(spherical=np.array((R, theta, phi)))

            if neighbor.position is None or neighbor.quaternion is None or agent.position is None or agent.quaternion is None:
                continue

            reframed = self.reframe(
                cartesian,
                neighbor.position, neighbor.quaternion,
                agent.position, agent.quaternion
            )

            if reframed is None:
                continue

            # Convert back to spherical
            spherical = self.cartesian_to_spherical(reframed)
            normalized_sphercial = self.normalize_spherical(spherical)
            
            # Keep temporal info
            transformed_features.append([*normalized_sphercial, entity_type, relative_step])

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
            
            #TODO: I really need to find a better way to handle the entity_type inside the flag channel.
            if delta_step < current_time:
                sphere[LidarChannels.distance.value][theta_idx][phi_idx] = r
                sphere[LidarChannels.flag.value][theta_idx][phi_idx] = entity_type.value / 5 if isinstance(
                    entity_type, EntityType) else entity_type
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

        if neighbor.sphere is None or neighbor.position is None: 
            return None
        
        if agent.position is None: 
            return None

        features_extracted = neighbor.lidar_features
        #temporal_sphere = neighbor.build_temporal_sphere(agent.step)
        #features_extracted = self.extract_features(temporal_sphere)

        #Ok - transform features to agent's frame
        if features_extracted is None:
            return self.create_sphere(np.asarray([]))

        transformed_features = self.transform_features(neighbor, agent)

        #time_channel = sphere_agent.shape[0]  # Assume next available channel
        #OK
        return self.create_sphere(np.asarray(transformed_features))

