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
        own_position: Optional[np.ndarray], own_quaternion: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Transforms a point from a neighbor's local LiDAR frame into the own_drone's local frame.
        Handles None positions and quaternions gracefully by returning None.

        Returns:
            A 3D point in the own_drone's local frame, or None if inputs are invalid.
        """
        import pybullet as p

        if (
            local_vector is None or
            neighbor_position is None or own_position is None or
            neighbor_quaternion is None or own_quaternion is None
        ):
            return None

        global_vector = p.rotateVector(neighbor_quaternion, local_vector)
        global_position = global_vector + neighbor_position

        relative_to_own = global_position - own_position
        own_q_inverted = self._invert_quaternion(own_quaternion)

        own_local_vector = p.rotateVector(own_q_inverted, relative_to_own)

        return np.array(own_local_vector)




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
        """

        if neighbor.lidar_features is None:
            return []

        transformed_features = []
        for feature in neighbor.lidar_features:

            if neighbor.position is None or neighbor.quaternion is None or own.position is None or own.quaternion is None:
                print(
                    "[WARNING] Missing position or quaternion for transformation. Skipping feature.")
                continue
            
            #TODO: É UMA TUPLA, NÃO OBJETO FEATURES. ENTÃO EU TENHO QUE 
            #ADICIONAR O ID NESSA TUPLA, NA QUARTA POSIÇÃO.
            #TODO: ESQUECER ESSE RELATIVE STEP, PQ ELE VAI ESTAR DESATUALIZADO.
            r, theta, phi, entity_type, _, publisher_id = feature
            if publisher_id == own.publisher_id:
               print(
                   "[DEBUG] [LIDAR MATH] [transform_features] READING OF ITSELF REMOVED")
               continue  # Skip synthetic echo of self from neighbor
            
            #if LidarMath._is_self_echo(publisher_id, own.publisher_id):
            #    print("READING OF ITSELF REMOVED")
            #    continue  # Skip synthetic echo of self from neighbor

            
            R = r * self.spec.max_radius
            # Convert to cartesian (spherical to cartesian)
            cartesian = self.spherical_to_cartesian(spherical=np.array((R, theta, phi)))

            reframed = self.reframe(
                cartesian,
                neighbor.position, neighbor.quaternion,
                own.position, own.quaternion
            )

            if reframed is None:
                continue

            # Convert back to spherical
            spherical = self.cartesian_to_spherical(reframed)
            normalized_spherical = self.normalize_spherical(spherical)
            
            # Keep temporal info
            #ONCE THE SNAPSHOT is garthered, its normalized delta is updated
            #therefore, considering no advance in step, it should be ok to use.
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
        It expects a empty sphere, that will be populated by its own perpective.
        In other words: the sphere should contain the closest feature.
        But, in some cases, the features from the neighbor's perspective may provide valuable information that is not captured by the own perspective. To handle
        that, it the criteria can be inverted.

        Each feature: [r, theta, phi, entity_type, delta_step, publisher_id]
        - `delta_step` is a relative temporal offset: 0 means freshest (most recent).
        - If current cell has `delta_step == 0`, it is the freshest and must be preserved.
        """
        kept_features = {}
        for feature in features:
            #r, theta, phi, entity_type, delta_step = feature
            feature_distance = feature[0]
            theta = feature[1]
            phi = feature[2]
            entity_type = feature[3]
            delta_step = feature[4]

            theta_idx = self.theta_index_from_radian(theta)
            phi_idx = self.phi_index_from_radian(phi)

            current_distance = sphere[LidarChannels.distance.value][theta_idx][phi_idx]

            #if feature_distance < current_distance:
            if self.decide_feature_overwrite(current_distance, feature_distance, invert_prioritization_criteria):

                print(f"[DEBUG] trocando {current_distance} por {feature_distance}")
                sphere[LidarChannels.distance.value][theta_idx][phi_idx] = feature_distance
                sphere[LidarChannels.flag.value][theta_idx][phi_idx] = entity_type.value / 5 if isinstance(
                    entity_type, EntityType) else entity_type
                sphere[LidarChannels.time.value][theta_idx][phi_idx] = delta_step

                kept_features[(theta_idx, phi_idx)] = feature

        return sphere, list(kept_features.values())

    def create_neighbor_sphere_transformed(self, transformed_features: np.ndarray) -> np.ndarray:
        """
        Here, we are creating a neighbor sphere that will feed the owns drone lidar from its point of view
        but we have a problem. during transformation, some features can be closer to the own drone than others
        if we use the same prioritization, closer better, we will lose valuable information (given that closer tend to be
        seen by drones own lidar) that comes from neighbors lidar.
        here is the catch, we need to invert the logic when adding the features HERE, and only HERE.
        
        """
        new_sphere = self.spec.empty_sphere()
        new_sphere, _ = self.add_features(new_sphere, transformed_features, invert_prioritization_criteria=True)
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

        print(f"[DEBUG] Reframing neighbor {neighbor.publisher_id}")
        print(f"    Neighbor position: {neighbor.position}, quaternion: {neighbor.quaternion}")
        print(f"    Own position:    {own.position}, quaternion:    {own.quaternion}")

        transformed_features = self.transform_features(neighbor, own)

        #time_channel = sphere_agent.shape[0]  # Assume next available channel
        #OK
        return self.create_neighbor_sphere_transformed(np.asarray(transformed_features))

