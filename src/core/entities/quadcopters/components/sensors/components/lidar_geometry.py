import itertools
import math
import numpy as np
from typing import Dict, List, Tuple

import numpy as np
from typing import List, Tuple
from math import sqrt, ceil

import numpy as np
import itertools
from typing import Tuple, List

from core.dataclasses.angle_grid import LIDARSpec
from core.enums.channel_index import LidarChannels
from core.dataclasses.perception_keys import PerceptionKeys

class CoordinateConverter:
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
        theta = np.arccos(z / radius)
        phi = np.arctan2(y, x)
        return np.array([radius, theta, phi])





def normalize_angle(angle: float, initial: float, final: float, n_points: int) -> int:
    """Converte ângulo contínuo em ponto discreto com wrap-around."""
    return round((angle - initial) / (final - initial) * n_points) % n_points


def normalize_theta(theta: float, n_theta_points: int, theta_range: Tuple[float, float]) -> int:
    return normalize_angle(theta, theta_range[0], theta_range[1], n_theta_points)


def normalize_phi(phi: float, n_phi_points: int, phi_range: Tuple[float, float]) -> int:
    return normalize_angle(phi, phi_range[0], phi_range[1], n_phi_points)


def normalize_distance(distance: float, radius: float) -> float:
    return distance / radius


def normalize_spherical(spherical: np.ndarray, radius: float, n_theta: int, n_phi: int,
                        theta_range: Tuple[float, float], phi_range: Tuple[float, float]) -> Tuple[float, int, int]:
    """Converte coordenada esférica contínua em valores normalizados e discretizados."""
    distance, theta, phi = spherical
    norm_distance = normalize_distance(distance, radius)
    theta_point = normalize_theta(theta, n_theta, theta_range)
    phi_point = normalize_phi(phi, n_phi, phi_range)
    return norm_distance, theta_point, phi_point


# ----------------------------
# Reframing
# ----------------------------

def reframe(cartesian: np.ndarray, old_origin: np.ndarray, new_origin: np.ndarray) -> np.ndarray:
    return cartesian + old_origin - new_origin




# ----------------------------
# Coordinate Conversion Utils
# ----------------------------

def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
    r, theta, phi = spherical
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def cartesian_to_spherical(cartesian: np.ndarray) -> np.ndarray:
    x, y, z = cartesian
    r = np.linalg.norm(cartesian)
    if r == 0:
        return np.array([0.0, 0.0, 0.0])
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi])





# ----------------------------
# Angular Indexing
# ----------------------------

def index_from_radian(radian: float, min_angle: float, max_angle: float, n_points: int) -> int:
    return round((radian - min_angle) / (max_angle - min_angle) * n_points) % n_points


def radian_from_index(index: int, n_points: int, min_angle: float, max_angle: float) -> float:
    return (index / n_points) * (max_angle - min_angle) + min_angle

# ----------------------------
# Extraction
# ----------------------------


def extract_features(sphere: np.ndarray, lidar_spec: LIDARSpec) -> List[Tuple[float, float, int, int]]:

    features = []
    for theta_idx, phi_idx in itertools.product(range(lidar_spec.n_theta_points), range(lidar_spec.n_phi_points)):
        norm_dist = sphere[LidarChannels.distance.value][theta_idx][phi_idx]
        entity_type = sphere[LidarChannels.flag.value][theta_idx][phi_idx]
        if norm_dist < 1:
            theta = radian_from_index(theta_idx, lidar_spec.n_theta_points, lidar_spec.theta_initial_radian, lidar_spec.theta_final_radian)
            phi = radian_from_index(phi_idx, lidar_spec.n_phi_points, lidar_spec.phi_initial_radian, lidar_spec.phi_final_radian)
            features.append((norm_dist, entity_type, theta, phi))
    return features


# ----------------------------
# Feature Transformation
# ----------------------------

def transform_features(neighbour_features, neighbour_position, agent_position) -> List[Tuple[float, float, float, float]]:

    transformed_features = []
    for feature in neighbour_features:
        norm_dist, entity_type, theta, phi = feature

        # Convert to cartesian (local to neighbour)
        cartesian = spherical_to_cartesian(np.array((norm_dist, theta, phi)))

        # Reframe to agent
        cartesian_reframed = reframe(
            cartesian, neighbour_position, agent_position)

        # Convert back to spherical
        polar_agent = cartesian_to_spherical(cartesian_reframed)

        transformed_features.append(
            (polar_agent[0], polar_agent[1], polar_agent[2], entity_type))

    return transformed_features


# ----------------------------
# Add to Sphere
# ----------------------------


def create_sphere(
    transformed_features: List[Tuple[float, float, float, float]],
    delta_step: int,
    lidar_spec: LIDARSpec
) -> np.ndarray:
    
    NO_ENTITY = 0
    # gen_sphere(n_points_theta, n_points_phi, n_channels=3)  # Ensure sphere has 3 channels
    new_sphere = lidar_spec.empty_sphere()

    

    for r, theta, phi, entity_type in transformed_features:
        theta_idx = index_from_radian(theta, lidar_spec.theta_initial_radian, lidar_spec.theta_final_radian, lidar_spec.n_theta_points)
        phi_idx = index_from_radian(phi, lidar_spec.phi_initial_radian, lidar_spec.phi_final_radian, lidar_spec.n_phi_points)
        
        current_flag = new_sphere[LidarChannels.flag.value][theta_idx][phi_idx]
        current_time = new_sphere[LidarChannels.time.value][theta_idx][phi_idx]

        # Priority rule: prefer entity presence even if delta_step is older
        if current_flag == NO_ENTITY or current_time > delta_step:
            new_sphere[LidarChannels.distance.value][theta_idx][phi_idx] = r
            new_sphere[LidarChannels.time.value][theta_idx][phi_idx] = delta_step
            new_sphere[LidarChannels.flag.value][theta_idx][phi_idx] = entity_type

    return new_sphere


# ----------------------------
# Main Pipeline Function
# ----------------------------

def neighbor_sphere_from_new_frame(
    neighbor_sphere: np.ndarray,
    neighbor_position: np.ndarray,
    neighbor_step: int,
    agent_position: np.ndarray,
    current_step: int,

    lidar_spec: LIDARSpec,
    
) -> np.ndarray:

    delta_step = current_step - neighbor_step

    #OK - EXTRACT LIDAR FEATURES
    features_extracted = extract_features(
        neighbor_sphere,
        lidar_spec=lidar_spec
    )

    #transform features to agent's frame
    transformed_features = transform_features(
        features_extracted,
        neighbor_position,
        agent_position,
    )

    #time_channel = sphere_agent.shape[0]  # Assume next available channel

    return create_sphere(
        transformed_features,
        delta_step,
        lidar_spec
    )

def normalize_sphere(
    sphere: np.ndarray,
    lidar_spec: LIDARSpec
) -> np.ndarray:
    # Normalize the distance channel

 

    sphere[LidarChannels.distance.value] = np.clip(
        sphere[LidarChannels.distance.value] / lidar_spec.max_radius, 0.0, 1.0)

    sphere[LidarChannels.time.value] = np.clip(sphere[LidarChannels.time.value] / lidar_spec.max_temporal_offset, 0.0, 1.0)
    return sphere




































def transform_lidar_views(
    neighbors: List[Dict],
    
    

    agent_position: np.ndarray,
    sphere_agent: np.ndarray,
    agent_step: int,
    current_step: int,
    
    lidar_spec: LIDARSpec,
) -> np.ndarray:

    neighbors_spheres = []
    for neighbor in neighbors:
        neighbor_imu = neighbor[PerceptionKeys.imu]
        neighbor_lidar = neighbor[PerceptionKeys.lidar]
        neighbor_step = neighbor[PerceptionKeys.step]


        new_sphere = neighbor_sphere_from_new_frame(
            neighbor_lidar,
            neighbor_imu.get("position", np.zeros(3)),
            neighbor_step,
            agent_position,
            current_step,

            lidar_spec=lidar_spec,
        )


        new_sphere = normalize_sphere(new_sphere, lidar_spec)
        neighbors_spheres.append(new_sphere)

    neighbors_spheres.append(sphere_agent)  # Include agent's own sphere
    return np.stack(neighbors_spheres, axis=0)


def count_points(radius: float, resolution: float,
                 theta_range: Tuple[float, float],
                 phi_range: Tuple[float, float]) -> Tuple[int, int]:
    """
    Calcula o número de divisões theta e phi com base na resolução (setores por m²).
    """
    sector_surface = 1 / resolution
    sector_side = math.sqrt(sector_surface)

    theta_side = theta_range[1] - theta_range[0]
    phi_side = phi_range[1] - phi_range[0]

    n_theta = math.ceil(theta_side / sector_side)
    n_phi = math.ceil(phi_side / sector_side)

    return n_theta, n_phi
















#UNECESSARY ?
# ----------------------------
# Sphere Expansion
# ----------------------------

def gen_expanded_sphere(sphere_agent: np.ndarray) -> np.ndarray:
    # Adds a 3rd dimension for time
    n_channels, n_theta, n_phi = sphere_agent.shape
    expanded = np.ones((n_channels + 1, n_theta, n_phi), dtype=np.float32)
    expanded[:n_channels] = sphere_agent
    return expanded


def expand_and_mark_time(sphere: np.ndarray, time_channel: int, agent_step: int) -> np.ndarray:
    expanded = gen_expanded_sphere(sphere)
    expanded[time_channel] = agent_step
    return expanded
