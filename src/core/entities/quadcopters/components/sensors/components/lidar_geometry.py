import itertools
import math
import numpy as np
from typing import Dict, List, Tuple

import numpy as np
from typing import List, Tuple
from math import sqrt, ceil


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


import numpy as np
import itertools
from typing import Tuple, List

# ----------------------------
# Reframing
# ----------------------------

def reframe(cartesian: np.ndarray, old_origin: np.ndarray, new_origin: np.ndarray) -> np.ndarray:
    return cartesian + old_origin - new_origin


# ----------------------------
# Extraction
# ----------------------------

def extract_lidar_features(sphere: np.ndarray, n_theta_points, n_phi_points, distance_channel, flag_channel) -> List[Tuple[float, float, int, int]]:
    features = []
    for theta_idx, phi_idx in itertools.product(range(n_theta_points), range(n_phi_points)):
        norm_dist = sphere[distance_channel][theta_idx][phi_idx]
        entity_type = sphere[flag_channel][theta_idx][phi_idx]
        if 0 < norm_dist < 1:
            features.append((norm_dist, entity_type, theta_idx, phi_idx))
    return features


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

def angle_idx(angle: float, angle_range: Tuple[float, float], n_points: int) -> int:
    return round((angle - angle_range[0]) / (angle_range[1] - angle_range[0]) * n_points) % n_points

def get_angle_from_idx(idx: int, n_points: int, angle_range: Tuple[float, float]) -> float:
    return (idx / n_points) * (angle_range[1] - angle_range[0]) + angle_range[0]


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

# ----------------------------
# Feature Transformation
# ----------------------------

def transform_features(features_extracted, neighbour_position, agent_position, n_theta_points, n_phi_points, angle_range_theta, angle_range_phi):

    transformed_features = []
    for feature in features_extracted:
        norm_dist, entity_type, theta_idx, phi_idx = feature
        theta = get_angle_from_idx(theta_idx, n_theta_points, angle_range_theta)
        phi = get_angle_from_idx(phi_idx, n_phi_points, angle_range_phi)

        # Convert to cartesian (local to neighbour)
        cartesian = spherical_to_cartesian(np.array((norm_dist, theta, phi)))

        # Reframe to agent
        cartesian_reframed = reframe(cartesian, neighbour_position, agent_position)

        # Convert back to spherical
        polar_agent = cartesian_to_spherical(cartesian_reframed)

        transformed_features.append((polar_agent[0], polar_agent[1], polar_agent[2], entity_type))

    return transformed_features


# ----------------------------
# Add to Sphere
# ----------------------------

#TODO: CODE COPIED FROM LIDAR.PY, REFACTOR TO A COMMON MODULE
def gen_sphere(
    n_theta_points, n_phi_points, n_channels: int = 2
) -> np.ndarray:
    return np.ones((n_channels, n_theta_points, n_phi_points), dtype=np.float32)

def create_sphere(
    transformed_features: List[Tuple[float, float, float, float]],
    delta_step: int,
    angle_range_theta: Tuple[float, float],
    angle_range_phi: Tuple[float, float],
    n_points_theta: int,
    n_points_phi: int,
    distance_channel: int,
    flag_channel: int,
    time_channel: int
) -> np.ndarray:
    
    NO_ENTITY = 0
    new_sphere = gen_sphere(n_points_theta, n_points_phi, n_channels=3)  # Ensure sphere has 3 channels

    

    for r, theta, phi, entity_type in transformed_features:
        theta_idx = angle_idx(theta, angle_range_theta, n_points_theta)
        phi_idx = angle_idx(phi, angle_range_phi, n_points_phi)

        current_flag = new_sphere[flag_channel][theta_idx][phi_idx]
        current_time = new_sphere[time_channel][theta_idx][phi_idx]

        # Priority rule: prefer entity presence even if delta_step is older
        if current_flag == NO_ENTITY or current_time > delta_step:
            new_sphere[distance_channel][theta_idx][phi_idx] = r
            new_sphere[time_channel][theta_idx][phi_idx] = delta_step
            new_sphere[flag_channel][theta_idx][phi_idx] = entity_type
            

    return new_sphere


# ----------------------------
# Main Pipeline Function
# ----------------------------

def neighbour_sphere_from_new_frame(
    sphere_neighbour: np.ndarray,
    n_theta_points: int,
    n_phi_points: int,
    distance_channel: int,
    flag_channel: int,
    time_channel: int,
    neighbour_position: np.ndarray,
    agent_position: np.ndarray,
    current_step: int,
    neighbour_step: int,
    angle_range_theta: Tuple[float, float],
    angle_range_phi: Tuple[float, float],
) -> np.ndarray:

    delta_step = current_step - neighbour_step

    features_extracted = extract_lidar_features(
        sphere_neighbour,
        n_theta_points,
        n_phi_points,
        distance_channel,
        flag_channel
    )

    transformed_features = transform_features(
        features_extracted,
        neighbour_position,
        agent_position,
        n_theta_points,
        n_phi_points,
        angle_range_theta,
        angle_range_phi
    )

    #time_channel = sphere_agent.shape[0]  # Assume next available channel

    return create_sphere(
        transformed_features,
        delta_step,
        angle_range_theta,
        angle_range_phi,
        n_theta_points,
        n_phi_points,
        distance_channel,
        flag_channel,
        time_channel
    )

def normalize_sphere(
    sphere: np.ndarray,
    distance_channel: int,
    time_channel: int,
    MAX_TEMPORAL_OFFSET: int = 10,
    MAX_RADIUS: int = 100
) -> np.ndarray:
    # Normalize the distance channel
    sphere[distance_channel] = np.clip(
        sphere[distance_channel] / MAX_RADIUS, 0.0, 1.0)

    sphere[time_channel] = np.clip(sphere[time_channel] / MAX_TEMPORAL_OFFSET, 0.0, 1.0)
    return sphere

def final_pipeline(
    neighbours: List[Dict],
    neighbour_lidar_key: str,
    neighbour_imu_key: str,
    neighbour_step_key: str,
   
    n_theta_points: int,
    n_phi_points: int,
    distance_channel: int,
    flag_channel: int,
    time_channel: int,

    agent_position: np.ndarray,
    sphere_agent: np.ndarray,
    agent_step: int,
    current_step: int,
    
    angle_range_theta: Tuple[float, float],
    angle_range_phi: Tuple[float, float],
    MAX_TEMPORAL_OFFSET: int = 10,
    MAX_RADIUS: int = 100
) -> np.ndarray:

    neighbours_spheres = []
    for neighbour in neighbours:
        neighbour_imu = neighbour[neighbour_imu_key]
        neighbour_lidar = neighbour[neighbour_lidar_key]
        neighbour_step = neighbour[neighbour_step_key]



        new_sphere = neighbour_sphere_from_new_frame(
            neighbour_lidar,
            n_theta_points,
            n_phi_points,
            distance_channel,
            flag_channel,
            time_channel,
            neighbour_imu['position'],
            agent_position,
            current_step,
            neighbour_step,
            angle_range_theta,
            angle_range_phi,
            
        )
        new_sphere = normalize_sphere(
            new_sphere, distance_channel, time_channel, MAX_TEMPORAL_OFFSET=MAX_TEMPORAL_OFFSET, MAX_RADIUS=MAX_RADIUS)
        neighbours_spheres.append(new_sphere)

    sphere_agent = expand_and_mark_time(
        sphere_agent, time_channel, agent_step)
    neighbours_spheres.append(sphere_agent)  # Include agent's own sphere
    return np.stack(neighbours_spheres, axis=0)
