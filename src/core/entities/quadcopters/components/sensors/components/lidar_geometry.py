import itertools
import math
import numpy as np
from typing import List, Tuple

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


def reframe(cartesian: np.ndarray, old_origin: np.ndarray, new_origin: np.ndarray)-> np.ndarray:
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

def gen_expanded_sphere(sphere_agent: np.ndarray):
    "expand to add time layer"
    expanded_sphere
    return expanded_sphere

# ----------------------------
# Main Pipeline Function
# ----------------------------

def add_feature(sphere_agent, transformed_features, delta_step, angle_range_theta, angle_range_phi, n_points_theta, n_points_phi, distance_channel, flag_channel, time_channel):

    expanded_sphere = gen_expanded_sphere(sphere_agent)
    

    for feature in transformed_features:
        r, theta, phi, entity_type = feature[0], feature[1], feature[2], feature[3]

        theta_idx = angle_idx(theta, angle_range_theta, n_points_theta)
        phi_idx = angle_idx(phi, angle_range_phi, n_points_phi)

        if expanded_sphere[theta_idx][phi_idx][flag_channel] == 0:
            expanded_sphere[theta_idx][phi_idx][flag_channel] = entity_type
            expanded_sphere[theta_idx][phi_idx][distance_channel] = r
            expanded_sphere[theta_idx][phi_idx][time_channel] = delta_step
        

def transform_features(features_extracted, neighbour_position, agent_position):

    transformed_features = []
    for feature in features_extracted:
        norm_dist, entity_type, theta_idx, phi_idx = feature
        cartesian = spherical_to_cartesian(np.array((norm_dist, theta_idx, phi_idx)))
        cartesian_reframed = reframe(cartesian, neighbour_position, agent_position)
        polar_agent = cartesian_to_spherical(cartesian_reframed)
        transformed_features.append((polar_agent + [entity_type]))

    return transformed_features

def final_pipeline_for_each_neighbor(sphere_neighbour: np.ndarray, n_theta_points, n_phi_points, distance_channel, flag_channel, neighbour_position, agent_position, sphere_agent: np.ndarray, current_step, neighbour_step):

    features_extracted = extract_lidar_features(sphere_neighbour, n_theta_points, n_phi_points, distance_channel, flag_channel)
    transformed_features = transform_features(features_extracted, neighbour_position, agent_position)
    delta_step = current_step - neighbour_step
    return add_feature(sphere_agent, transformed_features, delta_step)