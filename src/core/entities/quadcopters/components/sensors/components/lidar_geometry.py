import math
import numpy as np
from typing import Tuple


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
