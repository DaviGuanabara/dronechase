from dataclasses import dataclass
import math
from typing import Tuple
import numpy as np



@dataclass(frozen=True)
class LIDARSpec:
    phi_initial_radian: float
    phi_final_radian: float
    theta_initial_radian: float
    theta_final_radian: float
    resolution: float
    n_channels: int
    max_radius: float = 100.0
    max_temporal_offset: int = 10

    @property
    def phi_range(self) -> float:
        return self.phi_final_radian - self.phi_initial_radian

    @property
    def theta_range(self) -> float:
        return self.theta_final_radian - self.theta_initial_radian

    @property
    def sector_side(self) -> float:
        return math.sqrt(1 / self.resolution)

    @property
    def n_theta_points(self) -> int:
        return math.ceil(self.theta_range / self.sector_side)

    @property
    def n_phi_points(self) -> int:  
        return math.ceil(self.phi_range / self.sector_side)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.n_channels, self.n_theta_points, self.n_phi_points

    def stacked_sphere_shape(self, max_neighbors: int) -> Tuple[int, int, int, int]:
        """
        Return the shape of a stacked sphere: (N, C, θ, φ)
        where N = 1 (agent) + max_neighbors
        """
        return (1 + max_neighbors, *self.shape)

    def stacked_mask_shape(self, max_neighbors: int) -> Tuple[int]:
        """
        Return the shape of the mask for the stacked sphere.
        Each entry corresponds to whether that sphere is valid.
        """
        return (1 + max_neighbors,)
    
    def stacked_output_shapes(self, max_neighbors: int) -> Tuple[Tuple[int, int, int, int], Tuple[int]]:
        """
        Returns the shapes for:
        - stacked_sphere: (1 + max_neighbors, C, θ, φ)
        - mask: (1 + max_neighbors,)
        """
        stacked_sphere_shape = (1 + max_neighbors, *self.shape)
        mask_shape = (1 + max_neighbors,)
        return stacked_sphere_shape, mask_shape



    #@property
    def empty_sphere(self) -> np.ndarray:  # sourcery skip: remove-unreachable-code
        """
        Returns an empty LiDAR grid initialized with a constant value.

        The grid has shape (n_channels, n_theta, n_phi) (C, θ, φ), with all cells set to 1.0.

        Returns:
            np.ndarray: The empty (unpopulated) LiDAR grid.
        """

        return np.ones(self.shape, dtype=np.float32)
    
    def reset_sphere(self, sphere:np.ndarray):
        """
        fill all cells with one
        """
        sphere.fill(1)
    




    #def _count_points(self) -> Tuple[int, int]:
        """
        CONSIDER RADIUS ALWAYS EQUAL TO 1
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
        spherical = (radius, theta, phi)

        This function aims to calculate the number of points in polar angle (theta) and azimuthal angle (phi),
        in which theta is between 0 and pi and phi is between -pi and +phi.
        With 4 points as constraints, a sphere detectable sector can be obtained.
        Each sector is responsable for holding the distance and the type of the identified object.
        The set of sectors is further translated into matrix.

        Params
        -------
        radius: float related to maximum observable distance.
        resolution: float related to number of sectors per m2.

        Returns
        -------
        n_theta_points: int
        n_phi_points: int
        """
    #    sector_surface: float = 1 / grid_spec.resolution
    #    sector_side: float = math.sqrt(sector_surface)

    #    theta_range: float = grid_spec.theta_final_radian - grid_spec.theta_initial_radian
    #    phi_range: float = grid_spec.phi_final_radian - grid_spec.phi_initial_radian

    #    theta_side: float = theta_range  # * radius
    #    phi_side: float = phi_range  # * radius

    #    n_theta_points: int = math.ceil(theta_side / sector_side)
    #    n_phi_points: int = math.ceil(phi_side / sector_side)

    #    return n_theta_points, n_phi_points
    
