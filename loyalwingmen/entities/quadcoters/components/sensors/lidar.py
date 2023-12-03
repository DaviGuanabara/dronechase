import math
import numpy as np
from typing import Tuple, List, Dict
import pybullet as p
from gymnasium import spaces
from .sensor_interface import Sensor

from ....entity_type import EntityType

# from ....entity_type import EntityType
# from ..base.quadcopter_type import ObjectType


from enum import Enum, auto


class Channels(Enum):
    DISTANCE_CHANNEL = 0
    FLAG_CHANNEL = 1


class LiDARBufferManager:
    def __init__(self):
        self.buffer = {}

    def buffer_message(self, message: Dict, publisher_id: int):
        self.buffer[str(publisher_id)] = message

    def clear_buffer_data(self, publisher_id: int) -> None:
        self.buffer.pop(str(publisher_id), None)

    def get_all_data(self) -> Dict:
        return self.buffer


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


class LiDAR(Sensor):
    """
    This class aims to simulate a spherical LiDAR reading. It uses physics convention for spherical coordinates (radius, theta, phi).
    In this convention, theta is the polar angle and varies between 0 and +pi, from z to xy plane.
    Phi is the azimuthal angle, and varies between -pi and +pi, from x to y.
    https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """

    def __init__(
        self,
        parent_id: int,
        client_id: int,
        radius: float = 20,
        resolution: float = 8,
        debug: bool = False,
    ):
        """
        Params
        -------
        max_distance: float, the radius of the sphere
        resolution: number of sectors per m2
        """

        self.parent_id = parent_id
        self.client_id = client_id
        self.debug = debug
        self.debug_line_id = -1
        self.debug_lines_id = []

        self.n_channels: int = len(Channels)

        self.flag_size = 3

        self.radius = radius
        self.resolution = resolution

        self.THETA_INITIAL_RADIAN = 0
        self.THETA_FINAL_RADIAN = np.pi
        self.THETA_SIZE = self.THETA_FINAL_RADIAN - self.THETA_INITIAL_RADIAN

        self.PHI_INITIAL_RADIAN = -np.pi
        self.PHI_FINAL_RADIAN = np.pi
        self.PHI_SIZE = self.PHI_FINAL_RADIAN - self.PHI_INITIAL_RADIAN

        self.n_theta_points, self.n_phi_points = self._count_points(radius, resolution)
        self.sphere: np.ndarray = self._gen_sphere(
            self.n_theta_points, self.n_phi_points, self.n_channels
        )

        self.buffer_manager = LiDARBufferManager()
        self.parent_inertia: Dict = {}

        self.DISTANCE_CHANNEL_IDX: int = Channels.DISTANCE_CHANNEL.value
        self.FLAG_CHANNEL_IDX: int = Channels.FLAG_CHANNEL.value

        self.threshold: float = 1

    def _get_flag(self, entity_type: EntityType) -> float:
        object_type_length = len(EntityType)
        # Normalize the enum value to a fraction of its length
        flag_value = entity_type.value / object_type_length
        return flag_value

    def _get_ObjectType_from_flag(self, flag: float) -> EntityType:
        object_type_length = len(EntityType) + 1
        return EntityType(object_type_length * flag)

    # ============================================================================================================
    # Setup Functions
    # ============================================================================================================

    def _count_points(self, radius: float, resolution: float = 1) -> Tuple[int, int]:
        """
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

        sector_surface: float = 1 / resolution
        sector_side: float = math.sqrt(sector_surface)

        theta_range: float = self.THETA_FINAL_RADIAN - self.THETA_INITIAL_RADIAN
        phi_range: float = self.PHI_FINAL_RADIAN - self.PHI_INITIAL_RADIAN

        theta_side: float = theta_range  # * radius
        phi_side: float = phi_range  # * radius

        n_theta_points: int = math.ceil(theta_side / sector_side)
        n_phi_points: int = math.ceil(phi_side / sector_side)

        return n_theta_points, n_phi_points

    def convert_angle_point_to_angles(
        self, theta_point, phi_point, n_theta_points, n_phi_points
    ) -> Tuple[float, float]:
        theta_range: float = self.THETA_FINAL_RADIAN - self.THETA_INITIAL_RADIAN
        phi_range: float = self.PHI_FINAL_RADIAN - self.PHI_INITIAL_RADIAN

        theta = (theta_point - 0) / (
            n_theta_points - 0
        ) * theta_range + self.THETA_INITIAL_RADIAN
        phi = (phi_point - 0) / (n_phi_points - 0) * phi_range + self.PHI_INITIAL_RADIAN

        return theta, phi

    def _gen_sphere(
        self, n_theta_points, n_phi_points, n_channels: int = 2
    ) -> np.ndarray:
        return np.ones((n_channels, n_theta_points, n_phi_points), dtype=np.float32)

    def reset(self):
        self.sphere: np.ndarray = self._gen_sphere(
            self.n_theta_points, self.n_phi_points
        )

    # ============================================================================================================
    # Matrix Functions
    # ============================================================================================================

    def _add_spherical(
        self,
        target_id,
        spherical: np.ndarray,
        flag: float = 1,
    ):
        norm_distance, theta_point, phi_point = self._normalize_spherical(spherical)
        current_norm_distance = self.sphere[self.DISTANCE_CHANNEL_IDX][theta_point][
            phi_point
        ]

        if norm_distance > current_norm_distance:
            return

        self.sphere[self.DISTANCE_CHANNEL_IDX][theta_point][phi_point] = norm_distance
        self.sphere[self.FLAG_CHANNEL_IDX][theta_point][phi_point] = flag

    def _rotate_position(self, position: np.ndarray) -> np.ndarray:
        quaternion = self.parent_inertia["quaternion"]
        rotation_matrix = (
            np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3).T
        )
        return np.matmul(rotation_matrix, position)

    def _add_end_position(
        self,
        target_id,
        end_position: np.ndarray,
        current_position: np.ndarray = np.array([0, 0, 0]),
        flag: float = 1,
    ):
        relative_position: np.ndarray = end_position - current_position
        rotated_position = self._rotate_position(relative_position)
        distance: np.float32 = np.linalg.norm(rotated_position)

        if distance > 0 and distance < self.radius:
            spherical: np.ndarray = CoordinateConverter.cartesian_to_spherical(
                rotated_position
            )
            self._add_spherical(target_id, spherical, flag)

    def _add_end_position_for_entity(
        self, position: np.ndarray, entity_type: EntityType, target_id: int
    ) -> None:
        """Add an ending position for a given entity type."""
        if len(position) > 0:
            current_position = self.parent_inertia["position"]
            entity_flag = self._get_flag(entity_type)
            self._add_end_position(target_id, position, current_position, entity_flag)

    def get_sphere(self) -> np.ndarray:
        return self.sphere

    def get_data_shape(self) -> Tuple:
        return self.sphere.shape

    @property
    def buffer(self):
        return self.buffer_manager.buffer.copy()

    # ============================================================================================================
    # Sensor Functions
    # ============================================================================================================

    """
        
        This 'buffer_publisher_inertial_data' is the first step to update the sensor data
        it receives the inertial data from the publisher and stores it in the buffer
        the publisher is triggered when imu sensor is updated, in the quacopter class
        
        Then, you have 'update_data', that fetch tha data from the buffer
        Then, 'read_data' that returns the data to the sensor manager
    """

    def buffer_inertial_data(self, message: Dict, publisher_id: int):
        """
        Flight State here is only the inertial data
        This is the first step to update the sensor data
        it receives the inertial data from the publisher and stores it in the buffer
        the publisher is triggered when imu sensor is updated, in the quacopter class
        """
        if "termination" in message:
            self.buffer_manager.clear_buffer_data(publisher_id)

        elif publisher_id == self.parent_id:
            self.parent_inertia = message
        else:
            self.buffer_manager.buffer_message(message, publisher_id)

    def calculate_extremity_points(self, position, dimensions):
        half_dimensions = np.array(dimensions) / 2
        return {
            "top": position + [0, 0, half_dimensions[2]],
            "bottom": position - [0, 0, half_dimensions[2]],
            "left": position - [half_dimensions[0], 0, 0],
            "right": position + [half_dimensions[0], 0, 0],
            "front": position + [0, half_dimensions[1], 0],
            "back": position - [0, half_dimensions[1], 0],
        }

    def process_message(self, message):
        """
        This function is called by the sensor manager
        it fetch tha data from the buffer
        """
        position = message.get("position")
        dimensions = message.get("dimensions")

        if dimensions is None:
            return position, None

        # print(dimensions)
        extremity_points = self.calculate_extremity_points(position, dimensions)
        return position, extremity_points

    def update_data(self) -> None:
        buffer_data = self.buffer_manager.get_all_data()
        self.reset()

        for target_id, message in buffer_data.items():
            position = message.get("position")

            position, extremity_points = self.process_message(message)
            # print(message.get("publisher_type"))
            self._add_end_position_for_entity(
                position, message.get("publisher_type"), target_id
            )

    def read_data(self) -> Dict:
        return {"lidar": self.sphere}

    # ============================================================================================================
    # Normalizations Functions
    # ============================================================================================================

    def _normalize_angle(self, angle, initial_angle, final_angle, n_points) -> float:
        # linear interpolation
        return (
            round((angle - initial_angle) / (final_angle - initial_angle) * n_points)
            % n_points
        )

    def _normalize_theta(self, theta) -> float:
        return self._normalize_angle(
            theta,
            self.THETA_INITIAL_RADIAN,
            self.THETA_FINAL_RADIAN,
            self.n_theta_points,
        )

    def _normalize_phi(self, phi) -> float:
        return self._normalize_angle(
            phi, self.PHI_INITIAL_RADIAN, self.PHI_FINAL_RADIAN, self.n_phi_points
        )

    def _normalize_distance(self, distance, radius) -> float:
        return distance / radius

    def _normalize_spherical(self, spherical: np.ndarray) -> Tuple[float, float, float]:
        distance, theta, phi = spherical
        norm_distance = self._normalize_distance(distance, self.radius)
        theta_point = self._normalize_theta(theta)
        phi_point = self._normalize_phi(phi)

        return norm_distance, theta_point, phi_point

    def debug_sphere(self):
        current_position = self.parent_inertia["position"]
        print(f"Sphere_Shape = {self.sphere.shape} \n Sphere \n {self.sphere}")
        for theta_point in range(len(self.sphere[Channels.DISTANCE_CHANNEL.value])):
            for phi_point in range(
                len(self.sphere[Channels.DISTANCE_CHANNEL.value][theta_point])
            ):
                distance = self.sphere[Channels.DISTANCE_CHANNEL.value][theta_point][
                    phi_point
                ]
                if distance < 1:
                    end_position = self.convert_angle_point_to_cartesian(
                        theta_point=theta_point,
                        phi_point=phi_point,
                        n_theta_points=self.n_theta_points,
                        n_phi_points=self.n_phi_points,
                        distance=distance,
                        current_position=current_position,
                    )
                    if self.debug_line_id == -1:
                        self.debug_line_id = p.addUserDebugLine(
                            lineFromXYZ=current_position,
                            lineToXYZ=end_position,
                            lineColorRGB=[1, 0, 0],
                            lineWidth=3,
                            lifeTime=1,
                            physicsClientId=self.client_id,
                        )
                    else:
                        self.debug_line_id = p.addUserDebugLine(
                            lineFromXYZ=current_position,
                            lineToXYZ=end_position,
                            lineColorRGB=[1, 0, 0],
                            lineWidth=3,
                            lifeTime=1,
                            replaceItemUniqueId=self.debug_line_id,
                            physicsClientId=self.client_id,
                        )

    def convert_angle_point_to_cartesian(
        self,
        theta_point,
        phi_point,
        n_theta_points,
        n_phi_points,
        distance,
        current_position,
    ) -> np.ndarray:
        theta, phi = self.convert_angle_point_to_angles(
            theta_point, phi_point, n_theta_points, n_phi_points
        )
        cartesian_from_origin = CoordinateConverter.spherical_to_cartesian(
            np.array([distance * self.radius, theta, phi])
        )
        return cartesian_from_origin + current_position
