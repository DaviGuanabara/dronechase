import numpy as np
import pybullet as p

from typing import Dict, Optional
from enum import Enum, auto

from ....events.message_hub import MessageHub
from ..sensors.lidar import LiDAR
from ..sensors.imu import InertialMeasurementUnit

from ..dataclasses.flight_state import (
    FlightStateManager,
    FlightStateDataType,
)
from PyFlyt.core.drones.quadx import QuadX
from pybullet_utils.bullet_client import BulletClient

from .quadcopter_type import QuadcopterType


class Quadcopter:
    INVALID_ID = -1

    def __init__(
        self,
        quadx: QuadX,
        simulation: BulletClient,
        quadcopter_type: QuadcopterType = QuadcopterType.QUADCOPTER,
        quadcopter_name: Optional[str] = "",
        debug_on: bool = False,
        lidar_on: bool = False,
        lidar_radius:float = 5,
        lidar_resolution:int = 16
    ):
        self.quadx = quadx
        self.id: int = self.quadx.Id
        self.debug_on = debug_on
        self.lidar_on = lidar_on
        self.simulation = simulation
        self.client_id = simulation._client

        self.quadcopter_type = quadcopter_type
        self.quadcopter_name = quadcopter_name

        self.flight_state_manager: FlightStateManager = FlightStateManager()
        self.imu = InertialMeasurementUnit(quadx)
        self.lidar = LiDAR(self.id, self.client_id, radius = 5, resolution = 16)

        self.messageHub = MessageHub()
        self.messageHub.subscribe(
            topic="inertial_data", subscriber=self._subscriber_inertial_data
        )

    # =================================================================================================================
    # Communication
    # =================================================================================================================

    def _publish_inertial_data(self):
        """
        Publish the current flight state data to the message hub.
        """
        inertial_data = self.flight_state_manager.get_inertial_data()

        message = {**inertial_data, "publisher_type": self.quadcopter_type}
        #print(message)
        self.messageHub.publish(
            topic="inertial_data", message=message, publisher_id=self.id
        )

    def _subscriber_inertial_data(self, message: Dict, publisher_id: int):
        """
        Handle incoming flight state data from the message hub.

        Parameters:
        - flight_state (Dict): The flight state data received.
        - publisher_id (int): The ID of the publisher of the data.
        """
        #print(f"Received data from {publisher_id}")
        if self.lidar_on:
            self.lidar.buffer_inertial_data(message, publisher_id)

    # =================================================================================================================
    # Spawn
    # =================================================================================================================

    @staticmethod
    def spawn_loyalwingman(
        simulation: BulletClient,
        position: np.ndarray,
        quadcopter_name: Optional[str] = "",
        physics_hz: int = 240,
        np_random: np.random.RandomState = np.random.RandomState(),
        debug_on: bool = False,
        lidar_on: bool = False,
        lidar_radius:float = 5,
    ):
        quadx = QuadX(
            simulation,
            start_pos=position,
            start_orn=np.zeros(3),
            physics_hz=physics_hz,
            np_random=np_random,
            # **drone_options,
        )
        quadx.reset()
        quadx.set_mode(6)

        return Quadcopter(
            simulation=simulation,
            quadx=quadx,
            quadcopter_name=quadcopter_name,
            quadcopter_type=QuadcopterType.LOYALWINGMAN,
            debug_on=debug_on,
            lidar_on=lidar_on,
            lidar_radius=lidar_radius
        )

    @staticmethod
    def spawn_loiteringmunition(
        simulation: BulletClient,
        position: np.ndarray,
        quadcopter_name: Optional[str] = "",
        physics_hz: int = 240,
        np_random: np.random.RandomState = np.random.RandomState(),
        debug_on: bool = False,
        lidar_on: bool = False,
    ):
        quadx = QuadX(
            simulation,
            start_pos=position,
            start_orn=np.zeros(3),
            physics_hz=physics_hz,
            np_random=np_random,
            # **drone_options,
        )
        quadx.reset()
        quadx.set_mode(6)

        return Quadcopter(
            simulation=simulation,
            quadx=quadx,
            quadcopter_name=quadcopter_name,
            quadcopter_type=QuadcopterType.LOITERINGMUNITION,
            debug_on=debug_on,
            lidar_on=lidar_on
        )

    # =================================================================================================================
    # Position and Orientation
    # =================================================================================================================

    def replace(self, position: np.ndarray, attitude: np.ndarray):
        quaternion = self.simulation.getQuaternionFromEuler(attitude)
        self.simulation.resetBasePositionAndOrientation(self.id, position, quaternion)
        #self.update_imu()

    # =================================================================================================================
    # Sensors and Flight State
    # =================================================================================================================

    def update_imu(self):
        #print(f"update_imu{self.id}")
        self.update_state()

    def update_lidar(self):
        self.lidar.update_data()
        lidar_data = self.lidar.read_data()
        self._update_flight_state(lidar_data)
        if self.debug_on:
            self.lidar.debug_sphere()
        # Note: We don't publish the flight state here

    @property
    def lidar_shape(self) -> tuple:
        return self.lidar.get_data_shape()

    def _update_flight_state(self, sensor_data: Dict):
        self.flight_state_manager.update_data(sensor_data)

    @property
    def inertial_data(self):
        return self.flight_state_manager.get_data_by_type(FlightStateDataType.INERTIAL)

    @property
    def lidar_data(self):
        return self.flight_state_manager.get_data_by_type(FlightStateDataType.LIDAR)

    # def reset_flight_state(self):
    #    self.flight_state_manager = FlightStateManager()

    # =================================================================================================================
    # Actuators
    # =================================================================================================================

    def convert_command_to_setpoint(self, command: np.ndarray):
        """
        Convert the given command to a setpoint that can be used by the quadcopter's propulsion system.
        Parameters:
        - command: The command to be converted. It is composed by:
            - direction (3d) and magnitude (1d) of the desired velocity in the x, y, and z axes.
        Returns:
        - The converted setpoint.
        """

        raw_direction = command[:3]
        raw_direction_norm = np.linalg.norm(raw_direction)
        direction = raw_direction / (
            raw_direction_norm if raw_direction_norm > 0 else 1
        )
        magnutide = command[3]
        vx, vy, vz = magnutide * direction
        return np.array([vx, vy, 0, vz])

    def drive(self, motion_command: np.ndarray, show_name_on: bool = False):
        """
        Apply the given motion command to the quadcopter's propulsion system.
        Parameters:
        - motion_command: The command to be applied. This could be RPM values, thrust levels, etc.
        """

        if show_name_on:
            self.show_name()

        # Remmeber that was chosen mdode 6:
        # - 6: vx, vy, vr, vz
        # - vp, vq, vr = angular velocities
        # - vx, vy, vz = ground linear velocities
        self.quadx.setpoint = self.convert_command_to_setpoint(motion_command)

    # =================================================================================================================
    # Delete or Destroy
    # =================================================================================================================

    def detach_from_simulation(self):
        self.messageHub.terminate(self.id)
        self.simulation.removeBody(self.id)
        self.id = Quadcopter.INVALID_ID

        if hasattr(self, "text_id"):
            self.simulation.removeUserDebugItem(self.text_id)

    def show_name(self):
        quad_position = self.flight_state_manager.get_data("position").get("position")

        if quad_position is None:
            return
        # Place the text slightly below the quadcopter

        text_position = np.array(
            [quad_position[0], quad_position[1], quad_position[2] - 0.2]
        )
        # Add the text, and store its ID for later reference
        textColorRGB = [0, 1, 0]
        if self.quadcopter_type == QuadcopterType.LOYALWINGMAN:
            textColorRGB = [0, 0, 1]

        if self.quadcopter_type == QuadcopterType.LOITERINGMUNITION:
            textColorRGB = [1, 0, 0]

        if hasattr(self, "text_id"):
            self.text_id = self.simulation.addUserDebugText(
                self.quadcopter_name,
                text_position,
                textSize=1,
                textColorRGB=textColorRGB,
                replaceItemUniqueId=self.text_id,
            )

        else:
            self.text_id = self.simulation.addUserDebugText(
                self.quadcopter_name,
                text_position,
                textSize=0.1,
                textColorRGB=[1, 0, 0],
            )

    # =================================================================================================================
    # Quadx Commom Methods
    # =================================================================================================================

    @property
    def control_period(self):
        if hasattr(self, "quadx") and self.quadx is not None:
            return self.quadx.control_period

    def update_control(self):
        if hasattr(self, "quadx") and self.quadx is not None:
            self.quadx.update_control()

    def update_physics(self):
        if hasattr(self, "quadx") and self.quadx is not None:
            self.quadx.update_physics()

    def reset(self):
        if hasattr(self, "quadx") and self.quadx is not None:
            self.quadx.reset()

    def update_state(self):
        """
        Update the inertial data of the quadcopter.
        Only inertial data. Lidar is not considered here.
        """
        #self.quadx.update_state()

        self.imu.update_data()
        imu_data = self.imu.read_data()
        self._update_flight_state(imu_data)
        self._publish_inertial_data()

    def update_last(self):
        self.quadx.update_last()

    @property
    def physics_control_ratio(self):
        return self.quadx.physics_control_ratio

    def set_mode(self, mode: int):
        self.quadx.set_mode(mode)

    # @property
    # def state(self):
    #    return self.quadx.state

    # @property
    # def setpoint(self):
    #    return self.quadx.setpoint

    def set_setpoint(self, setpoint):
        self.quadx.setpoint = setpoint

    @property
    def max_speed(self):
        """
        Quadcopter Parameters from cf2x.urdf em loyalwingmen assets
        An interesting fact is that this parameters are not seem in the cf2x.urdf PyFlyt file.
        They got an update where the final values needed to calculations are already there,
        and not the parameters that are used to calculate them, like propeller radius, etc.
        """
        speed_modulator = 1
        max_speed_kmh = 30
        KMH_TO_MS = 1000 / 3600
        return speed_modulator * max_speed_kmh * KMH_TO_MS
