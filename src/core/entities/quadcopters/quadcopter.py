import numpy as np
import pybullet as p  # type: ignore

from typing import Dict, Optional
from enum import Enum, auto

from PyFlyt.core.drones.quadx import QuadX  # type: ignore
from pybullet_utils.bullet_client import BulletClient  # type: ignore


# =================================================================================================================
# Components
# =================================================================================================================

from core.dataclasses.message_context import MessageContext
from core.entities.quadcopters.components import LIDAR, FusedLIDAR, InertialMeasurementUnit, FlightStateDataType, FlightStateManager, Gun
from core.entities.entity_type import EntityType
from core.entities.quadcopters.components.sensors.interfaces.base_lidar import BaseLidar
from core.notification_system import TopicsEnum, MessageHub

# TODO: remove subscrition of current step of the components, i should centralize this here (at quadcopter class).
# TODO: make function called hit, it will be used instead of disarm.


class Quadcopter:
    INVALID_ID = -1

    def __init__(
        self,
        quadx: QuadX,
        simulation: BulletClient,
        lidar:Optional[BaseLidar] = None,
        quadcopter_type: EntityType = EntityType.QUADCOPTER,
        quadcopter_name: Optional[str] = "",
        debug_on: bool = False,
        lidar_on: bool = False,
        #lidar_radius: float = 5,
        #lidar_resolution: int = 16,
        shoot_range: float = 0.8,
        formation_position: np.ndarray = np.array([0, 0, 0]),
        
    ):
        
        self.step = 0
        self.quadx = quadx
        self.formation_position = formation_position

        self.id: int = self.quadx.Id
        self.debug_on = debug_on
        self.lidar_on = lidar_on
        self.simulation = simulation
        self.client_id = simulation._client

        self.quadcopter_type = quadcopter_type
        self.quadcopter_name = quadcopter_name

        # print("init components")
        self.flight_state_manager: FlightStateManager = FlightStateManager()
        self.imu = InertialMeasurementUnit(quadx)
        
        if lidar is None:
            self.lidar_on = False
        else:
            self.lidar = lidar
        #lidar_class(
         #   self.id, self.client_id, radius=lidar_radius, resolution=lidar_resolution
        #)
        self.setup_gun(self.id, shoot_range)

        self.setup_message_hub()

        # print("init components done")

        self.init_physical_properties()
        self.update_color()

        self._is_agent = False

    def init_physical_properties(self):
        # print("getting dynamics info")
        self.dynamics_info = self.simulation.getDynamicsInfo(self.id, -1)

        self.shape_data = self.simulation.getVisualShapeData(self.id)[0]
        self.dimensions = self.shape_data[3]

        self.num_joints = self.simulation.getNumJoints(self.id)

        self.joint_indices = list(range(-1, self.num_joints))
        self.mass = self.dynamics_info[0]
        self._armed = True

    def update_color(self):
        Red = [1, 0, 0, 1]
        LightRed = [1, 0.5, 0.5, 1]
        
        if self.is_agent:
            if self.armed:
                # Green for armed agent
                Green = [0, 1, 0, 1]
                p.changeVisualShape(self.id, -1, rgbaColor=Green)

            else:
                LightGreen = [0.5, 1, 0.5, 1]
                p.changeVisualShape(self.id, -1, rgbaColor=LightGreen)

        elif self.quadcopter_type == EntityType.LOYALWINGMAN:
            if self.armed:
                DarkBlue = [0, 0, 0.8, 1]
                p.changeVisualShape(
                    self.id, -1, rgbaColor=DarkBlue
                )  # Green for armed pursuer
            else:
                LightBlue = [0.5, 0.5, 1, 1]
                p.changeVisualShape(
                    self.id, -1, rgbaColor=LightBlue
                )  # Light Green for disarmed pursuer

        elif self.quadcopter_type == EntityType.LOITERINGMUNITION:
            if self.armed:
                # Red for armed invader
                p.changeVisualShape(self.id, -1, rgbaColor=Red)
            else:
                p.changeVisualShape(
                    self.id, -1, rgbaColor=LightRed
                )  # Light Red or Pink for disarmed invader

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
        lidar_radius: float = 5,
        use_fused_lidar: bool = True,
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

        # print(f"Lidar Radius: {lidar_radius}")
        lidar_resolution = 16

        #lidar_class = FusedLIDAR if use_fused_lidar else LIDAR
        #lidar = lidar_class(quadx.Id, simulation._client, radius=lidar_radius,resolution=lidar_resolution,debug=debug_on)
        lidar = FusedLIDAR(parent_id=quadx.Id, client_id=simulation._client, debug=debug_on, radius=lidar_radius, resolution=lidar_resolution)



        return Quadcopter(
            simulation=simulation,
            quadx=quadx,
            quadcopter_name=quadcopter_name,
            quadcopter_type=EntityType.LOYALWINGMAN,
            debug_on=debug_on,
            lidar_on=lidar_on,
            #lidar_radius=lidar_radius,
            formation_position=position,
            lidar=lidar
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
        lidar_radius: float = 5,
    ):
        #TODO: I WILL TREAT LIDAR ALWAYS OFF FOR LOITERING MUNITION
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

        # logic exclusive for loyalwingmen
        # lidar_class = FusedLidar if use_fused_lidar else LiDAR

        return Quadcopter(
            simulation=simulation,
            quadx=quadx,
            quadcopter_name=quadcopter_name,
            quadcopter_type=EntityType.LOITERINGMUNITION,
            debug_on=debug_on,
            lidar_on=lidar_on,
            formation_position=position,
        )

    # =================================================================================================================
    # Weapons
    # =================================================================================================================

    """
    First, the gun is set to shoot at a certain range.
    """

    def setup_gun(self, id, shoot_range):
        # def setup_gun(self, id, munition, shoot_range):
        # Retirei o munition como agumento do Gun para poder alterar somente na própria classe a quantidade de munição
        # self.gun = Gun(id, munition=munition, shoot_range=shoot_range)
        # ficando assim mais rápido.
        # munition
        self.gun = Gun(id, shoot_range=shoot_range)
        self.shoot_range = shoot_range

    def shoot(self):
        return self.gun.shoot()

    # =================================================================================================================
    # Communication
    # =================================================================================================================

    # TODO: Make a communication class
    def setup_message_hub(self):
        self.messageHub = MessageHub()
        self.messageHub.subscribe(
            topic=TopicsEnum.INERTIAL_DATA_BROADCAST,
            subscriber=self._subscriber_inertial_data,
        )

        if self.lidar_on:
            self.messageHub.subscribe(
                topic=TopicsEnum.LIDAR_DATA_BROADCAST,
                subscriber=self._subscriber_lidar_data
            )

        self.messageHub.subscribe(
            topic=TopicsEnum.AGENT_STEP_BROADCAST,
            subscriber=self._subscriber_broadcast_step
        )

    def _publish(self, topic:TopicsEnum, message: Dict):
        message_context = self.messageHub.create_message_context(publisher_id=self.id, step = self.step, entity_type=self.quadcopter_type)
        self.messageHub.publish(topic=topic, message=message, message_context=message_context)


    def _publish_lidar_data(self, message: Dict):
        #print(f"Publishing lidar data for Quadcopter {self.id} on topic {TopicsEnum.LIDAR_DATA_BROADCAST.name}")
        
        self._publish(topic=TopicsEnum.LIDAR_DATA_BROADCAST, message=message)

    def _publish_inertial_data(self):
        """
        Publish the current flight state data to the message hub.
        All publishment from quadcopter sensors must have the publihser_type
        """
        inertial_data = self.flight_state_manager.get_inertial_data()
        inertial_data["dimensions"] = self.dimensions

        message = {**inertial_data, "publisher_type": self.quadcopter_type}
        self._publish(topic=TopicsEnum.INERTIAL_DATA_BROADCAST, message=message)


    def _subscriber_lidar_data(self, message: Dict, message_context: MessageContext):
        """
        Handle incoming LiDAR data from the message hub.

        Parameters:
        - message (Dict): The LiDAR data received.
        - publisher_id (int): The ID of the publisher of the data.
        """
        # print(f"Received lidar data from {publisher_id}")
        if self.lidar_on:
            self.lidar.buffer_lidar_data(message, message_context)

    
    def _subscriber_inertial_data(self, message: Dict, message_context: MessageContext):
        """
        Handle incoming flight state data from the message hub.

        Parameters:
        - flight_state (Dict): The flight state data received.
        - publisher_id (int): The ID of the publisher of the data.
        """
        # print(f"Received inertial data from {publisher_id}")
        if self.lidar_on:
            self.lidar.buffer_inertial_data(message, message_context)

    def _subscriber_broadcast_step(self, message: Dict, message_context: MessageContext):
        step = message.get("step")
        if step is not None:
            self.step = step
        #TODO: update_step in BaseLidar
        if self.lidar_on:
            self.lidar.buffer_step_broadcast(message, message_context)

    # =================================================================================================================
    # Sensors and Flight State
    # =================================================================================================================

    def update_imu(self):
        # print(f"update_imu{self.id}")
        self.update_state()

    def update_lidar(self):
        # print("updating lidar")
        #if self.quadcopter_type == EntityType.LOYALWINGMAN:
            #print(f"[DEBUG] LoyalWingman {self.id}: Should update its LiDAR data. Lidar is {'on' if self.lidar_on else 'off'}")
        if self.lidar_on:
   
            self.lidar.update_data()
            lidar_data = self.lidar.read_data()
            if not hasattr(self, "_printed_once_read"):
                print(f"[DEBUG] quadcopter read_data keys: {list(lidar_data.keys())}")
                self._printed_once_read = True
            
            self._update_flight_state(lidar_data)
            self._publish_lidar_data(lidar_data)
        # if self.debug_on:
        #    self.lidar.debug_sphere()
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

    # TODO: is gun available ?
    # return self.gun.is_available()

    def set_munition(self, munition):
        self.gun.set_munition(munition)

    @property
    def is_gun_available(self):
        return self.gun.is_available()

    @property
    def is_munition_available(self):
        return self.gun.has_munition()

    @property
    def gun_state(self) -> np.ndarray:
        return self.gun.get_state()

    @property
    def gun_state_shape(self) -> tuple:
        return self.gun.get_state_shape()

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
        self.last_motion_command = motion_command
        self.quadx.setpoint = self.convert_command_to_setpoint(motion_command)

    @property
    def last_action(self):
        if hasattr(self, "last_motion_command"):
            return self.last_motion_command
        return np.zeros(4)

    # =================================================================================================================
    # Simulation Manager
    # =================================================================================================================

    def detach_from_simulation(self):
        self.messageHub.terminate(self.id)
        self.simulation.removeBody(self.id)
        self.id = Quadcopter.INVALID_ID

        if hasattr(self, "text_id"):
            self.simulation.removeUserDebugItem(self.text_id)

    def replace(self, position: np.ndarray, attitude: np.ndarray):
        quaternion = self.simulation.getQuaternionFromEuler(attitude)
        self.simulation.resetBasePositionAndOrientation(
            self.id, position, quaternion)
        self.formation_position = position
        if self._armed:
            self.update_imu()

    @property
    def armed(self):
        return self._armed

    def arm(self):
        self.simulation.changeDynamics(self.id, -1, mass=self.mass)
        self._setCollisionState(True)
        self._armed = True

        # print(f"Q{self.id} - updating imu")
        self.update_imu()
        # print("updating its color")
        self.update_color()
        # print("Armed")

        # self.reset()

        # self.quadx.reset()
        self.gun.reset()

    def disarm(self):
        self.simulation.changeDynamics(self.id, -1, mass=0)
        self._setCollisionState(False)

        self.simulation.resetBaseVelocity(
            self.id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]
        )

        self._armed = False
        self.messageHub.terminate(self.id)
        self.update_color()

        # self.gun.reset()
        self.quadx.body.reset()
        self.quadx.motors.reset()
        self.quadx.setpoint = np.zeros(4)
        self.quadx.pwm = np.zeros(4)
        self.quadx.disable_artificial_damping()

    def hit(self):
        """TODO: life managemente should be here"""
        self.disarm()

    def _setCollisionState(self, activate):
        """
        Set the collision state for all joints of the entity.

        :param activate: A boolean value. If True, collisions are activated. If False, collisions are deactivated.
        """
        group = int(activate)  # 0  # other objects don't collide with me
        mask = int(
            activate
        )  # 0: don't collide with any other object, 1: collide with any other object

        for joint_indice in self.joint_indices:
            p.setCollisionFilterGroupMask(self.id, joint_indice, group, mask)

    def show_name(self):
        quad_position = self.flight_state_manager.get_data(
            "position").get("position")

        if quad_position is None:
            return
        # Place the text slightly below the quadcopter

        text_position = np.array(
            [quad_position[0], quad_position[1], quad_position[2] - 0.2]
        )
        # Add the text, and store its ID for later reference
        textColorRGB = [0, 1, 0]
        if self.quadcopter_type == EntityType.LOYALWINGMAN:
            textColorRGB = [0, 0, 1]

        if self.quadcopter_type == EntityType.LOITERINGMUNITION:
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

        self.gun.reset()

    def update_state(self):
        """
        Update the inertial data of the quadcopter.
        Only inertial data. Lidar is not considered here.
        """

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
        max_speed_kmh = 10
        KMH_TO_MS = 1000 / 3600
        return speed_modulator * max_speed_kmh * KMH_TO_MS

    def set_as_agent(self):
        """
        Marks this quadcopter as the agent.
        It updates internal components accordingly (e.g., LiDAR fusion).
        """
        #print("drone set as agent")
        #print(f"lidar is Fused ? {isinstance(self.lidar, FusedLIDAR)}")
        if isinstance(self.lidar, FusedLIDAR):
            #TODO: IN MAINTANANCE
            self.lidar.enable_fusion()

        self._is_agent = True
        self.quadcopter_name = f"Agent_{self.id}"

    @property
    def is_agent(self) -> bool:
        return getattr(self, "_is_agent", False)
