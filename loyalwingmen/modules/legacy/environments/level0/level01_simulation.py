import pybullet as p
import pybullet_data
import numpy as np
import random
from typing import Tuple, Optional, Union, Dict

from ...quadcoters.quadcopter_factory import (
    QuadcopterFactory,
    Quadcopter,
    QuadcopterType,
    LoyalWingman,
    LoiteringMunition,
    OperationalConstraints,
    FlightStateDataType,
    LoiteringMunitionBehavior,
    CommandType,
)

from ..helpers.normalization import normalize_inertial_data

from ..helpers.environment_parameters import EnvironmentParameters
from time import time


class PIDDOCARASimulator:
    def __init__(
        self,
        dome_radius: float,
        environment_parameters: EnvironmentParameters,
    ):
        self.dome_radius = dome_radius

        self.environment_parameters = environment_parameters
        print("level 1 simulation")
        self.init_simulation()

    def init_simulation(self):
        """Initialize the simulation and entities."""
        # Initialize pybullet, load plane, gravity, etc.

        if self.environment_parameters.GUI:
            client_id = self.setup_pybulley_GUI()

        else:
            client_id = self.setup_pybullet_DIRECT()

        p.setGravity(
            0,
            0,
            -self.environment_parameters.G,
            physicsClientId=client_id,
        )

        p.setRealTimeSimulation(0, physicsClientId=client_id)  # No Realtime Sync

        p.setTimeStep(
            self.environment_parameters.timestep,
            physicsClientId=client_id,
        )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=client_id,
        )

        self.environment_parameters.client_id = client_id
        self.factory = QuadcopterFactory(self.environment_parameters)

        position, ang_position = self.gen_initial_position()

        self.loyal_wingman: LoyalWingman = self.factory.create_loyalwingman(
            position,
            np.zeros(3),
            CommandType.VELOCITY_TO_CONTROLLER,
            "Agent",
            quadcopter_role="Interceptor",
        )

        self.loitering_munition: LoiteringMunition = (
            self.factory.create_loiteringmunition(
                np.zeros(3),
                np.zeros(3),
                quadcopter_name="target",
                quadcopter_role="Infiltrator",
            )
        )

        self.loitering_munition.set_behavior(LoiteringMunitionBehavior.FROZEN)

        self.last_action: np.ndarray = np.zeros(4)

        self.reset()

    def setup_pybullet_DIRECT(self):
        return p.connect(p.DIRECT)

    def setup_pybulley_GUI(self):
        client_id = p.connect(p.GUI)
        for i in [
            p.COV_ENABLE_RGB_BUFFER_PREVIEW,
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        ]:
            p.configureDebugVisualizer(i, 0, physicsClientId=client_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=3,
            cameraYaw=-30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=client_id,
        )
        ret = p.getDebugVisualizerCamera(physicsClientId=client_id)

        return client_id

    def _reset_simulation(self):
        """Housekeeping function.
        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.
        """

        #### Set PyBullet's parameters #############################

        p.resetSimulation(physicsClientId=self.environment_parameters.client_id)

    def gen_initial_position(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random position within the dome."""

        position = np.random.rand(3)  # * (self.dome_radius / 2)
        ang_position = np.random.uniform(0, 2 * np.pi, 3)

        return position, ang_position

    def _housekeeping(self):
        pos, ang_pos = self.gen_initial_position()

        target_pos, target_ang_pos = np.array([0, 0, 0]), np.array([0, 0, 0])

        if self.loyal_wingman is not None:
            self.loyal_wingman.detach_from_simulation()

        if self.loitering_munition is not None:
            self.loitering_munition.detach_from_simulation()

        self.loyal_wingman = self.factory.create_loyalwingman(
            position=pos,
            ang_position=np.zeros(3),
            command_type=CommandType.VELOCITY_TO_CONTROLLER,
            quadcopter_name="agent",
            quadcopter_role="Interceptor",
        )
        self.loitering_munition = self.factory.create_loiteringmunition(
            position=target_pos,
            ang_position=target_ang_pos,
            quadcopter_name="target",
            quadcopter_role="Infiltrator",
        )

        self.loyal_wingman.update_imu()
        self.loitering_munition.update_imu()

        if self.environment_parameters.GUI:
            self.loyal_wingman.show_name()

        self.loitering_munition.set_behavior(LoiteringMunitionBehavior.FROZEN)
        self.start_time = time()

        lw_flight_state = self.loyal_wingman.flight_state
        lm_flight_state = self.loitering_munition.flight_state

        distance_vector = self._calculate_distance_vector(
            lw_flight_state, lm_flight_state
        )
        self.last_distance = np.linalg.norm(distance_vector)

    def reset(self):
        """Reset the simulation to its initial state."""

        # self._reset_simulation()

        self._housekeeping()

        observation = self.compute_observation()
        info = self.compute_info()

        return observation, info

    def compute_observation(self) -> np.ndarray:
        """Return the observation of the simulation."""

        lw = self.loyal_wingman
        lm = self.loitering_munition
        # assert lw is not None, "Loyal wingman is not initialized"
        # assert lm is not None

        lw_state = lw.flight_state_by_type(FlightStateDataType.INERTIAL)
        lm_state = lm.flight_state_by_type(FlightStateDataType.INERTIAL)

        distance_vector = self._calculate_distance_vector(lw_state, lm_state)

        distance_to_target = np.linalg.norm(distance_vector)
        direction_to_target = distance_vector / max(float(distance_to_target), 1)

        distance_to_target_normalized = distance_to_target / (2 * self.dome_radius)
        lw_inertial_data_normalized = normalize_inertial_data(
            lw_state, lw.operational_constraints, self.dome_radius
        )
        lm_inertial_data_normalized = normalize_inertial_data(
            lm_state, lm.operational_constraints, self.dome_radius
        )

        lw_values = self.convert_dict_to_array(lw_inertial_data_normalized)
        lm_values = self.convert_dict_to_array(lm_inertial_data_normalized)

        return np.array(
            [
                *lw_values,
                *lm_values,
                *direction_to_target,
                distance_to_target_normalized,
                *self.last_action,
            ],
            dtype=np.float32,
        )

    def distance(self, lw_state: Dict, lm_state: Dict) -> float:
        distance_vector = self._calculate_distance_vector(lw_state, lm_state)
        return float(np.linalg.norm(distance_vector))

    def observation_size(self):
        position = 3
        velocity = 3
        attitude = 3
        angular_rate = 3
        acceleration = 3
        angular_acceleration = 3

        lw_inertial_data_shape = (
            position
            + velocity
            + attitude
            + angular_rate
            + acceleration
            + angular_acceleration
        )
        lm_inertial_data_shape = (
            position
            + velocity
            + attitude
            + angular_rate
            + acceleration
            + angular_acceleration
        )

        direction_to_target_shape = 3
        distance_to_target_shape = 1
        last_action_shape = 4

        return (
            lw_inertial_data_shape
            + lm_inertial_data_shape
            + direction_to_target_shape
            + distance_to_target_shape
            + last_action_shape
        )

    def compute_termination(self) -> bool:
        """Calculate if the simulation is done."""

        end_time = time()
        elapsed_time = end_time - self.start_time
        max_episode_time = self.environment_parameters.max_episode_time

        if max_episode_time > 0 and elapsed_time > max_episode_time:
            return True

        if self.is_outside_dome(self.loitering_munition):
            return True

        return bool(self.is_outside_dome(self.loyal_wingman))

    def step(self, rl_action: np.ndarray):
        """Execute a step in the simulation based on the RL action."""

        # assert self.loyal_wingman is not None, "Loyal wingman is not initialized"
        # assert self.loitering_munition is not None

        self.last_action = rl_action

        for _ in range(self.environment_parameters.aggregate_physics_steps):
            self.loyal_wingman.update_imu()
            self.loitering_munition.update_imu()
            self.loyal_wingman.drive(rl_action)
            self.loitering_munition.drive_via_behavior()
            p.stepSimulation()

        self.loyal_wingman.update_imu()
        self.loitering_munition.update_imu()

        self.loyal_wingman.update_lidar()
        self.loitering_munition.update_lidar()

        lw_inertial_data = self.loyal_wingman.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )
        lm_inertial_data = self.loitering_munition.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )

        observation = self.compute_observation()
        reward = self.compute_reward()
        terminated = self.compute_termination()

        truncated = self.compute_truncation()
        info = self.compute_info()

        if self.distance(lw_inertial_data, lm_inertial_data) < 0.2:
            self.replace_loitering_munition()

        return observation, reward, terminated, truncated, info

    def compute_truncation(self):
        return False

    def compute_info(self):
        return {}

    def compute_reward(self):
        # assert self.loyal_wingman is not None, "Loyal wingman is not initialized"
        # assert self.loitering_munition is not None

        bonus = 0
        penalty = 0
        score = 0

        lw_flight_state = self.loyal_wingman.flight_state
        lm_flight_state = self.loitering_munition.flight_state

        distance_vector = self._calculate_distance_vector(
            lw_flight_state, lm_flight_state
        )
        distance = np.linalg.norm(distance_vector)

        velocity = lw_flight_state.get("velocity", np.zeros(3))

        if distance < self.last_distance:
            direction = self.last_distance - distance
            direction_normalized = (
                direction / np.linalg.norm(direction)
                if np.linalg.norm(direction) > 0
                else np.zeros(3)
            )
            component_in_direction = (
                np.dot(velocity, direction_normalized) * direction_normalized
            )
            bonus += 10 * np.linalg.norm(component_in_direction)

        score = self.dome_radius - distance

        if distance < 0.2:
            bonus += 1_000

        if distance > self.dome_radius:
            penalty += 1_000

        # print(lw_flight_state.get("position", np.zeros(3)), lm_flight_state.get("position", np.zeros(3)), distance, score + bonus - penalty)
        return score + bonus - penalty

    def _calculate_distance_vector(self, lw_state: Dict, lm_state: Dict) -> np.ndarray:
        lw_pos = lw_state.get("position", np.zeros(3))
        lm_pos = lm_state.get("position", np.zeros(3))
        return lm_pos - lw_pos

    def is_outside_dome(self, entity: Quadcopter) -> bool:
        """Check if the entity is outside the dome radius."""
        inertial_data = entity.flight_state_by_type(FlightStateDataType.INERTIAL)
        position = inertial_data.get("position", np.zeros(3))

        return float(np.linalg.norm(position)) > self.dome_radius

    def replace_loitering_munition(self):
        """Reset the loitering munition to a random position within the dome."""

        position, ang_position = self.gen_initial_position()

        # assert self.loitering_munition is not None
        # self.loitering_munition.detach_from_simulation()
        # self.loitering_munition = self.factory.create_loiteringmunition(
        #    position=position, ang_position=ang_position
        # )

        quaternion = p.getQuaternionFromEuler(ang_position)
        p.resetBasePositionAndOrientation(
            self.loitering_munition.id,
            position,
            quaternion,
            physicsClientId=self.environment_parameters.client_id,
        )

        self.loitering_munition.update_imu()
        behaviors = [
            LoiteringMunitionBehavior.STRAIGHT_LINE,
            LoiteringMunitionBehavior.CIRCLE,
            LoiteringMunitionBehavior.FROZEN,
        ]
        random_behavior = random.choice(behaviors)
        self.loitering_munition.set_behavior(random_behavior)

    def convert_dict_to_array(self, dictionary: Dict) -> np.ndarray:
        array = np.array([])
        for key in dictionary:
            value: np.ndarray = (
                dictionary[key] if dictionary[key] is not None else np.array([])
            )
            array = np.concatenate((array, value))

        return array

    def close(self):
        p.disconnect(physicsClientId=self.environment_parameters.client_id)

    def _physics(
        self,
        rpm,
        quadcopter: Quadcopter,
    ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        forces = np.array(rpm**2) * quadcopter.quadcopter_specs.KF
        torques = np.array(rpm**2) * quadcopter.quadcopter_specs.KM

        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]
        for i in range(4):
            p.applyExternalForce(
                quadcopter.id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.environment_parameters.client_id,
            )
        p.applyExternalTorque(
            quadcopter.id,
            4,
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self.environment_parameters.client_id,
        )
