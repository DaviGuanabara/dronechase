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


class DroneChaseStaticTargetSimulation:
    def __init__(
        self,
        dome_radius: float,
        environment_parameters: EnvironmentParameters,
    ):
        self.dome_radius = dome_radius

        self.environment_parameters = environment_parameters
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
            0,  # -self.environment_parameters.G,
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
            position, np.zeros(3), quadcopter_name="agent"
        )

        self.loitering_munition: LoiteringMunition = (
            self.factory.create_loiteringmunition(
                np.zeros(3), np.zeros(3), quadcopter_name="target"
            )
        )

        self.last_action: np.ndarray = np.zeros(4)

        self.reset()

    def update_time_text(self):
        simulation_calls = self.step_simulation_call
        time_in_seconds = simulation_calls * self.environment_parameters.timestep

        # _, _, _, _, _, _, camera_position, _, _, _, _, _ = p.getDebugVisualizerCamera(
        #    physicsClientId=self.environment_parameters.client_id
        # )

        text = "elapsed time:{:.2f}s".format(time_in_seconds)
        if hasattr(self, "time_text"):
            self.time_text = p.addUserDebugText(
                text,
                [-0.5, 0, -0.5],
                [1, 0, 1],
                textSize=1,
                lifeTime=0,
                replaceItemUniqueId=self.time_text,
                physicsClientId=self.environment_parameters.client_id,
            )

        else:
            self.time_text = p.addUserDebugText(
                text,
                [-0.5, 0, -0.5],
                [1, 0, 1],
                textSize=1,
                lifeTime=0,
                physicsClientId=self.environment_parameters.client_id,
            )

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

    def gen_initial_position(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random position within the dome."""

        position = np.random.rand(3)  # * (self.dome_radius / 2)
        ang_position = np.random.uniform(0, 2 * np.pi, 3)

        return position, ang_position

    def _housekeeping(self):
        pos, ang_pos = self.gen_initial_position()
        self.step_simulation_call = 0

        target_pos, target_ang_pos = np.array([0, 0, 0]), np.array([0, 0, 0])

        if self.loyal_wingman is not None:
            self.loyal_wingman.detach_from_simulation()

        if self.loitering_munition is not None:
            self.loitering_munition.detach_from_simulation()

        self.loyal_wingman = self.factory.create_loyalwingman(
            position=pos,
            ang_position=np.zeros(3),
            command_type=CommandType.VELOCITY_DIRECT,
            quadcopter_name="The RL Agent",
            quadcopter_role="interceptor",
        )
        self.loitering_munition = self.factory.create_loiteringmunition(
            position=target_pos,
            ang_position=target_ang_pos,
            quadcopter_name="The Target",
            quadcopter_role="intruder",
        )

        self.loyal_wingman.update_imu()
        self.loitering_munition.update_imu()

        self.loitering_munition.set_behavior(LoiteringMunitionBehavior.FROZEN)
        self.start_time = time()

    def reset(self):
        """Reset the simulation to its initial state."""

        self._housekeeping()

        observation = self.compute_observation()
        info = self.compute_info()

        return observation, info

    def compute_observation(self) -> dict:
        """Return the observation of the simulation."""

        lw = self.loyal_wingman
        lw_state = lw.flight_state

        lidar: np.ndarray = lw_state.get(
            "lidar",
            np.zeros(
                lw.get_lidar_shape(),
                dtype=np.float32,
            ),
        )
        lidar = np.array(
            lidar,
            dtype=np.float32,
        )

        inertial_data_raw = normalize_inertial_data(
            lw_state, self.loyal_wingman.operational_constraints, self.dome_radius
        )
        inertial_data = self.convert_dict_to_array(inertial_data_raw)
        inertial_data = np.array(inertial_data, dtype=np.float32)

        return {
            "lidar": lidar.astype(np.float32),
            "inertial_data": inertial_data.astype(np.float32),
            "last_action": self.last_action.astype(np.float32),
        }

    def distance(self, lw_state: Dict, lm_state: Dict) -> float:
        distance_vector = self._calculate_distance_vector(lw_state, lm_state)
        return float(np.linalg.norm(distance_vector))

    def observation_shape(self) -> dict:
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

        lw = self.loyal_wingman

        lidar_shape = lw.get_lidar_shape()

        last_action_shape = 4

        return {
            "lidar": lidar_shape,
            "inertial_data": lw_inertial_data_shape,
            "last_action": last_action_shape,
        }

    def compute_termination(self) -> bool:
        """Calculate if the simulation is done."""

        max_episode_time = self.environment_parameters.max_episode_time

        if (
            self.step_simulation_call
            > max_episode_time * self.environment_parameters.simulation_frequency
        ):
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
            self.loyal_wingman.drive(rl_action)
            self.loitering_munition.drive_via_behavior()
            p.stepSimulation()
            self.step_simulation_call += 1
            self.update_time_text()

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

        lw_inertial_data = self.loyal_wingman.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )
        lm_inertial_data = self.loitering_munition.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )

        distance_vector = self._calculate_distance_vector(
            lw_inertial_data, lm_inertial_data
        )
        distance = np.linalg.norm(distance_vector)

        score = self.dome_radius - distance

        if distance < 0.2:
            bonus += 1_000

        if distance > self.dome_radius:
            penalty += 1_000

        # ====================================================
        # Bonus for how fast it gets close the target
        # - Velocity factor that lies in
        #   the direction of the target
        # ====================================================

        direction = distance_vector / distance if distance > 0 else np.zeros(3)

        velocity = lw_inertial_data.get("velocity", np.zeros(3))

        velocity_component = np.dot(velocity, direction)
        velocity_towards_target = max(velocity_component, 0)

        bonus += 2 * float(np.linalg.norm(x=velocity_towards_target))

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
        self.loitering_munition.set_behavior(LoiteringMunitionBehavior.CIRCLE)

    def convert_dict_to_array(self, dictionary: Dict) -> np.ndarray:
        array = np.array([])
        for key in dictionary:
            value: np.ndarray = (
                dictionary[key] if dictionary[key] is not None else np.array([])
            )
            array = np.concatenate(
                (array, value),
                dtype=np.float32,
            )

        return array

    def close(self):
        p.disconnect(physicsClientId=self.environment_parameters.client_id)
