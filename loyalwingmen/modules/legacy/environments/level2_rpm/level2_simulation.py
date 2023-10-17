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


class L2RPMDroneChaseSimulation:
    def __init__(
        self,
        dome_radius: float,
        environment_parameters: EnvironmentParameters,
        interactive_mode: bool = False,
    ):
        self.dome_radius = dome_radius

        self.environment_parameters = environment_parameters
        self.init_simulation(interactive_mode)

    def init_simulation(self, interactive_mode: bool = False):
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

        """
        position, ang_position = self.gen_initial_position()

        if interactive_mode:
            print("INTERACTIVE MODE IS ON. MAKE SURE YOU ARE NOT TRAINING IT")
            command_type = CommandType.VELOCITY_DIRECT
        else:
            print("RPM IS ON. IT SUPPOSED BEING IN TRAINING")
            command_type = CommandType.RPM

        self.loyal_wingman: LoyalWingman = self.factory.create_loyalwingman(
            position, np.zeros(3), quadcopter_name="agent", command_type=command_type
        )

        self.loitering_munition: LoiteringMunition = (
            self.factory.create_loiteringmunition(
                np.zeros(3), np.zeros(3), quadcopter_name="target"
            )
        )

        self.last_action: np.ndarray = np.zeros(4)
        """

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

        if hasattr(self, "loyal_wingman") and self.loyal_wingman is not None:
            self.loyal_wingman.detach_from_simulation()

        if hasattr(self, "loitering_munition") and self.loitering_munition is not None:
            self.loitering_munition.detach_from_simulation()

        self.loyal_wingman: LoyalWingman = self.factory.create_loyalwingman(
            position=pos,
            ang_position=np.zeros(3),
            command_type=CommandType.RPM,
            quadcopter_name="agent",
        )
        self.loitering_munition: LoiteringMunition = (
            self.factory.create_loiteringmunition(
                position=target_pos,
                ang_position=target_ang_pos,
                quadcopter_name="target",
            )
        )

        self.loyal_wingman.update_imu()
        self.loitering_munition.update_imu()

        self.loitering_munition.set_behavior(LoiteringMunitionBehavior.FROZEN)
        self.last_action = np.zeros(4)
        self.current_action = np.zeros(4)

        self.start_time = time()

        lw_state = self.loyal_wingman.flight_state_by_type(FlightStateDataType.INERTIAL)
        lm_state = self.loitering_munition.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )
        distance_vector = self._calculate_distance_vector(lw_state, lm_state)
        self.last_distance_to_target = np.linalg.norm(distance_vector)

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
                *self.current_action,
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

        self.last_action = self.current_action
        self.current_action = rl_action

        for _ in range(self.environment_parameters.aggregate_physics_steps):
            self.loyal_wingman.drive(rl_action)
            self.loitering_munition.drive_via_behavior()
            p.stepSimulation()

        self.loyal_wingman.update_imu()
        self.loitering_munition.update_imu()

        # self.loyal_wingman.update_lidar()
        # self.loitering_munition.update_lidar()

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

    def get_angular_data(
        self, inertial_data: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        attitude = inertial_data.get("attitude", np.zeros(3))
        angular_velocity = inertial_data.get("angular_velocity", np.zeros(3))
        angular_acceleration = inertial_data.get("angular_acceleration", np.zeros(3))

        return attitude, angular_velocity, angular_acceleration

    def angular_difference(self, attitude1, attitude2):
        """
        Compute the element-wise difference between two attitude arrays in radians.
        Each result will be in [-pi, pi].
        """
        diff = attitude1 - attitude2

        diff[diff > np.pi] -= 2 * np.pi
        diff[diff < -np.pi] += 2 * np.pi

        return diff

    def compute_reward(self):
        max_bonus = 10_000
        max_penalty = 10_000

        penalty: float = 0
        bonus: float = 0
        score: float = 0

        lw_inertial_data = self.loyal_wingman.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )

        lm_inertial_data = self.loitering_munition.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )

        lw_position = lw_inertial_data.get("position", np.zeros(3))
        lm_position = lm_inertial_data.get("position", np.zeros(3))
        distance_vetor = lm_position - lw_position
        distance_intensity = np.linalg.norm(distance_vetor)

        # ====================================================
        # Basic Pontuation
        #   - Pontuation for proximity to the target,
        #     a reward for being alive and inside the dome
        #
        # ====================================================

        score = float(self.dome_radius - distance_intensity)

        # ====================================================
        # Bonus for being closer to the target
        #   - Huge bonus for < 0.2
        #
        # Penalty for being outside the dome
        #  - Huge penalty for > dome_radius
        #
        # ====================================================

        if distance_intensity < self.last_distance_to_target:
            bonus += float(self.last_distance_to_target - distance_intensity)

        self.last_distance_to_target = distance_intensity

        if distance_intensity < 0.2:
            bonus += max_bonus

        if distance_intensity > self.dome_radius:
            penalty += max_penalty

        # ====================================================
        # Bonus for how fast it gets close the target
        # - Velocity factor that lies in
        #   the direction of the target
        # ====================================================

        direction = (
            distance_vetor / distance_intensity
            if distance_intensity > 0
            else np.zeros(3)
        )

        velocity = lw_inertial_data.get("velocity", np.zeros(3))

        velocity_component = np.dot(velocity, direction)
        velocity_towards_target = max(velocity_component, 0)

        bonus += float(np.linalg.norm(x=velocity_towards_target))

        # ====================================================
        # Bonus for stability
        # - Attitude <= 0.1
        #
        # Penalty for Unstability
        #  - Attitude > 0.1
        #  - Angular Velocity that amplifies attitude
        #  - Angular Acceleration that amplifies attitude
        #
        # ====================================================
        t = (
            self.environment_parameters.timestep
            * self.environment_parameters.aggregate_physics_steps
        )
        attitude, ang_vel, ang_acc = self.get_angular_data(lw_inertial_data)
        next_attitude = attitude + ang_vel * t + ang_acc * (t**2) / 2

        attitude_intensity = np.linalg.norm(attitude)

        # penalize for more than 5 degrees
        if attitude_intensity > 5 * 2 * np.pi / 360:
            penalty += float(attitude_intensity)
        else:
            bonus += 1

        direction_to_stable = np.zeros(3) - attitude
        directional_change = next_attitude - attitude
        is_becoming_stable = np.dot(direction_to_stable, directional_change) > 0

        if not is_becoming_stable:
            diff_to_next = np.abs(self.angular_difference(attitude, next_attitude))
            diff_to_stable = np.abs(self.angular_difference(np.zeros(3), next_attitude))
            penalty += float(np.linalg.norm(diff_to_next - diff_to_stable))

        # ====================================================
        # Penalty for expending energy
        # - i am looking for smoothier moviments
        #   so, i think that reducing the intensity
        #   of the actions is a good idea
        #
        #
        # ====================================================

        penalty += float(0.1 * np.linalg.norm(self.current_action))

        # ====================================================
        # Penalty for jerkiness
        # - i am looking for smoothier moviments
        #   so, i think that reducing the intensity
        #   of the actions is a good idea
        #
        #
        # ====================================================

        action_change = self.last_action - self.current_action
        penalty += float(np.linalg.norm(action_change))

        bonus = min(bonus, max_bonus)
        penalty = min(penalty, max_penalty)
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
        self.loitering_munition.detach_from_simulation()
        self.loitering_munition = self.factory.create_loiteringmunition(
            position=position, ang_position=ang_position
        )

        self.loitering_munition.update_imu()

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
