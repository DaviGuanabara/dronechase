import pybullet as p
import numpy as np
import pybullet_data
import random
from typing import Tuple, Optional, Union, Dict, List
from time import time

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
from ...quadcoters.components.controllers.DSLPIDControl import DSLPIDControl

from ..helpers.environment_parameters import EnvironmentParameters


class PIDTuningSimulation:
    def __init__(
        self,
        dome_radius: float,
        environment_parameters: EnvironmentParameters,
        seed: int = 0,
    ):
        """
        Initializes the PID tuning simulation.

        Args:
        - environment_parameters: Configuration for the environment.
        - dome_radius: The maximum allowed radius for the drone to travel from its origin.
        - seed: Random seed for reproducibility.
        """

        np.random.seed(seed)
        self.target_velocity = np.ones(3)
        self.drone_initial_state = np.zeros(3)
        self.command_type = CommandType.VELOCITY_TO_CONTROLLER
        self.environment_parameters = environment_parameters

        self.factory = QuadcopterFactory(environment_parameters)
        self.start_time = time()
        self.dome_radius = dome_radius
        self.init_simulation()
        self.reset()

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

    def _init_loyalwingman(
        self, initial_position: np.ndarray, initial_angular: np.ndarray
    ):
        """
        Initializes the loyal wingman drone with given position and angular position.
        """
        self.loyalwingman = self.factory.create_loyalwingman(
            initial_position, initial_angular, self.command_type
        )
        self.loyalwingman.update_imu()

    def _compute_control(
        self, controller: DSLPIDControl, target_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Computes the control RPMs using the given controller and target velocity.
        """
        inertial_data = self.loyalwingman.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )

        yaw = (
            inertial_data["attitude"][2]
            if inertial_data["attitude"] is not None
            else 0.0
        )
        target_rpy = np.array([0, 0, yaw])

        aggregate_physics_steps = self.environment_parameters.aggregate_physics_steps
        timestep = self.environment_parameters.timestep
        control_timestep = aggregate_physics_steps * timestep

        rpm, _, _ = controller.computeControl(
            control_timestep,
            inertial_data["position"],
            inertial_data["quaternions"],
            inertial_data["velocity"],
            inertial_data["attitude"],
            target_pos=inertial_data["position"],
            target_rpy=target_rpy,
            target_vel=target_velocity,
        )

        return rpm

    def step(self, action: List[float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Performs one step in the simulation given the action.
        """
        pid_coefficientes = self._preprocesss_action(action)
        controller = DSLPIDControl(
            self.loyalwingman.model,
            self.loyalwingman.quadcopter_specs,
            self.environment_parameters,
            pid_coefficientes,
        )
        rpm = self._compute_control(controller, self.target_velocity)

        self.loyalwingman.drive(rpm)

        for _ in range(self.environment_parameters.aggregate_physics_steps):
            p.stepSimulation()

        # Compute reward and check termination criteria
        self.loyalwingman.update_imu()
        terminated = self.is_done()
        reward = self.compute_reward()
        observation = self.compute_observation()

        return observation, reward, terminated, {}

    def compute_observation(self) -> np.ndarray:
        lw = self.loyalwingman
        lw_state = lw.flight_state_by_type(FlightStateDataType.INERTIAL)
        norm_lw_state = normalize_inertial_data(
            lw_state, lw.operational_constraints, self.dome_radius
        )

        target_velocity = self.target_velocity
        last_action = self.last_action
        converted_norm_lw_state = self.convert_dict_to_array(norm_lw_state)
        print("converted_norm_lw_state", converted_norm_lw_state)
        print(target_velocity)
        print(last_action)
        return np.array(
            [
                *converted_norm_lw_state,
                *target_velocity,
                *last_action,
            ],
            dtype=np.float32,
        )

    def observation_size(self):
        lw_state = 18
        target_velocity = 3
        last_action = 18
        return lw_state + target_velocity + last_action

    def _preprocesss_action(self, action: List[float]) -> Dict[str, np.ndarray]:
        """
        Processes the raw action into PID coefficients.
        """

        MAX_PID_COEFF = 100_000
        keys = [
            "P_COEFF_FOR",
            "I_COEFF_FOR",
            "D_COEFF_FOR",
            "P_COEFF_TOR",
            "I_COEFF_TOR",
            "D_COEFF_TOR",
        ]
        assert len(action) == 3 * len(keys), "Invalid action length"
        return {
            keys[i]: MAX_PID_COEFF * np.array(action[3 * i : 3 * (i + 1)])
            for i in range(len(keys))
        }

    def generate_target_velocity(self) -> np.ndarray:
        """
        Generates a random target velocity.
        """
        # Generating a 3D velocity vector
        return np.random.uniform(-1, 1, 3)

    def reset(self):
        """
        Resets the simulation environment.
        """
        initial_position = np.zeros(3)
        initial_angular_position = np.zeros(3)
        self.target_velocity = np.ones(3)
        self.last_action = np.zeros(18)

        self._init_loyalwingman(initial_position, initial_angular_position)
        self.start_time = time()

        return self.compute_observation(), {}

    def compute_reward(self):
        """
        Computes the reward for the current state.
        """
        inertial_data = self.loyalwingman.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )

        actual_velocity = inertial_data["velocity"]
        velocity_difference = float(
            np.linalg.norm(actual_velocity - self.target_velocity)
        )

        return -velocity_difference

    def is_done(self):  # sourcery skip: remove-unnecessary-cast
        """
        Determines if the episode should terminate.
        """
        elapsed_time = time() - self.start_time
        if elapsed_time > 20:
            return True

        inertial_data = self.loyalwingman.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )

        if np.linalg.norm(inertial_data["position"]) > self.dome_radius:
            return True

        actual_velocity = inertial_data["velocity"]
        velocity_difference = np.linalg.norm(actual_velocity - self.target_velocity)

        return bool(velocity_difference > 0.5)

    def close(self):
        p.disconnect(physicsClientId=self.environment_parameters.client_id)

    def convert_dict_to_array(self, dictionary: Dict) -> np.ndarray:
        array = np.array([])
        for key in dictionary:
            value: np.ndarray = (
                dictionary[key] if dictionary[key] is not None else np.array([])
            )
            array = np.concatenate((array, value))

        return array
