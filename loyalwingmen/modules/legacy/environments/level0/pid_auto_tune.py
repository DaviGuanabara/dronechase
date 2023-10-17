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


from ..helpers.environment_parameters import EnvironmentParameters
from time import time
from .nova_controladora import QuadcopterController
from gymnasium import spaces, Env
import time


class PIDAutoTuner(Env):
    def __init__(
        self,
        dome_radius: float,
        rl_frequency: float,
        simulation_frequency: float,
        GUI: bool = False,
    ):
        self.dome_radius = dome_radius
        self.rl_frequency = rl_frequency
        self.simulation_frequency = simulation_frequency

        self.setup_Parameteres(
            simulation_frequency,
            rl_frequency,
            GUI,
        )
        self.init_simulation()

        self.observation_space = self._observation_space()
        self.action_space = self._action_space()

    def setup_Parameteres(self, simulation_frequency, rl_frequency, GUI):
        self.environment_parameters = EnvironmentParameters(
            G=9.8,
            NEIGHBOURHOOD_RADIUS=np.inf,
            simulation_frequency=simulation_frequency,
            rl_frequency=rl_frequency,
            timestep=1 / simulation_frequency,
            aggregate_physics_steps=int(simulation_frequency / rl_frequency),
            max_distance=100,
            error=0.5,
            client_id=-1,
            debug=False,
            GUI=GUI,
        )

    def set_frequency(self, simulation_frequency, rl_frequency):
        self.environment_parameters.simulation_frequency = simulation_frequency
        self.environment_parameters.rl_frequency = rl_frequency

    def manage_debug_text(self, text: str, debug_text_id=None):
        return p.addUserDebugText(
            text,
            position=np.zeros(3),
            eplaceItemUniqueId=debug_text_id,
            textColorRGB=[1, 1, 1],
            textSize=1,
            lifeTime=0,
        )

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

    def _reset_simulation(self, gains: np.ndarray):
        """Housekeeping function.
        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.
        """

        #### Set PyBullet's parameters #############################
        if hasattr(self, "quadcopter"):
            self.quadcopter.detach_from_simulation()
        self.quadcopter: LoyalWingman = self.factory.create_loyalwingman(
            np.zeros(3), np.zeros(3), command_type=CommandType.RPM
        )

        self.controller = QuadcopterController(
            self.quadcopter.operational_constraints,
            self.quadcopter.quadcopter_specs,
            self.environment_parameters,
            control_frequency=self.rl_frequency,
            use_quadcopter_model=False,
        )

        self.controller.update_pid_gains(gains)

    """
    def convert_command_to_desired_velocity(self, command: np.ndarray):
        
        Converte o comando do RL em velocidade desejada.

        :param command: np.ndarray, Comando do RL
        :return: np.ndarray, Velocidade desejada [vx, vy, vz]

        speed_limit = self.quadcopter.operational_constraints.speed_limit
        intensity = np.linalg.norm(command[3])

        if np.linalg.norm(command[:3]) > 0:
            direction = command[:3] / np.linalg.norm(command[:3])
            return speed_limit * direction * intensity

        return speed_limit * command[:3] * intensity
    """

    def apply_step_input(self, desired_velocity: np.ndarray = np.ones(3)):
        """Execute a step in the simulation based on the RL action."""
        # self._reset_simulation()
        self.quadcopter.update_imu()
        speed_limit = self.quadcopter.operational_constraints.speed_limit

        inertial_data = self.quadcopter.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )
        rpm = self.controller.compute_rpm(
            desired_velocity * speed_limit, inertial_data, dt=1 / self.rl_frequency
        )

        for _ in range(self.environment_parameters.aggregate_physics_steps):
            self.quadcopter.drive(rpm)
            p.stepSimulation(physicsClientId=self.environment_parameters.client_id)

        self.quadcopter.update_imu()
        inertial_data = self.quadcopter.flight_state_by_type(
            FlightStateDataType.INERTIAL
        )

        return inertial_data

    def _preprocess_responses(self, responses):
        attitudes = np.array([response["attitude"] for response in responses])
        velocities = np.array([response["velocity"] for response in responses])
        return attitudes, velocities

    def tune(self):
        responses = [self.apply_step_input() for _ in range(10)]
        attitudes, velocites = self._preprocess_responses(responses)

        znfo_attitudes = []
        for attitude in attitudes:
            for i in range(3):
                K, T = self.analyze_response(attitude[i])
                znfo_attitudes.append(np.array(self.ziegler_nichols_first_order(K, T)))

        znfo_velocities = []
        for velocity in velocites:
            for i in range(3):
                K, T = self.analyze_response(velocity[i])
                znfo_velocities.append(np.array(self.ziegler_nichols_first_order(K, T)))

        # TODO: TUNE CONTROLLER
        self.tune_controller(znfo_attitudes, znfo_velocities)

    def tune_controller(self, znfo_attitudes, znfo_velocities):
        mean_kp_attitude = np.mean([params[0] for params in znfo_attitudes])
        mean_ki_attitude = np.mean([params[1] for params in znfo_attitudes])
        mean_kd_attitude = np.mean([params[2] for params in znfo_attitudes])

        mean_kp_velocity = np.mean([params[0] for params in znfo_velocities])
        mean_ki_velocity = np.mean([params[1] for params in znfo_velocities])
        mean_kd_velocity = np.mean([params[2] for params in znfo_velocities])
        # TODO: TUNE CONTROLLER

    def analyze_response(self, responses):
        # Escolha o método de cálculo do ganho estável (K)

        K = max(responses) - min(responses)

        # Calculando o tempo constante (T)
        target_value = responses[0] + 0.632 * K
        for i, response in enumerate(responses):
            if response >= target_value:
                T = i * self.environment_parameters.timestep
                break
        else:
            # Se a resposta nunca exceder 63,2% do valor final, então T é indefinido
            T = float("inf")

        return K, T

    def close(self):
        p.disconnect(physicsClientId=self.environment_parameters.client_id)

    def ziegler_nichols_first_order(self, K, T):
        kp = 0.6 * (T / K)
        ti = 2 * T
        td = 0.5 * T

        ki = kp / ti
        kd = kp * td

        return kp, ki, kd

    def cohen_coon(self, K, T):
        kp = (4.0 / 3.0) * (T / K)
        ti = T * ((32.0 + 6 * np.sqrt(6)) / (13.0 + 8 * np.sqrt(6)))
        td = T * (4.0 / (11.0 + 2 * np.sqrt(6)))

        ki = kp / ti
        kd = kp * td

        return kp, ki, kd

    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.
        Returns
        -------
        int:
            The PyBullet Client Id.
        """
        return self.environment_parameters.client_id

    ################################################################################

    def _action_space(self):
        # direction and intensity fo velocity
        return spaces.Box(
            low=np.array([-1, -1, -1, 0]),
            high=np.array([1, 1, 1, 1]),
            shape=(4,),
            dtype=np.float32,
        )

    def _observation_space(self):
        return spaces.Box(
            low=-1,
            high=1,
            shape=(5,),
            dtype=np.float32,
        )

    def reset(self, seed: int = 0):
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        return np.zeros(5, dtype=np.float32), 0, False, False, {}
