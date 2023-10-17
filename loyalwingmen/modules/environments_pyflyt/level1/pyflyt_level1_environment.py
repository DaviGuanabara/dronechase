import numpy as np

from pynput.keyboard import Key, KeyCode
from collections import defaultdict

from typing import Dict
from gymnasium import spaces, Env

from .components.pyflyt_level1_simulation import AviarySimulation
from ...quadcoters.components.base.quadcopter import Quadcopter
from .components.quadcopter_manager import QuadcopterManager
from .components.normalization import normalize_inertial_data


class PyflytL1Enviroment(Env):
    """
    This class aims to demonstrate a environment with one Loyal Wingmen and one Loitering Munition,
    in a simplest way possible.
    It was developed for Stable Baselines 3 2.0.0 or higher.
    Gym Pybullet Drones was an inspiration for this project. For more: https://github.com/utiasDSL/gym-pybullet-drones

    Finally, I tried to do my best inside of my time constraint. Then, sorry for messy code.
    """

    CATCH_DISTANCE = 0.2

    def __init__(
        self,
        dome_radius: float = 10,
        rl_frequency: int = 15,
        GUI: bool = False,
    ):
        """Initialize the environment."""

        self.CATCH_DISTANCE = 0.2
        self.MAX_REWARD = 1_000

        self.dome_radius = dome_radius
        self.simulation = AviarySimulation(world_scale=dome_radius, render=GUI)
        self.quadcopter_manager = QuadcopterManager(self.simulation)

        #### Frequency Adjustments ##################
        self.rl_frequency = rl_frequency
        ctrl_hz = self.simulation.ctrl_hz
        self.aggregate_sim_steps = int(ctrl_hz / rl_frequency)

        #### Create action and observation spaces ##################
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

        self.max_step_calls = 20 * rl_frequency
        self.step_calls = 0
        self.quadcopter_manager.spawn_invader(1, np.array([[0, 0, 0]]), "invader")
        self.quadcopter_manager.spawn_persuer(1, np.array([[1, 1, 1]]), "persuer")
        self.show_name_on = GUI

    def reset(self, seed=0):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """

        # I want to avoid the overhead of reseting the simulation.
        # self.simulation.reset()

        ##Reinit Variables
        self.step_calls = 0
        self.last_action = np.zeros(4)
        self.last_distance = 2 * self.dome_radius

        ##Replace drones to start position.
        self.quadcopter_manager.replace_all_to_starting_position()
        # self.quadcopter_manager.get_invaders()[0].drive(np.array([-1, 1, 1, 0.4]))
        # self.quadcopter_manager.update_control_physics_imu()
        self.update_last_distance()
        observation = self.compute_observation()
        info = self.compute_info()

        return observation, info

    ################################################################################

    def step(self, rl_action: np.ndarray):
        self.step_calls += 1
        self.last_action: np.ndarray = rl_action
        persuer = self.quadcopter_manager.get_persuers()[0]
        invader = self.quadcopter_manager.get_invaders()[0]
        persuer.drive(rl_action, self.show_name_on)

        for _ in range(self.aggregate_sim_steps):
            self.simulation.step()

        persuer.update_imu()
        invader.update_imu()
        observation = self.compute_observation()
        reward = self.compute_reward()
        terminated = self.compute_termination()
        truncated = False
        info = self.compute_info()

        self.replace_invader_if_close()
        self.update_last_distance()
        return observation, reward, terminated, truncated, info

    def replace_invader_if_close(self):
        invader = self.quadcopter_manager.get_invaders()[0]
        persuer = self.quadcopter_manager.get_persuers()[0]

        distance, direction = self.compute_distance(invader, persuer)
        # invader_position = invader.inertial_data["position"]
        # persuer_position = persuer.inertial_data["position"]
        # print("distance", distance)
        if distance < self.CATCH_DISTANCE:
            print("replace invader")
            position = np.random.uniform(-1, 1, 3)
            attitude = np.zeros(3)
            # motion_command = 3 * np.random.uniform([-1, -1, -1, 0], [1, 1, 1, 0.4], 4)

            # self.quadcopter_manager.replace_quadcopter(invader, position, attitude)
            invader.replace(position, attitude)
            motion_command = np.array([position[0], position[1], 0, position[2]])
            invader.set_setpoint(motion_command)
            invader.update_imu()
            invader.update_control()
            invader.update_physics()

    # ====================================================================================================
    # observation, reward, terminated, truncated, info
    # ====================================================================================================

    def compute_observation(self) -> np.ndarray:
        """Return the observation of the simulation."""

        invader: Quadcopter = self.quadcopter_manager.get_invaders()[0]
        persuer: Quadcopter = self.quadcopter_manager.get_persuers()[0]

        # print("invader position", invader.inertial_data["position"])
        distance, direction = self.compute_distance(invader, persuer)
        distance_normalized = distance / (2 * self.dome_radius)

        invader_data: np.ndarray = self.process_inertial_state(invader)
        persuer_data: np.ndarray = self.process_inertial_state(persuer)

        velocity = persuer.inertial_data["position"]

        velocity_factor = np.dot(velocity, direction) / np.linalg.norm(velocity)

        return np.array(
            [
                *persuer_data,
                *invader_data,
                *direction,
                distance_normalized,
                *self.last_action,
            ],
            dtype=np.float32,
        )

    def compute_reward(self):
        score, bonus, penalty = 0, 0, 0

        invader: Quadcopter = self.quadcopter_manager.get_invaders()[0]
        persuer: Quadcopter = self.quadcopter_manager.get_persuers()[0]

        distance, _ = self.compute_distance(invader, persuer)

        if distance < self.last_distance:
            velocity = persuer.inertial_data.get("velocity", np.zeros(3))
            bonus += 10 * np.linalg.norm(velocity)

        score = - distance

        if distance < self.CATCH_DISTANCE:
            bonus += self.MAX_REWARD

        if distance > self.dome_radius:
            penalty += self.MAX_REWARD

        return score + bonus - penalty

    def compute_termination(self) -> bool:
        """Calculate if the simulation is done."""

        if self.step_calls > self.max_step_calls:
            return True

        if self.is_outside_dome(self.simulation.drones[1]):
            return True

        return bool(self.is_outside_dome(self.simulation.drones[0]))

    ## Helpers #####################################################################

    def update_last_distance(self):
        persuer: Quadcopter = self.quadcopter_manager.get_invaders()[0]
        invader: Quadcopter = self.quadcopter_manager.get_persuers()[0]
        distance, _ = self.compute_distance(invader, persuer)
        self.last_distance = distance

    def compute_distance(self, invader: Quadcopter, persuer: Quadcopter):
        pi = persuer.inertial_data["position"]
        pf = invader.inertial_data["position"]
        distance_vector = pf - pi

        distance = np.linalg.norm(distance_vector)
        # self.last_distance = distance_to_target
        if distance == 0:
            direction = distance_vector
        else:
            direction = distance_vector / distance

        return distance, direction

    def process_inertial_state(self, quadcopter: Quadcopter) -> np.ndarray:
        inertial_data = quadcopter.inertial_data
        max_speed = quadcopter.max_speed
        dome_radius = self.dome_radius
        normalized_inertial_data = normalize_inertial_data(
            inertial_data, max_speed, dome_radius
        )

        return self.convert_dict_to_array(normalized_inertial_data)

    def convert_dict_to_array(self, dictionary: Dict) -> np.ndarray:
        array = np.array([])
        for key in dictionary:
            value: np.ndarray = (
                dictionary[key] if dictionary[key] is not None else np.array([])
            )
            array = np.concatenate((array, value))

        return array

    def is_outside_dome(self, entity: Quadcopter) -> bool:
        """Check if the entity is outside the dome radius."""
        inertial_data = entity.inertial_data

        position = inertial_data.get("position", np.zeros(3))

        return float(np.linalg.norm(position)) > self.dome_radius

    ################################################################################

    def close(self):
        """Terminates the environment."""

        self.simulation.close()

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
        """Returns the observation space of the environment.
        Returns
        -------
        ndarray
            A Box() shape composed by:

            loyal wingman inertial data: (3,)
            loitering munition inertial data: (3,)
            direction to the target: (3,)
            distance to the target: (1,)
            last action: (4,)

        Last Action is the last action applied to the loyal wingman.
        Its three first elements are the direction of the velocity vector and the last element is the intensity of the velocity vector.
        the direction varies from -1 to 1, and the intensity from 0 to 1.

        the other elements of the Box() shape varies from -1 to 1.

        """
        size = self.observation_size()
        low = -1 * np.ones(size)
        low[-1] = 0
        high = np.ones(size)

        return spaces.Box(
            low=low,
            high=high,
            shape=(size,),
            dtype=np.float32,
        )

    def observation_size(self):
        position = 3
        velocity = 3
        attitude = 3
        angular_rate = 3

        lw_inertial_data_shape = position + velocity + attitude + angular_rate
        lm_inertial_data_shape = position + velocity + attitude + angular_rate

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

    def manage_debug_text(self, text: str, debug_text_id=None):
        return self.simulation.addUserDebugText(
            text,
            position=np.zeros(3),
            eplaceItemUniqueId=debug_text_id,
            textColorRGB=[1, 1, 1],
            textSize=1,
            lifeTime=0,
        )

    def compute_info(self):
        
        return {}

    ################################################################################

    def get_keymap(self):
        keycode = KeyCode()
        default_action = [0, 0, 0, 0.001]  # Modify this to your actual default action

        key_map = defaultdict(lambda: default_action)
        key_map.update(
            {
                Key.up: [0, 1, 0, 0.001],
                Key.down: [0, -1, 0, 0.001],
                Key.left: [-1, 0, 0, 0.001],
                Key.right: [1, 0, 0, 0.001],
                keycode.from_char("w"): [0, 0, 1, 0.001],
                keycode.from_char("s"): [0, 0, -1, 0.001],
            }
        )

        return key_map