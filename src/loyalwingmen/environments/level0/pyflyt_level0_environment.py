import numpy as np
import pybullet as p

from pynput.keyboard import Key, KeyCode
from collections import defaultdict

import numpy as np
import pybullet as p

from typing import Dict, List, Tuple
from gymnasium import spaces, Env

from .pyflyt_level0_simulation_functional import AviarySimulation
from ...quadcoters.components.base.quadcopter import Quadcopter


class PyflytL0Enviroment(Env):
    """
    This class aims to demonstrate a environment with one Loyal Wingmen and one Loitering Munition,
    in a simplest way possible.
    It was developed for Stable Baselines 3 2.0.0 or higher.
    Gym Pybullet Drones was an inspiration for this project. For more: https://github.com/utiasDSL/gym-pybullet-drones

    Finally, I tried to do my best inside of my time constraint. Then, sorry for messy code.
    """

    def __init__(
        self,
        dome_radius: float = 20,
        rl_frequency: int = 30,
        GUI: bool = False,
        debug=False,
    ):
        physics_hz = 240
        self.simulation = AviarySimulation(
            world_scale=dome_radius, physics_hz=physics_hz, render=GUI
        )
        self.simulation.set_mode(6)
        self.rl_frequency = rl_frequency

        update_period = self.simulation.update_period
        update_hz = 1 / update_period
        self.aggregate_sim_steps = int(update_hz / rl_frequency)

        #### Create action and observation spaces ##################
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.dome_radius = dome_radius
        self.step_simulation_call = 0

    def manage_debug_text(self, text: str, debug_text_id=None):
        return p.addUserDebugText(
            text,
            position=np.zeros(3),
            eplaceItemUniqueId=debug_text_id,
            textColorRGB=[1, 1, 1],
            textSize=1,
            lifeTime=0,
        )

    def reset(self, seed=1):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """

        self.simulation.reset()
        self.step_simulation_call = 0
        self.last_action = np.zeros(4)
        self.last_distance = 2 * self.dome_radius

        observation = self.compute_observation()
        info = self.compute_info()

        return observation, info

    def compute_info(self):
        return {}

    ################################################################################

    def step(self, rl_action: np.ndarray):
        self.last_action: np.ndarray = rl_action
        command: np.ndarray = self.rl_action_to_mode_6_command(rl_action)

        for _ in range(self.aggregate_sim_steps):
            self.step_simulation_call += 1
            self.simulation.step(command)

        observation = self.compute_observation()
        reward = self.compute_reward()
        terminated = self.compute_termination()
        truncated = False
        info = {}

        lm = self.simulation.get_loiteringmunitions()[0]
        lw = self.simulation.get_loyalwingmen()[0]

        lw_state: dict = lw.flight_state
        lm_state: dict = lm.flight_state

        distance_vector = self._calculate_distance_vector(lw_state, lm_state)
        distance_to_target = np.linalg.norm(distance_vector)

        if distance_to_target < 0.2:
            self.simulation.replace_loiteringmunition()

        return observation, reward, terminated, truncated, info

    def compute_termination(self) -> bool:
        """Calculate if the simulation is done."""

        # end_time = time()
        # elapsed_time = end_time - self.start_time
        max_episode_time = 20
        # if max_episode_time > 0 and elapsed_time > max_episode_time:
        #    return True

        if self.step_simulation_call > max_episode_time * (
            1 / self.simulation.update_period
        ):
            return True

        if self.is_outside_dome(self.simulation.drones[1]):
            return True

        return bool(self.is_outside_dome(self.simulation.drones[0]))

    def is_outside_dome(self, entity: Quadcopter) -> bool:
        """Check if the entity is outside the dome radius."""
        inertial_data = entity.flight_state

        position = inertial_data.get("position", np.zeros(3))

        return float(np.linalg.norm(position)) > self.dome_radius

    def compute_observation(self) -> np.ndarray:
        """Return the observation of the simulation."""

        lm = self.simulation.get_loiteringmunitions()[0]
        lw = self.simulation.get_loyalwingmen()[0]

        # assert lw is not None, "Loyal wingman is not initialized"
        # assert lm is not None

        lw_state: dict = lw.flight_state
        lm_state: dict = lm.flight_state

        distance_vector = self._calculate_distance_vector(lw_state, lm_state)

        distance_to_target = np.linalg.norm(distance_vector)
        self.last_distance = distance_to_target
        direction_to_target = distance_vector / max(float(distance_to_target), 1)

        distance_to_target_normalized = distance_to_target / (2 * self.dome_radius)

        lw_inertial_data_normalized = normalize_inertial_data(
            lw_state, lw.max_speed, self.dome_radius
        )
        lm_inertial_data_normalized = normalize_inertial_data(
            lm_state, lm.max_speed, self.dome_radius
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

    def convert_dict_to_array(self, dictionary: Dict) -> np.ndarray:
        array = np.array([])
        for key in dictionary:
            value: np.ndarray = (
                dictionary[key] if dictionary[key] is not None else np.array([])
            )
            array = np.concatenate((array, value))

        return array

    def _calculate_distance_vector(self, lw_state: Dict, lm_state: Dict) -> np.ndarray:
        lw_pos = lw_state.get("position", np.zeros(3))
        lm_pos = lm_state.get("position", np.zeros(3))
        return lm_pos - lw_pos

    def compute_reward(self):
        # assert self.loyal_wingman is not None, "Loyal wingman is not initialized"
        # assert self.loitering_munition is not None

        bonus = 0
        penalty = 0
        score = 0

        lw_flight_state: dict = self.simulation.get_loyalwingmen()[0].flight_state
        lm_flight_state: dict = self.simulation.get_loiteringmunitions()[0].flight_state

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

        return score + bonus - penalty

    ################################################################################

    def close(self):
        """Terminates the environment."""

        self.simulation.close()

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
        # acceleration = 3
        # angular_acceleration = 3

        lw_inertial_data_shape = (
            position
            + velocity
            + attitude
            + angular_rate
            # + acceleration
            # + angular_acceleration
        )
        lm_inertial_data_shape = (
            position
            + velocity
            + attitude
            + angular_rate
            # + acceleration
            # + angular_acceleration
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

    def rl_action_to_mode_6_command(self, rl_action: np.array):
        "Considering mode 6"
        non_normalized_direction: np.ndarray = rl_action[:3]
        module = np.linalg.norm(non_normalized_direction)

        direction: np.ndarray = non_normalized_direction / (module if module > 0 else 1)
        intensity = rl_action[3]

        velocity: np.ndarray = intensity * direction
        return np.array([velocity[0], velocity[1], 0, velocity[2]])

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


# TODO: CHANGE THE NAME NORMALIZE_FLIGHT_STATE TO NORMALIZE_INERTIAL_DATA
def normalize_inertial_data(
    inertial_data: Dict,
    max_speed: float,
    dome_radius: float,
) -> Dict:
    # (2 * pi rad / rev)
    angular_max_speed = 2 * np.pi

    position = inertial_data.get("position", np.zeros(3))
    velocity = inertial_data.get("velocity", np.zeros(3))
    attitude = inertial_data.get("attitude", np.zeros(3))
    angular_rate = inertial_data.get("angular_rate", np.zeros(3))

    norm_position = normalize_position(position, dome_radius)
    norm_attitude = normalize_attitude(attitude)

    norm_velocity = normalize_velocity(velocity, max_speed)
    norm_angular_rate = normalize_angular_rate(angular_rate, angular_max_speed)

    return {
        "position": norm_position,
        "velocity": norm_velocity,
        "attitude": norm_attitude,
        "angular_rate": norm_angular_rate,
    }


def normalize_acceleration(acceleration: np.ndarray, max_acceleration: float):
    if max_acceleration == 0:
        return acceleration

    acceleration /= max_acceleration
    return np.clip(
        acceleration,
        -1,
        1,
        dtype=np.float32,
    )


def normalize_angular_acceleration(
    angular_acceleration: np.ndarray, max_angular_acceleration: float
):
    if max_angular_acceleration == 0:
        return angular_acceleration

    angular_acceleration /= max_angular_acceleration
    return np.clip(
        angular_acceleration,
        -1,
        1,
        dtype=np.float32,
    )


def normalize_position(position: np.ndarray, dome_radius: float):
    if dome_radius == 0:
        return position

    position /= dome_radius
    return np.clip(
        position,
        -1,
        1,
        dtype=np.float32,
    )


def normalize_velocity(velocity: np.ndarray, max_velocity: float):
    if max_velocity == 0:
        return velocity

    velocity /= max_velocity
    return np.clip(
        velocity,
        -1,
        1,
        dtype=np.float32,
    )


def normalize_attitude(attitude: np.ndarray, max_attitude: float = np.pi):
    if max_attitude == 0:
        return attitude

    attitude /= max_attitude
    return np.clip(
        attitude,
        -1,
        1,
        dtype=np.float32,
    )


def normalize_angular_rate(angular_rate: np.ndarray, max_angular_rate: float):
    if max_angular_rate == 0:
        return angular_rate

    angular_rate /= max_angular_rate
    return np.clip(
        angular_rate,
        -1,
        1,
        dtype=np.float32,
    )
