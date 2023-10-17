import gymnasium as gym
import numpy as np
import pybullet as p
from typing import Dict
from enum import Enum, auto


import time

# import curses
import random
import math

from pynput.keyboard import Key, KeyCode
from collections import defaultdict

import numpy as np
import pybullet as p
import pybullet_data


import gymnasium as gym
from gymnasium import spaces, Env

from typing import Dict, List, Tuple, Union, Optional
from ..helpers.environment_parameters import EnvironmentParameters
from .level2_simulation import (
    DroneChaseStaticTargetSimulation as Level1Simulation,
)


class Level2_lidar(Env):
    """
    This class aims to demonstrate a environment with one Loyal Wingmen and one Loitering Munition,
    in a simplest way possible.
    It was developed for Stable Baselines 3 2.0.0 or higher.
    Gym Pybullet Drones was an inspiration for this project. For more: https://github.com/utiasDSL/gym-pybullet-drones

    Finally, I tried to do my best inside of my time constraint. Then, sorry for messy code.
    """

    def __init__(
        self,
        simulation_frequency: int = 240,
        rl_frequency: int = 30,
        dome_radius: float = 20,
        GUI: bool = False,
        debug: bool = False,
    ):
        self.setup_Parameteres(simulation_frequency, rl_frequency, GUI, debug)
        self.simulation = Level1Simulation(dome_radius, self.environment_parameters)

        #### Create action and observation spaces ##################
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

    def setup_Parameteres(self, simulation_frequency, rl_frequency, GUI, debug):
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
            debug=debug,
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

    def get_parameteres(self):
        return self.environment_parameters

    def reset(self, seed=1):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """

        observation, info = self.simulation.reset()

        return observation, info

    ################################################################################

    def step(self, rl_action: np.ndarray):
        observation, reward, terminated, Truncated, info = self.simulation.step(
            rl_action
        )
        return observation, reward, terminated, Truncated, info

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
        observation_shape = self.simulation.observation_shape()
        # print("inertial_data", observation_shape["inertial_data"])
        return spaces.Dict(
            {
                "lidar": spaces.Box(
                    0,
                    1,
                    shape=observation_shape["lidar"],
                    dtype=np.float32,
                ),
                "inertial_data": spaces.Box(
                    -np.ones((observation_shape["inertial_data"],)),
                    np.ones((observation_shape["inertial_data"],)),
                    shape=(observation_shape["inertial_data"],),
                    dtype=np.float32,
                ),
                "last_action": spaces.Box(
                    np.array([-1, -1, -1, 0]),
                    np.array([1, 1, 1, 1]),
                    shape=(observation_shape["last_action"],),
                    dtype=np.float32,
                ),
            }
        )

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
