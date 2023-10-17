"""The Aviary class, the core of how PyFlyt handles UAVs in the PyBullet simulation environment."""
from __future__ import annotations

import time
from itertools import repeat
from typing import Any, Callable, Sequence
from warnings import warn

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client


from PyFlyt.core.abstractions import DroneClass, WindFieldClass
from PyFlyt.core.drones import Fixedwing, QuadX, Rocket

from ...quadcoters.components.base.quadcopter import (
    Quadcopter,
    QuadcopterType,
)

DroneIndex = int


class AviarySimulation(bullet_client.BulletClient):
    """Aviary class, the core of how PyFlyt handles UAVs in the PyBullet simulation environment.

    The `aviary` is a handler for physics stepping, setpoint handling, collisions tracking, and much more.
    It provides a common endpoint from where users may control drones or define tasks.

    Args:
        start_pos (np.ndarray): an `(n, 3)` array for the starting X, Y, Z positions for each drone.
        start_orn (np.ndarray): an `(n, 3)` array for the starting orientations for each drone, in terms of Euler angles.
        drone_type (str | Sequence[str]): a _lowercase_ string representing what type of drone to spawn.
        drone_type_mappings (None | dict(str, Type[DroneClass])): a dictionary mapping of `{str: DroneClass}` for spawning custom drones.
        drone_options (dict[str, Any] | Sequence[dict[str, Any]]): dictionary mapping of custom parameters for each drone.
        wind_type (None | str | Type[WindField]): a wind field model that will be used throughout the simulation.
        wind_options (dict[str, Any] | Sequence[dict[str, Any]]): dictionary mapping of custom parameters for the wind field.
        render (bool): a boolean whether to render the simulation.
        physics_hz (int): physics looprate (not recommended to be changed).
        world_scale (float): how big to spawn the floor.
        seed (None | int): optional int for seeding the simulation RNG.
    """

    def __init__(
        self,
        render: bool = False,
        physics_hz: int = 240,
        world_scale: float = 1.0,
        seed: None | int = None,
    ):
        """Initializes a PyBullet environment that hosts UAVs and other entities.

        The Aviary class itself inherits from a BulletClient, so any function that a PyBullet client has, this class will have.
        The Aviary also handles dealing with physics and control looprates, as well as automatic construction of several default UAVs and their corresponding cameras.

        Args:
            start_pos (np.ndarray): an `(n, 3)` array for the starting X, Y, Z positions for each drone.
            start_orn (np.ndarray): an `(n, 3)` array for the starting orientations for each drone, in terms of Euler angles.
            drone_type (str | Sequence[str]): a _lowercase_ string representing what type of drone to spawn.
            drone_type_mappings (None | dict(str, Type[DroneClass])): a dictionary mapping of `{str: DroneClass}` for spawning custom drones.
            drone_options (dict[str, Any] | Sequence[dict[str, Any]]): dictionary mapping of custom parameters for each drone.
            wind_type (None | str | Type[WindField]): a wind field model that will be used throughout the simulation.
            wind_options (dict[str, Any] | Sequence[dict[str, Any]]): dictionary mapping of custom parameters for the wind field.
            render (bool): a boolean whether to render the simulation.
            physics_hz (int): physics looprate (not recommended to be changed).
            world_scale (float): how big to spawn the floor.
            seed (None | int): optional int for seeding the simulation RNG.
        """
        super().__init__(p.GUI if render else p.DIRECT)
        print("\033[A                             \033[A")

        # do not change because pybullet doesn't like it
        # default physics looprate is 240 Hz
        self.physics_hz = physics_hz
        self.physics_period = 1.0 / physics_hz

        # constants
        start_pos = np.array([0.0, 0.0, 1.0])
        start_orn = np.array([0.0, 0.0, 0.0])

        self.start_pos = start_pos
        self.start_orn = start_orn

        # set the world scale and directories
        self.world_scale = world_scale
        self.setAdditionalSearchPath(pybullet_data.getDataPath())

        # render
        self.render = render
        self.rtf_debug_line = self.addUserDebugText(
            text="RTF here", textPosition=[0, 0, 0], textColorRGB=[1, 0, 0]
        )

        self.reset(seed)

    def reset(self, seed: None | int = None):
        """Resets the simulation.

        Args:
            seed (None | int): seed
        """
        self.resetSimulation()
        self.setGravity(0, 0, -9.81)
        self.physics_steps: int = 0
        self.aviary_steps: int = 0
        self.elapsed_time: float = 0

        # reset the camera position to a sane place
        self.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 1],
        )

        # define new RNG
        self.np_random = np.random.RandomState(seed=seed)

        # construct the world
        self.planeId = self.loadURDF(
            "plane.urdf", useFixedBase=True, globalScaling=self.world_scale
        )

        self.loyalwingman: Quadcopter = Quadcopter.from_quadx(
            p=self,
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            physics_hz=self.physics_hz,
            np_random=self.np_random,
            quadcopter_type=QuadcopterType.LOYALWINGMAN,
        )

        control_hz = self.loyalwingman.control_period
        self.updates_per_step = int(self.physics_hz / control_hz)
        self.update_period = 1.0 / control_hz

        # sanity check the control looprates

        # rtf tracking parameters
        self.now = time.time()
        self._frame_elapsed = 0.0
        self._sim_elapsed = 0.0

        # reset all drones and initialize required states
        self.loyalwingman.reset()
        self.loyalwingman.update_state()
        self.loyalwingman.update_last()

    def state(self, index: DroneIndex) -> np.ndarray:
        """Returns the state for the indexed drone.

        This is a (4, 3) array, where:
            - `state[0, :]` represents body frame angular velocity
            - `state[1, :]` represents ground frame angular position
            - `state[2, :]` represents body frame linear velocity
            - `state[3, :]` represents ground frame linear position

        Args:
            index (DRONE_INDEX): index

        Returns:
            np.ndarray: state
        """
        return self.loyalwingman.state

    def set_mode(self, flight_modes: int | list[int]):
        """Sets the flight control mode of each drone in the environment.

        Args:
            flight_modes (int | list[int]): flight mode
        """

        self.loyalwingman.set_mode(flight_modes)

    def set_setpoint(self, setpoint: np.ndarray):
        """Sets the setpoint of one drone in the environment.

        Args:
            index (DRONE_INDEX): index
            setpoint (np.ndarray): setpoint
        """
        # talvexz mudar para set_setpoint
        self.loyalwingman.setpoint = setpoint

    def step(self):
        """Steps the environment, this automatically handles physics and control looprates, one step is equivalent to one control loop step."""
        # compute rtf if we're rendering
        if self.render:
            elapsed = time.time() - self.now
            self.now = time.time()

            self._sim_elapsed += self.update_period * self.updates_per_step
            self._frame_elapsed += elapsed

            time.sleep(max(self._sim_elapsed - self._frame_elapsed, 0.0))

            # print RTF every 0.5 seconds, this actually adds considerable overhead
            if self._frame_elapsed >= 0.5:
                # calculate real time factor based on realtime/simtime
                RTF = self._sim_elapsed / (self._frame_elapsed + 1e-6)
                self._sim_elapsed = 0.0
                self._frame_elapsed = 0.0

                self.rtf_debug_line = self.addUserDebugText(
                    text=f"RTF: {RTF:.3f}",
                    textPosition=[0, 0, 0],
                    textColorRGB=[1, 0, 0],
                    replaceItemUniqueId=self.rtf_debug_line,
                )

        # step the environment enough times for one control loop of the slowest controller
        for step in range(self.updates_per_step):
            # update onboard avionics conditionally

            if step % self.loyalwingman.physics_control_ratio == 0:
                self.loyalwingman.update_control()

            # update physics and state
            self.loyalwingman.update_physics()
            self.loyalwingman.update_state()

            # advance pybullet
            self.stepSimulation()

            self.physics_steps += 1
            self.elapsed_time = self.physics_steps / self.physics_hz

        # update the last components of the drones, this is usually limited to cameras only
        self.loyalwingman.update_last()

        self.aviary_steps += 1
