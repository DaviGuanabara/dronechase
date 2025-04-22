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
from threatengage.modules.quadcoters.components.base.quadcopter import (
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
        wind_type: None | str | type[WindFieldClass] = None,
        wind_options: dict[str, Any] = {},
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

        # check the physics hz
        if physics_hz != 240.0:
            warn(
                f"Physics_hz is currently {physics_hz}, not the 240.0 that is recommended by pybullet. There may be physics errors."
            )

        start_pos = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        start_orn = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        self.num_drones = start_pos.shape[0]
        self.start_pos = start_pos
        self.start_orn = start_orn

        self.physics_hz = physics_hz
        self.physics_period = 1.0 / physics_hz

        self.wind_type = wind_type
        self.wind_options = wind_options

        # set the world scale and directories
        self.world_scale = world_scale
        self.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.loiteringmunition_position = np.random.rand(4)

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

        self.drones = [
            Quadcopter.from_quadx(
                p=self,
                start_pos=self.start_pos[0],
                start_orn=self.start_orn[0],
                physics_hz=self.physics_hz,
                np_random=self.np_random,
                quadcopter_type=QuadcopterType.LOYALWINGMAN,
            ),
            Quadcopter.from_quadx(
                p=self,
                start_pos=self.start_pos[1],
                start_orn=self.start_orn[0],
                physics_hz=self.physics_hz,
                np_random=self.np_random,
                quadcopter_type=QuadcopterType.LOITERINGMUNITION,
            ),
        ]

        # initialize the wind field
        self.wind_field: None | WindFieldClass | Callable
        if self.wind_type is None:
            # no wind field
            self.wind_field = None

        # constants for tracking how many times to step depending on control hz
        all_control_hz = [int(1.0 / drone.control_period) for drone in self.drones]
        self.updates_per_step = int(self.physics_hz / np.min(all_control_hz))
        self.update_period = 1.0 / np.min(all_control_hz)

        # sanity check the control looprates
        if len(all_control_hz) > 0:
            all_control_hz.sort()
            all_ratios = np.array(all_control_hz)[1:] / np.array(all_control_hz)[:-1]
            assert all(
                r % 1.0 == 0.0 for r in all_ratios
            ), "Looprates must form common multiples of each other."

        # rtf tracking parameters
        self.now = time.time()
        self._frame_elapsed = 0.0
        self._sim_elapsed = 0.0

        # reset all drones and initialize required states
        self.setup_drones()

    def get_loyalwingmen(self):
        return [
            drone
            for drone in self.drones
            if drone.quadcopter_type == QuadcopterType.LOYALWINGMAN
        ]

    def get_loiteringmunitions(self):
        return [
            drone
            for drone in self.drones
            if drone.quadcopter_type == QuadcopterType.LOITERINGMUNITION
        ]

    def setup_drones(self):
        self.set_loiteringmunition_setpoint(self.loiteringmunition_position)
        for drone in self.drones:
            drone.reset()
            drone.update_state()
            drone.update_last()

    def register_wind_field_function(self, wind_field: Callable):
        """For less complicated wind field models (time invariant models), this allows the registration of a normal function as a wind field model.

        Args:
            wind_field (Callable): given the time float and a position as an (n, 3) array, must return a (n, 3) array representing the local wind velocity.
        """
        assert callable(wind_field), "`wind_field` function must be callable."
        WindFieldClass._check_wind_field_validity(wind_field)
        self.wind_field = wind_field

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
        return self.drones[index].flight_state

    @property
    def all_states(self) -> list[np.ndarray]:
        """Returns a list of states for all drones in the environment.

        This is a `num_drones` list of (4, 3) arrays, where each element in the list corresponds to the i-th drone state.

        Similar to the `state` property, the states contain information corresponding to:
            - `state[0, :]` represents body frame angular velocity
            - `state[1, :]` represents ground frame angular position
            - `state[2, :]` represents body frame linear velocity
            - `state[3, :]` represents ground frame linear position

        This function is not very optimized, if you want the state of a single drone, do `state(i)`.

        Returns:
            np.ndarray: list of states
        """
        states = []
        for drone in self.drones:
            states.append(drone.flight_state)

        return states

    @property
    def all_aux_states(self) -> list[np.ndarray]:
        """Returns a list of auxiliary states for all drones in the environment.

        This is a `num_drones` list of auxiliary states.

        This function is not very optimized, if you want the aux state of a single drone, do `aux_state(i)`.

        Returns:
            np.ndarray: list of auxiliary states
        """
        aux_states = []
        for drone in self.drones:
            aux_states.append(drone.aux_state)

        return aux_states

    def print_all_bodies(self):
        """Debugging function used to print out all bodies in the environment along with their IDs."""
        bodies = dict()
        for i in range(self.getNumBodies()):
            bodies[i] = self.getBodyInfo(i)[-1].decode("UTF-8")

        from pprint import pprint

        pprint(bodies)

    def set_mode(self, flight_mode: int):
        """Sets the flight control mode of each drone in the environment.

        Args:
            flight_modes (int | list[int]): flight mode
        """

        for drone in self.drones:
            drone.set_mode(flight_mode)

    def set_setpoint(self, index: DroneIndex, setpoint: np.ndarray):
        """Sets the setpoint of one drone in the environment.

        Args:
            index (DRONE_INDEX): index
            setpoint (np.ndarray): setpoint
        """
        self.drones[index].setpoint = setpoint

    def set_all_setpoints(self, setpoints: np.ndarray):
        """Sets the setpoints of each drone in the environment.

        Args:
            setpoints (np.ndarray): list of setpoints
        """

        for i, drone in enumerate(self.drones):
            drone.set_setpoint(setpoint=setpoints[i])

    def set_loyalwingman_setpoint(self, setpoint: np.ndarray):
        self.drones[0].set_setpoint(setpoint)

    def set_loiteringmunition_setpoint(self, setpoint: np.ndarray):
        self.drones[1].set_setpoint(setpoint)

    def replace_loiteringmunition(self):
        new_position = np.random.rand(3)
        quaternion = self.getQuaternionFromEuler(ornObj=np.random.rand(3))
        self.resetBasePositionAndOrientation(
            bodyUniqueId=self.drones[1], posObj=np.random.rand(3), ornObj=quaternion
        )
        self.drones[1].reset()
        self.drones[1].set_setpoint(
            np.array(new_position[0], new_position[1], 0, new_position[2])
        )
        self.drones[1].update_state()
        self.drones[1].update_last()

    def _render_step(self):
        # compute rtf if we're rendering
        if self.render:
            elapsed = time.time() - self.now
            self.now = time.time()

            self._sim_elapsed += self.update_period * self.updates_per_step
            self._frame_elapsed += elapsed

            # time.sleep(max(self._sim_elapsed - self._frame_elapsed, 0.0))

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

    def step(self, action_commands):
        """Steps the environment, this automatically handles physics and control looprates, one step is equivalent to one control loop step."""
        self.set_loyalwingman_setpoint(action_commands)

        for step in range(self.updates_per_step):
            # update onboard avionics conditionally
            for drone in self.drones:
                drone.update_control()
                drone.update_physics()
                drone.update_state()

            self.stepSimulation()

            self.physics_steps += 1
            self.elapsed_time = self.physics_steps / self.physics_hz

        self.aviary_steps += 1

    # =======================================================================
    #  Drones Management
    # =======================================================================

    def _create_drone(self, position: np.ndarray, quadcopter_type: QuadcopterType):
        position = np.random.rand(3)
        orientation = self.getQuaternionFromEuler(np.zeros(3))
        drone = Quadcopter.from_quadx(
            p=self,
            start_pos=position,
            start_orn=orientation,
            physics_hz=self.physics_hz,
            np_random=self.np_random,
            quadcopter_type=quadcopter_type,
        )

        self.drones.append(drone)

    def create_persuer(self, num_drones: int, positions: np.ndarray):
        # Supondo que cada linha do ndarray 'positions' contém as coordenadas x, y, z de um drone.
        if positions.shape[0] != num_drones:
            raise ValueError(
                "Número de posições fornecidas não corresponde ao número de drones solicitados."
            )

        for i in range(num_drones):
            position = positions[i]
            self._create_drone(position, QuadcopterType.LOYALWINGMAN)

    def create_invader(self, num_drones: int, positions: np.ndarray):
        if positions.shape[0] != num_drones:
            raise ValueError(
                "Número de posições fornecidas não corresponde ao número de drones solicitados."
            )

        for i in range(num_drones):
            position = positions[i]
            self._create_drone(position, QuadcopterType.LOITERINGMUNITION)
