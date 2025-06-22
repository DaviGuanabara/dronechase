import os
import sys

import numpy as np
from enum import Enum
from typing import Optional, Union, NamedTuple, Dict, List, cast

from core.entities.quadcopters.quadcopter import Quadcopter
from core.entities.entity_type import EntityType

# from ..simulation.level4_simulation import L4AviarySimulation as SimulationAviaryL4
import threading
from core.entities.immovable_structures.immovable_structures import ImmovableStructures
from core.entities.navigators.loitering_munition_navigator import KamikazeNavigator


class EntitiesManager:
    # ===========================================================================
    # Single Threaded Singleton
    # Thread-Local Singleton
    # This is a thread-local singleton. Each thread will have its own instance of the class.
    # ensuring the multiple environment creation in training.
    # ===========================================================================

    DEFAULT_LIDAR_RADIUS = 5
    _thread_local_data = threading.local()

    def __new__(cls):
        if not hasattr(cls._thread_local_data, "_instance"):
            cls._thread_local_data._instance = super(
                EntitiesManager, cls).__new__(cls)
            cls._thread_local_data._instance._initialize()
        return cls._thread_local_data._instance

    def _initialize(self) -> None:
        """Initialize attributes for the instance."""
        self.drone_registry: Dict[int, Quadcopter] = {}
        self.kamikaze_navigator = KamikazeNavigator(np.array([0, 0, 1]))
        self.debug = False

        self.agent_id = -1

    # ===========================================================================
    # Lazy Initialization
    # On purpuse, i am not initializing the simulation on _initialize, to ensure
    # that the user will call the setup_simulation method, or
    # quadcopter manager will not work.
    # ===========================================================================

    def setup_simulation(self, simulation):
        self.simulation = simulation

    def setup_debug(self, debug: bool):
        self.debug = debug

    # def __init__(self, simulation: AviaryL4, debug_on: bool = False) -> None:
    #    self.simulation = simulation
    #    self.debug_on = debug_on
    #    self.drone_registry: Dict[int, Quadcopter] = {}

    # ===========================================================================
    # Creation of Quadcopters
    # ===========================================================================

    def spawn_quadcopter(
        self,
        positions: np.ndarray,
        quadcopter_type: EntityType,
        names: Union[List[str], str] = "no_name",
        lidar_radius: float = DEFAULT_LIDAR_RADIUS,
        use_fused_lidar: bool = False,
    ) -> List[Quadcopter]:
        # Check if names is a list, if not, convert it to a list
        if not isinstance(names, list):
            names = [names]

        # Ensure names list is the same size as positions, filling gaps with "quadcopter"
        if len(names) < len(positions):
            names.extend(["quadcopter"] * (len(positions) - len(names)))

        drones = []
        for i, position in enumerate(positions):
            name = names[i] if i < len(names) else "quadcopter"

            if quadcopter_type == EntityType.LOYALWINGMAN:
                drone = Quadcopter.spawn_loyalwingman(
                    self.simulation,
                    position,
                    name,
                    debug_on=False,
                    lidar_on=True,
                    lidar_radius=lidar_radius,
                    use_fused_lidar=use_fused_lidar,
                )
            else:
                drone = Quadcopter.spawn_loiteringmunition(
                    self.simulation,
                    position,
                    name,
                    debug_on=False,
                    lidar_on=True,
                    lidar_radius=lidar_radius,
                )

            self.simulation.active_drones[drone.id] = drone
            drone.update_imu()
            self.drone_registry[drone.id] = drone
            drones.append(drone)

        return drones

    def spawn_pursuer(
        self,
        positions: np.ndarray,
        names: Union[List[str], str] = "no_name",
        lidar_radius: float = DEFAULT_LIDAR_RADIUS,
        use_fused_lidar=False
    ) -> List[Quadcopter]:

        print("[DEBUG] EntitiesManager: spawning pursuer")

        return self.spawn_quadcopter(
            positions, EntityType.LOYALWINGMAN, names, lidar_radius, use_fused_lidar=use_fused_lidar
        )

    def spawn_invader(
        self, positions: np.ndarray, names: Union[List[str], str] = "no_name"
    ) -> List[Quadcopter]:
        return self.spawn_quadcopter(positions, EntityType.LOITERINGMUNITION, names)

    # ===========================================================================
    # Creation of Static Structures
    # ===========================================================================

    def spawn_ground(self, dome_radius: float):
        # TODO: IT IS HARD CODED INTO THE TASK. I HAVE TO CHANGE IT.
        self.ground = ImmovableStructures.spawn_ground(
            self.simulation, [0, 0, -6], dome_radius
        )

    def spawn_protected_building(self):
        # self.building = ImmovableStructures.spawn_building(self.simulation, [0, 0, 0])
        pass

    # ===========================================================================
    # Get Quadcopters
    # ===========================================================================

    def filter_expression(
        self,
        drone: Quadcopter,
        quadcopter_type: Optional[EntityType],
        armed: Optional[bool],
        id: Optional[int],
    ) -> bool:
        return all([
            quadcopter_type is None or drone.quadcopter_type == quadcopter_type,
            armed is None or drone.armed == armed,
            id is None or drone.id == id,
        ])

    def get_quadcopters(
        self,
        *,
        quadcopter_type: Optional[EntityType] = None,
        armed: Optional[bool] = None,
        id: Optional[int] = None,
    ) -> List[Quadcopter]:
        drones = list(self.drone_registry.values())
        return [
            drone
            for drone in drones
            if self.filter_expression(drone, quadcopter_type, armed, id)
        ]

    def get_all(self) -> "list[Quadcopter]":
        return list(self.drone_registry.values())

    def get_armed_invaders(self) -> "list[Quadcopter]":
        return self.get_quadcopters(
            quadcopter_type=EntityType.LOITERINGMUNITION, armed=True
        )

    def get_disarmed_invaders(self) -> "list[Quadcopter]":
        return self.get_quadcopters(
            quadcopter_type=EntityType.LOITERINGMUNITION, armed=False
        )

    def get_armed_pursuers(self) -> "list[Quadcopter]":
        return self.get_quadcopters(quadcopter_type=EntityType.LOYALWINGMAN, armed=True)

    def get_disarmed_pursuers(self) -> "list[Quadcopter]":
        return self.get_quadcopters(
            quadcopter_type=EntityType.LOYALWINGMAN, armed=False
        )

    def get_all_armed(self) -> "list[Quadcopter]":
        return self.get_quadcopters(armed=True)

    def get_all_invaders(self) -> "list[Quadcopter]":
        return self.get_quadcopters(quadcopter_type=EntityType.LOITERINGMUNITION)

    def get_all_pursuers(self) -> "list[Quadcopter]":
        return self.get_quadcopters(quadcopter_type=EntityType.LOYALWINGMAN)

    # def get_rl_agent(self)

    # ===========================================================================
    # Arm and Disarm.
    # ===========================================================================

    # TODO: I have to replace the disarmed quadcopters to not be an invisible obstacle that can be collided with.
    def arm_by_ids(self, ids: "list[int]"):
        for id in ids:
            self.arm_by_quadcopter(self.drone_registry[id])

    def disarm_by_ids(self, ids: "list[int]"):
        for id in ids:
            self.disarm_by_quadcopter(self.drone_registry[id])

    def arm_all(self):
        for drone in list(self.drone_registry.values()):
            # print(f"Arming Quadcopter {drone.id}")
            self.arm_by_quadcopter(drone)

    def arm_all_pursuers(self):
        for drone in list(self.drone_registry.values()):
            if drone.quadcopter_type == EntityType.LOYALWINGMAN:
                self.arm_by_quadcopter(drone)

    def disarm_all(self):
        for drone in list(self.drone_registry.values()):
            self.disarm_by_quadcopter(drone)

    def disarm_all_invaders(self):
        for drone in list(self.drone_registry.values()):
            if drone.quadcopter_type == EntityType.LOITERINGMUNITION:
                self.disarm_by_quadcopter(drone)

    def arm_by_quadcopter(self, quadcopter: Quadcopter):
        quadcopter.arm()
        self.simulation.active_drones[quadcopter.id] = quadcopter

    def disarm_by_quadcopter(self, quadcopter: Quadcopter):
        quadcopter.disarm()
        self.simulation.active_drones.pop(quadcopter.id, None)

    # ===========================================================================
    # Shoot
    # ===========================================================================

    def shoot_by_ids(self, from_id: int, target_id: int):
        from_drone = self.drone_registry.get(from_id)
        target_drone = self.drone_registry.get(target_id)

        if from_drone is None or target_drone is None:
            return False

        if from_drone.shoot():
            self.disarm_by_quadcopter(target_drone)
            return True
        return False

    # ===========================================================================
    # Drive Quadcopters
    # ===========================================================================

    def drive_invaders(self, offset_handler):
        invaders: List[Quadcopter] = self.get_armed_invaders()

        for invader in invaders:
            self.kamikaze_navigator.update(invader, offset_handler)

    """
        def drive_invaders(self):
            invaders = self.get_armed_invaders()

            if not invaders:  # and self.debug_on:
                print("No armed invaders to drive.")
                print(self.simulation.active_drones)
                return

            pursuers = self.get_armed_pursuers()
            # Check if there are any pursuers
            if not pursuers:
                # if self.debug_on:
                #    print("No armed pursuers present.")
                return

            # 1. Stack all invader positions into a single numpy array.
            positions = np.array(
                [invader.inertial_data["position"] for invader in invaders]
            )
            target_positions = np.array(
                [pursuer.inertial_data["position"] for pursuer in pursuers]
            )

            # 2. Negate this array.
            # ONLY ONE PURSUER
            inverted_positions = target_positions[0] - positions

            # 3. Add a column of 0.3s to the negated array.
            motion_commands = np.hstack(
                (inverted_positions, 0.2 * np.ones((len(invaders), 1)))
            )

            # Drive each invader with the corresponding motion command
            for invader, command in zip(invaders, motion_commands):
                invader.drive(command)



    """

    # ===========================================================================
    # Others
    # ===========================================================================
    def update_control_physics_imu(self, drone: Quadcopter):
        drone.update_control()
        drone.update_physics()
        # drone.update_state()
        # update imu is a more meaningful name
        # describing that Lidar or other sensors are not updated.
        drone.update_imu()

    def replace_quadcopter(
        self,
        drone: Quadcopter,
        position: np.ndarray,
        attitude: np.ndarray = np.zeros(3),
    ):
        drone.replace(position, attitude)

    def replace_quadcopters(self, drones: List[Quadcopter], positions: np.ndarray):
        for drone, position in zip(drones, positions):
            self.replace_quadcopter(drone, position)

    def replace_quadcopters_by_ids(self, ids: List[int], positions: np.ndarray):
        drones: List[Quadcopter] = cast(
            List[Quadcopter],
            [
                self.drone_registry.get(id)
                for id in ids
                if self.drone_registry.get(id) is not None
            ],
        )
        self.replace_quadcopters(drones, positions)

    def _select_loyalwingman_randomly(self, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random.RandomState()
        return (
            rng.choice(candidate_ids)
            if (
                candidate_ids := [
                    d_id
                    for d_id, quad in self.drone_registry.items()
                    if quad.quadcopter_type == EntityType.LOYALWINGMAN
                ]
            )
            else -1
        )

    def set_agent(self, drone_id: int = -1, rng: Optional[np.random.RandomState] = None) -> bool:
        """
        Sets the RL agent among the registered drones.

        If `drone_id` is -1, a random LOYALWINGMAN will be selected.

        Returns True if an agent was successfully set, False otherwise.
        """
        if drone_id == -1:
            drone_id = self._select_loyalwingman_randomly(rng)

        if drone_id not in self.drone_registry:
            return False

        self.agent_id = drone_id

        agent = self.drone_registry[drone_id]
        agent.set_as_agent()
        return True

    def get_agent_id(self):
        return self.agent_id

    def get_agent(self):
        return self.drone_registry[self.agent_id]

    def get_allies(self, armed: bool = False) -> List[Quadcopter]:
        agent_id = self.get_agent_id()
        pursuers = self.get_all_pursuers()

        allies = [drone for drone in pursuers if drone.id != agent_id]

        if armed:
            allies = [drone for drone in allies if drone.armed]

        return allies
