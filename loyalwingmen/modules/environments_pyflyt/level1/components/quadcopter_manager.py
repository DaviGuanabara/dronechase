import os, sys

import numpy as np
from enum import Enum
from typing import Optional, Union, NamedTuple
from loyalwingmen.modules.quadcoters.components.base.quadcopter import (
    Quadcopter,
    QuadcopterType,
)
from loyalwingmen.modules.environments_pyflyt.level1.components.pyflyt_level1_simulation import (
    AviarySimulation as AviaryL1,
)


class QuadcopterBlueprint(NamedTuple):
    position: np.ndarray
    orientation: np.ndarray
    quadcopter_type: QuadcopterType
    quadcopter_name: str = "no_name"


class QuadcopterManager:
    def __init__(self, simulation: AviaryL1) -> None:
        self.simulation = simulation
        self.blueprints: list[QuadcopterBlueprint] = []

    # ===========================================================================
    # Creation of Quadcopters
    # ===========================================================================

    def spawn_persuer(
        self, num_drones: int, position: np.ndarray, name: str = "no_name"
    ):
        if num_drones < 1:
            raise ValueError("Número de drones deve ser maior que 1.")

        initial_len = len(self.simulation.drones)
        for i in range(num_drones):
            drone = Quadcopter.spawn_loyalwingman(self.simulation, position[i], name)
            drone.update_imu()
            drone.set_mode(6)
            data = QuadcopterBlueprint(
                position[i], np.zeros(3), QuadcopterType.LOYALWINGMAN
            )
            self.blueprints.append(data)
            self.simulation.drones.append(drone)
        return self.simulation.drones[initial_len:]
    

    def spawn_invader(
        self, num_drones: int, positions: np.ndarray, name: str = "no_name"
    ):
        if num_drones < 1:
            raise ValueError("Número de drones deve ser maior que 1.")

        initial_len = len(self.simulation.drones)
        for i in range(num_drones):
            drone = Quadcopter.spawn_loiteringmunition(
                self.simulation, positions[i], name
            )

            drone.update_imu()
            drone.set_mode(7)
            drone.drive(
                np.array([positions[i][0], positions[i][1], 0, positions[i][2]])
            )
            data = QuadcopterBlueprint(
                positions[i], np.zeros(3), QuadcopterType.LOITERINGMUNITION
            )
            self.blueprints.append(data)
            self.simulation.drones.append(drone)
        return self.simulation.drones[initial_len:]

    def get_invaders(self) -> "list[Quadcopter]":
        drones: "list[Quadcopter]" = self.simulation.drones
        return [
            drone
            for drone in drones
            if drone.quadcopter_type == QuadcopterType.LOITERINGMUNITION
        ]

    def get_persuers(self) -> "list[Quadcopter]":
        drones: "list[Quadcopter]" = self.simulation.drones
        return [
            drone
            for drone in drones
            if drone.quadcopter_type == QuadcopterType.LOYALWINGMAN
        ]

    def spawn_quadcopters_from_blueprints(self):
        # Filtra e agrupa os blueprints por tipo
        loyalwingman_positions = [
            blueprint.position
            for blueprint in self.blueprints
            if blueprint.quadcopter_type == QuadcopterType.LOYALWINGMAN
        ]

        loiteringmunition_positions = [
            blueprint.position
            for blueprint in self.blueprints
            if blueprint.quadcopter_type == QuadcopterType.LOITERINGMUNITION
        ]

        # Cria os quadcopters de uma vez
        if loyalwingman_positions:
            positions_np = np.array(loyalwingman_positions)
            self.spawn_persuer(
                len(loyalwingman_positions), positions_np, name="no_name"
            )

        if loiteringmunition_positions:
            positions_np = np.array(loiteringmunition_positions)
            self.spawn_invader(
                len(loiteringmunition_positions), positions_np, name="no_name"
            )

    def update_control_physics_imu(self, drone: Quadcopter):
        drone.update_control()
        drone.update_physics()
        # drone.update_state()
        # update imu is a more meaningful name
        # describing that Lidar or other sensors are not updated.
        drone.update_imu()

    def reset_quadcopters(self):
        drones: "list[Quadcopter]" = self.simulation.drones
        for drone in drones:
            drone.detach_from_simulation()

        self.spawn_quadcopters_from_blueprints()

    def replace_all_to_starting_position(self):
        drones: "list[Quadcopter]" = self.simulation.drones
        for i in range(len(drones)):
            drones[i].replace(
                self.blueprints[i].position, self.blueprints[i].orientation
            )

    def replace_quadcopter(
        self, drone: Quadcopter, position: np.ndarray, attitude: np.ndarray
    ):
        drone.replace(position, attitude)

    def replace_invader(
        self, invader: Quadcopter, position: np.ndarray, attitude: np.ndarray
    ):
        print("replace invader")
        invader.replace(position, attitude)
        motion_command = np.array([position[0], position[1], 0, position[2]])
        # TODO: set_setpoint must be removed. It is a temporary solution
        # The solution should be using the 'drive' method
        # But, is necessary to put conditions there to work in different 'modes'
        invader.set_setpoint(motion_command)
        invader.update_imu()
        invader.update_control()
        invader.update_physics()

    # ===========================================================================
    # Removers of Quadcopters
    # ===========================================================================
