import os, sys

import numpy as np
from enum import Enum
from typing import Optional, Union, NamedTuple
from loyalwingmen.modules.quadcoters.components.base.quadcopter import (
    Quadcopter,
    QuadcopterType,
)
from .pyflyt_level3_simulation import (
    L3AviarySimulation as AviaryL3,
)



class QuadcopterBlueprint(NamedTuple):
    position: np.ndarray
    orientation: np.ndarray
    quadcopter_type: QuadcopterType
    quadcopter_name: str = "no_name"


class QuadcopterManager:
    def __init__(self, simulation: AviaryL3, debug_on:bool = False) -> None:
        self.simulation = simulation
        self.blueprints: list[QuadcopterBlueprint] = []
        self.debug_on = debug_on

    def get_quadcopter_by_id(self, id: int) -> Optional[Quadcopter]:
        drones: "list[Quadcopter]" = self.simulation.drones
        return next((drone for drone in drones if drone.id == id), None)
    
    def get_quadcopters(self) -> "list[Quadcopter]":
        drones: "list[Quadcopter]" = self.simulation.drones
        return drones
    
    # ===========================================================================
    # Creation of Quadcopters
    # ===========================================================================

    def spawn_pursuer(self, positions: np.ndarray, name: str = "no_name", lidar_radius:float = 5):
        if positions.size < 1:
            raise ValueError("Número de drones deve ser maior que 1.")

        initial_len = len(self.simulation.drones)
        for position in positions:
            drone = Quadcopter.spawn_loyalwingman(self.simulation, position, name, debug_on=False, lidar_on=True, lidar_radius=lidar_radius)
            drone.update_imu()

            data = QuadcopterBlueprint(
                position, np.zeros(3), QuadcopterType.LOYALWINGMAN
            )
            self.blueprints.append(data)
            self.simulation.drones.append(drone)
        return self.simulation.drones[initial_len:]

    def spawn_invader(self, positions: np.ndarray, name: str = "no_name"):
        if positions.size < 1:
            raise ValueError("Número de drones deve ser maior que 1.")

        initial_len = len(self.simulation.drones)
        for position in positions:
            drone = Quadcopter.spawn_loiteringmunition(self.simulation, position, name, debug_on=False, lidar_on=False)
            drone.update_imu()
            data = QuadcopterBlueprint(
                position, np.zeros(3), QuadcopterType.LOITERINGMUNITION
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

    def get_pursuers(self) -> "list[Quadcopter]":
        drones: "list[Quadcopter]" = self.simulation.drones
        return [
            drone
            for drone in drones
            if drone.quadcopter_type == QuadcopterType.LOYALWINGMAN
        ]

    def delete_invader_blueprints(self):
        self.blueprints = [
            blueprint
            for blueprint in self.blueprints
            if blueprint.quadcopter_type != QuadcopterType.LOITERINGMUNITION
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
            self.spawn_pursuer(
                positions_np, name="no_name"
            )

        if loiteringmunition_positions:
            positions_np = np.array(loiteringmunition_positions)
            self.spawn_invader(
                positions_np, name="no_name"
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
        
    def clear_invaders(self):
        drones: "list[Quadcopter]" = self.simulation.drones

        for drone in drones:
            if drone.quadcopter_type == QuadcopterType.LOITERINGMUNITION:
                drone.detach_from_simulation()

        self.simulation.drones = [drone for drone in drones if drone.quadcopter_type != QuadcopterType.LOITERINGMUNITION]
        self.delete_invader_blueprints()
        
    def clear_pursuers(self):
        drones: "list[Quadcopter]" = self.simulation.drones

        for drone in drones:
            if drone.quadcopter_type == QuadcopterType.LOYALWINGMAN:
                drone.detach_from_simulation()

        self.simulation.drones = [drone for drone in drones if drone.quadcopter_type != QuadcopterType.LOYALWINGMAN]
        self.delete_invader_blueprints()

    def delete_invader_by_id(self, ids: "list[int]"):
        
        drones: "list[Quadcopter]" = self.simulation.drones

        # Convert ids to a set for faster membership checks
        ids_set = set(ids)

        # Predicate function using lambda to check if a drone should be removed
        should_remove = lambda drone: drone.quadcopter_type == QuadcopterType.LOITERINGMUNITION and drone.id in ids_set

        # Detach invaders matching the criteria
        invaders_to_detach = list(filter(should_remove, drones))
        if len(invaders_to_detach) > 0:
            for invader in invaders_to_detach:
                invader.detach_from_simulation()
                
            # Update the drone list to exclude the detached invaders
            self.simulation.drones = [drone for drone in drones if drone.id != -1]


    def replace_all_to_starting_position(self):
        drones: "list[Quadcopter]" = self.simulation.drones
        for i in range(len(drones)):
            drones[i].replace(
                self.blueprints[i].position, self.blueprints[i].orientation
            )
            drones[i].update_imu()

    def replace_quadcopter(
        self, drone: Quadcopter, position: np.ndarray, attitude: np.ndarray
    ):
        drone.replace(position, attitude)

    def replace_invader(
        self, invader: Quadcopter, position: np.ndarray, attitude: np.ndarray
    ):
        
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
    # Drive Quadcopters
    # ===========================================================================

    def drive_invaders(self):
        invaders = self.get_invaders()

        # 1. Stack all invader positions into a single numpy array.
        positions = np.array([invader.inertial_data["position"] for invader in invaders])
        
        # 2. Negate this array.
        inverted_positions = -positions

        # 3. Add a column of 0.3s to the negated array.
        motion_commands = np.hstack((inverted_positions, 0.2 * np.ones((len(invaders), 1))))

        # Drive each invader with the corresponding motion command
        for invader, command in zip(invaders, motion_commands):
            invader.drive(command)