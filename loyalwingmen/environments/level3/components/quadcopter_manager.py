import os, sys

import numpy as np
from enum import Enum
from typing import Optional, Union, NamedTuple, Dict, List, cast
from loyalwingmen.modules.quadcoters.components.base.quadcopter import (
    Quadcopter,
    ObjectType,
)
from .pyflyt_level3_simulation import (
    L3AviarySimulation as AviaryL3,
)

class QuadcopterManager:
    def __init__(self, simulation: AviaryL3, debug_on:bool = False) -> None:
        self.simulation = simulation
        self.debug_on = debug_on
        self.drone_registry: Dict[int, Quadcopter] = {}

    
    # ===========================================================================
    # Creation of Quadcopters
    # ===========================================================================
    
    def spawn_quadcopter(self, positions: np.ndarray, quadcopter_type: ObjectType, name: str = "no_name", lidar_radius:float = 5):
        
        instantiator = Quadcopter.spawn_loyalwingman if quadcopter_type == ObjectType.LOYALWINGMAN else Quadcopter.spawn_loiteringmunition
            
        for position in positions:
            drone = instantiator(self.simulation, position, name, debug_on=False, lidar_on=True, lidar_radius=lidar_radius)
            self.simulation.active_drones[drone.id] = drone
            drone.update_imu()
            self.drone_registry[drone.id] = drone

    def spawn_pursuer(self, positions: np.ndarray, name: str = "no_name", lidar_radius:float = 5):
        self.spawn_quadcopter(positions, ObjectType.LOYALWINGMAN, name, lidar_radius)
       
    def spawn_invader(self, positions: np.ndarray, name: str = "no_name"):
        self.spawn_quadcopter(positions, ObjectType.LOITERINGMUNITION, name)

    # ===========================================================================
    # Get Quadcopters
    # ===========================================================================
    
    def filter_expression(self, drone: Quadcopter, quadcopter_type: Optional[ObjectType], armed: Optional[bool], id: Optional[int]) -> bool:
        return (
            (quadcopter_type is None or drone.quadcopter_type == quadcopter_type)
            and (armed is None or drone.armed == armed)
            and (id is None or drone.id == id)
        )

    def get_quadcopters(self, *, quadcopter_type: Optional[ObjectType] = None, armed: Optional[bool] = None, id: Optional[int] = None) -> List[Quadcopter]:

        drones = list(self.drone_registry.values())
        return [
            drone for drone in drones
            if self.filter_expression(drone, quadcopter_type, armed, id)
        ]
        
        
    def get_armed_invaders(self) -> "list[Quadcopter]":
        return self.get_quadcopters(quadcopter_type=ObjectType.LOITERINGMUNITION, armed=True)

    def get_disarmed_invaders(self) -> "list[Quadcopter]":
        return self.get_quadcopters(quadcopter_type=ObjectType.LOITERINGMUNITION, armed=False)

    def get_armed_pursuers(self) -> "list[Quadcopter]":
        return self.get_quadcopters(quadcopter_type=ObjectType.LOYALWINGMAN, armed=True)

    def get_disarmed_pursuers(self) -> "list[Quadcopter]":
        return self.get_quadcopters(quadcopter_type=ObjectType.LOYALWINGMAN, armed=False)

    def get_all_invaders(self) -> "list[Quadcopter]":
        return self.get_quadcopters(quadcopter_type=ObjectType.LOITERINGMUNITION)
    
    def get_all_pursuers(self) -> "list[Quadcopter]":
        return self.get_quadcopters(quadcopter_type=ObjectType.LOYALWINGMAN)
            
    # ===========================================================================
    # Arm and Disarm.
    # ===========================================================================
    
    #TODO: I have to replace the disarmed quadcopters to not be an invisible obstacle that can be collided with.
    def arm_by_ids(self, ids: "list[int]"):
        for id in ids:
            self.arm_by_quadcopter(self.drone_registry[id])
            
    def disarm_by_ids(self, ids: "list[int]"):
        
        for id in ids:
            self.disarm_by_quadcopter(self.drone_registry[id])

    def arm_all(self):
        for drone in list(self.drone_registry.values()):
            #print(f"Arming Quadcopter {drone.id}")
            self.arm_by_quadcopter(drone)
            
    def disarm_all(self):
        for drone in list(self.drone_registry.values()):
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
        from_quadcopter = self.drone_registry[from_id]
        #print(f"Quadcopter can fire ? {from_quadcopter.gun.can_fire()} - Gun Step: {from_quadcopter.gun.current_step} - lidar step:{from_quadcopter.lidar.current_step}") 
        if shot_successful := from_quadcopter.gun.shoot():
            #print("shoot successful")
            self.disarm_by_ids([target_id])
            return shot_successful
        
        #print("shoot unsuccessful")
        return False


    # ===========================================================================
    # Drive Quadcopters
    # ===========================================================================

    def drive_invaders(self):
        invaders = self.get_armed_invaders()
        
        if not invaders and self.debug_on:
            print("No armed invaders to drive.")
            print(self.simulation.active_drones)
            return

        pursuers = self.get_armed_pursuers()
        # Check if there are any pursuers
        if not pursuers:
            if self.debug_on:
                print("No armed pursuers present.")
            return
        
        # 1. Stack all invader positions into a single numpy array.
        positions = np.array([invader.inertial_data["position"] for invader in invaders])
        target_positions = np.array([pursuer.inertial_data["position"] for pursuer in pursuers])
        
        # 2. Negate this array.
        #ONLY ONE PURSUER
        inverted_positions = target_positions[0] - positions

        # 3. Add a column of 0.3s to the negated array.
        motion_commands = np.hstack((inverted_positions, 0.2 * np.ones((len(invaders), 1))))

        # Drive each invader with the corresponding motion command
        for invader, command in zip(invaders, motion_commands):
            invader.drive(command)
            
            
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
        self, drone: Quadcopter, position: np.ndarray, attitude: np.ndarray = np.zeros(3)
    ):
        drone.replace(position, attitude)
        
    def replace_quadcopters(self, drones: List[Quadcopter], positions: np.ndarray):
        for drone, position in zip(drones, positions):
            self.replace_quadcopter(drone, position)
            
    def replace_quadcopters_by_ids(self, ids: List[int], positions: np.ndarray):
        drones:List[Quadcopter] = cast(List[Quadcopter],[self.drone_registry.get(id) for id in ids if self.drone_registry.get(id) is not None])
        self.replace_quadcopters( drones, positions)


        
    