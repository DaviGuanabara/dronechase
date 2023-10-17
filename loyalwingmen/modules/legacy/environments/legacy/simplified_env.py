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


from modules.factories.drone_factory import DroneFactory
from modules.factories.loiteringmunition_factory import (
    LoiteringMunitionFactory,
    LoiteringMunition,
)
from modules.factories.loyalwingman_factory import LoyalWingmanFactory, LoyalWingman

from modules.dataclasses.dataclasses import EnvironmentParameters
from modules.models.lidar import CoordinateConverter
from typing import List
from modules.models.lidar import Channels

from pathlib import Path
import os
from typing import Tuple
import math

class DroneChaseEnvLevel1(Env):
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
        gravity: float = 9.81,
        GUI: bool = False,
        seed: int = 0
        
    ):
        np.random.seed(seed)
        self.simulation = Simulation(simulation_frequency=simulation_frequency, gravity=gravity, GUI=GUI)
         
        self.observation_space = self._observationSpace(self.simulation.get_observation_size())
        self.action_space = self._actionSpace(self.simulation.get_action_size())
        
        self.aggregate_physics_steps = int(simulation_frequency / rl_frequency)  
    
    def _observationSpace(self, shape_size: int):
        return spaces.Box(low=-1, high=1, shape=(shape_size,), dtype=np.float32)
        
    
    def _actionSpace(self, shape_size):
        return spaces.Box(low=-1, high=1, shape=(shape_size,), dtype=np.float32)
    
    def step(self, action):
        self.simulation.set_action(action)
        return self.simulation.step(aggregate_physics_steps=self.aggregate_physics_steps)
    
    def reset(self, seed: int = 0):
        return self.simulation.reset(seed)
        

class Simulation():
    def __init__(self, simulation_frequency: int = 240, gravity: float = 9.81, GUI: bool = False, dome_radius:int = 10, max_velocity: float = 1, action_type: int = 1, initial_position: np.ndarray = np.array([3, 3, 3])):
        self.setup_pybullet(simulation_frequency=simulation_frequency, gravity=gravity, GUI=GUI)
        
        self.dome_radius = dome_radius
        self.MAX_VELOCITY = max_velocity
        self.action_type = action_type
        
        self.initial_position = initial_position
        self.drone_id = self.create_drone(initial_position)   
        self.target_position = np.array([0, 0, 0])
        self.last_action = np.zeros(self.get_action_size())
        self.MAX_TIME = 5
        self.RESET_TIME = time.time()
        
        
   
        
         
    def setup_pybullet(self, simulation_frequency, gravity, GUI: bool = False):

        if GUI:
            self.client_id = self.setup_pybulley_GUI()
            
        else:
            self.client_id = self.setup_pybullet_DIRECT()
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())    
            
        p.setGravity(
            0,
            0,
            0,#-gravity,
            physicsClientId=self.client_id,
        )
        
        p.setTimeStep(
            1 / simulation_frequency,
            physicsClientId=self.client_id,
        )
    
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
    
    def sample_spherical(self):
        radius = np.random.rand() * self.dome_radius
        random_position = np.random.uniform(-1, 1, (1, 3))[0]
    
        magnitude = np.linalg.norm(random_position, axis=0)
        
        return radius * random_position / magnitude
        
    def reset(self, seed: int = 0):
        #np.random.seed(seed)
        random_position = self.sample_spherical()
        p.resetSimulation(physicsClientId=self.client_id)
        self.drone_id = self.create_drone(random_position)
        
        #p.resetBasePositionAndOrientation(
        #        self.drone_id,
        #        posObj=random_position,
        #        ornObj=np.array([0, 0, 0, 1]),  # [x,y,z,w]
        #        physicsClientId=self.client_id,
        #    )
        
        self.RESET_TIME = time.time()
        
        return self.compute_observation(), self.compute_info() 
    
    def get_action_size(self):
        if self.action_type == 0:
            return 3
        
        if self.action_type == 1:
            return 3
        
        if self.action_type == 2:
            return 4
        
        return 3
    
    def get_observation_size(self):
        state = self.get_state()
        size = 0
        for key in state:
            array: np.ndarray = state[key]
            size += array.size
        
        
        return size
    

    #####################################################################################################
    # Drone Functions
    #####################################################################################################
    
    def create_drone(self, position) -> int:
        urdf_name = "cf2x"  + ".urdf"
        base_path = str(Path(os.getcwd()).parent.absolute())
        
        urdf_file_path = base_path + "\\assets\\" + urdf_name  # "cf2x.urdf"

        id = p.loadURDF(
            fileName=urdf_file_path,
            basePosition=position,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client_id,
        )
        
        return id
    
    def set_action(self, action: np.ndarray):
        
        self.last_action = action
        
        while action.size < 3:
            action = np.concatenate([action, [0]])
            
        if self.action_type == 0:
            action = self.MAX_VELOCITY * action
            
            p.resetBaseVelocity(
                self.drone_id,
                action,
                np.array([0, 0, 0]),
                physicsClientId=self.client_id,
            )
            
        if self.action_type == 1:
            p.applyExternalForce(
                objectUniqueId=self.drone_id,
                linkIndex=-1,
                forceObj=action,
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client_id,
                
            )
            
        if self.action_type == 2:
            position, orientation = p.getPositionAndOrientation(
                self.drone_id,
                physicsClientId=self.client_id,
            )
            p.resetBasePositionAndOrientation(
                self.drone_id,
                position,
                action[:3],
                physicsClientId=self.client_id,
            )
            p.applyExternalForce(
                objectUniqueId=self.drone_id,
                linkIndex=-1,
                forceObj=[0, 0, action[3]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client_id,
                
            )
        
        #
        """
        Abaixo está uma das observações adquiridas durante o treinamento do drone, com 40 Milhões de steps. Tem algo muito esquisito. 
        A velcidade linear [-2.6049018e-02 -3.2533258e-02 -3.2603025e-01] está diferente da última ação: [-2.6089132e-02
        -3.2583356e-02  7.4040890e-04], eles deveriam estar iguais. Deve ser algum problema com o pybullet. Talvez essa função não foi feita
        para ser atualizada a cada passo de tempo.
        
        reward:7.95 - action:[-0.02608913 -0.03258336  0.00074041], observation:[ 2.8671153e-02  1.8143277e-01 -9.0536118e-02  0.0000000e+00
        0.0000000e+00  0.0000000e+00  1.0000000e+00 -2.6049018e-02
        -3.2533258e-02 -3.2603025e-01  0.0000000e+00  0.0000000e+00
        0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
        -1.4000648e-01 -8.8596946e-01  4.4210446e-01 -2.6089132e-02
        -3.2583356e-02  7.4040890e-04]
        """
        
        
        #p.applyExternalForce(
        #    objectUniqueId=self.drone_id,
        #    linkIndex=-1,
        #    forceObj=action,
        #    posObj=[0, 0, 0],
        #    flags=p.LINK_FRAME,
        #    physicsClientId=self.client_id,
            
        #)
        
    #####################################################################################################
    # Step
    #####################################################################################################                
          
    def step(self, aggregate_physics_steps: int = 1):
        for _ in range(aggregate_physics_steps):
            p.stepSimulation()

        observation = self.compute_observation()
        reward = self.compute_reward()
        terminated = self.compute_done()
        info = self.compute_info()

        return observation, reward, terminated, False, info
    
    def get_state(self) -> dict:
        position, orientation = p.getBasePositionAndOrientation(
                self.drone_id,
                physicsClientId=self.client_id,
            )
        
        linear_velocity, angular_velocity = p.getBaseVelocity(
            self.drone_id,
            physicsClientId=self.client_id,
        )
        
        target_position: np.ndarray = self.target_position
        direction = (target_position - position)/np.linalg.norm(np.array(target_position) - np.array(position))
        last_action = self.last_action
        
        state = {}
        state["position"] = np.array(position)
        state["orientation"] = np.array(orientation)
        state["linear_velocity"] = np.array(linear_velocity)
        state["angular_velocity"] = np.array(angular_velocity)    
        state["target_position"] = target_position
        state["direction"] = direction
        state["last_action"] = last_action
        
        return state
    
    def compute_observation(self) -> np.ndarray:
        state = self.get_state()
        position = self._normalize_position(state["position"])
        orientation = state["orientation"]
        linear_velocity = self._normalizeVelocity(state["linear_velocity"])
        angular_velocity = state["angular_velocity"]
        target_position = self._normalize_position(state["target_position"])
        direction = state["direction"]
        last_action = self._normalizeVelocity(state["last_action"])
        
        observation = np.concatenate((position, orientation, linear_velocity, angular_velocity, target_position, direction, last_action), dtype=np.float32)
        return observation

    def compute_reward(self) -> float:
        position = self.get_state()["position"]
        dome_radius = self.dome_radius
        
        bonus = 0
        
        if np.linalg.norm(position) < 0.2:
            bonus = 100_000
            return True
        
        return dome_radius -1 * float(np.linalg.norm(self.target_position - position)) + bonus
        
    def compute_done(self):
        position = self.get_state()["position"]
        dome_radius = self.dome_radius
        
        if np.linalg.norm(position) > dome_radius:
            #print("out of bounds", position, float(np.linalg.norm(self.target_position - position)))
            return True
        
        if np.linalg.norm(position) < 0.2:
            #print("reached target")
            return True
        
        if time.time() - self.RESET_TIME > self.MAX_TIME:
            #print("time out")
            return True
        
        return False
    
    def compute_info(self):
        return {}
    
    

    
    
    #####################################################################################################
    # Normalization
    #####################################################################################################

    def _normalizeVelocity(self, velocity: np.ndarray) -> np.ndarray:
        
        normalized_velocity = (np.clip(velocity, -self.MAX_VELOCITY, self.MAX_VELOCITY) /
                            np.full_like(velocity, self.MAX_VELOCITY))
        return normalized_velocity

    def _normalize_position(self, position: np.ndarray) -> np.ndarray:
        
        normalized = (np.clip(position, -self.dome_radius, self.dome_radius) /
                            np.full_like(position, fill_value=self.dome_radius)

                            )
        
        return normalized
    
    def get_keymap(self):
        keycode = KeyCode()
        default_action = [0.0, 0.0]

        key_map = defaultdict(lambda: default_action)
        key_map.update({
            Key.up: [0, .1],
            Key.down: [0, -.1],
            Key.left: [-.1, 0],
            Key.right: [.1, 0],
            #keycode.from_char("w"): [0, 0, 1.0, .1],
            #keycode.from_char("s"): [0, 0, -1.0, .1],
        })
        
        return key_map  