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
class DroneChaseEnv(Env):
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
        speed_amplification: float = 1,
        GUI: bool = False,
        debug: bool = False,
        
    ):
        #### client #############################################
        if GUI:
            client_id = self.setup_pybulley_GUI()
            #p.addUserDebugParameter("button",1,0,1)
            self.debug = debug
            

        else:
            client_id = self.setup_pybullet_DIRECT()
            self.debug = False

        #### Constants #############################################
        self.setup_Parameteres(simulation_frequency, rl_frequency, client_id, debug)

        #### Options ###############################################
        self.RESET_TIME = time.time()

        #### Factories #############################################
        self.setup_factories(speed_amplification)

        #### Housekeeping ##########################################
        self._housekeeping()

        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        loyalwingman: LoyalWingman = self.loyalwingmen[0]
        self.observation_space = loyalwingman.observation_space()

    def setup_factories(self, speed_amplification: float):
        env_p = self.environment_parameters
        self.lwingman_factory = LoyalWingmanFactory(env_p, speed_amplification)
        self.lmunition_factory = LoiteringMunitionFactory(env_p)

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

    def setup_Parameteres(self, simulation_frequency, rl_frequency, client_id, debug):
        self.environment_parameters = EnvironmentParameters(
            G=9.8,
            NEIGHBOURHOOD_RADIUS=np.inf,
            simulation_frequency=simulation_frequency,
            rl_frequency=rl_frequency,
            timestep_period=1 / simulation_frequency,
            aggregate_physics_steps=int(simulation_frequency / rl_frequency),
            client_id=client_id,
            max_distance=100,
            error=0.5,
            debug=debug
        )
        
    def set_frequency(self, simulation_frequency, rl_frequency):
        self.environment_parameters.simulation_frequency = simulation_frequency
        self.environment_parameters.rl_frequency = rl_frequency
        

    def manage_debug_text(self, text: str, position: list = [0, 0, 0], debug_text_id=None):
        if debug_text_id is not None:
            p.removeUserDebugItem(debug_text_id)

        return p.addUserDebugText(str(text), position)
            

    def get_parameteres(self):
        return self.environment_parameters

    def apply_target_behavior(self, obstacle):
        obstacle.apply_frozen_behavior()

    def reset(self, seed=1):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """
        p.resetSimulation(
            physicsClientId=self.environment_parameters.client_id)
        self.RESET_TIME = time.time()

        #### Housekeeping ##########################################

        self._housekeeping()
        observation = self._computeObs()
        
        self.last_distance: float = -1
        return observation, self._computeInfo()

    ################################################################################

    def step(self, rl_action: np.ndarray):
        """Advances the environment by one simulation step.
        Parameters
        ----------
        rl_action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.
            this action is a velocity vector that will be converted in unitary vector and intensity to be 
            used in the PID Controller.
            It was chosen to use a velocity vector because it is more intuitive to the user. Futhermore, uses no non-linear
            function as arctan, arccos like in unitary spherical direction and intensity, which brings more stability to the
            neural network learning.
            Finally, the use of cartesian direction and intensity can be a problem because it not respect
            the unitary constraint, meaning that the direction is not unitary.
            
        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current epoisode is over, check the specific implementation of `_computeDone()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.
        """

        #theta, phi, intensity = np.pi * \
        #    rl_action[0], np.pi * rl_action[1], 1 * rl_action[2]
        #radius = 1
        #spherical = np.array([radius, theta, phi])
        #cartesian = np.array(
        #    CoordinateConverter.spherical_to_cartesian(spherical))
        #velocity_action = np.append(cartesian, intensity)
        self.current_timestep += 1
        
        velocity_action: np.ndarray = np.array(rl_action)

        for _ in range(self.environment_parameters.aggregate_physics_steps):
            # multiple drones not ready. TODO: setup multiple drones
            for lw in self.loyalwingmen:
                loyalwingman: LoyalWingman = lw
                # velocity_action = rl_action

                self.last_action = velocity_action
                loyalwingman.apply_velocity_action(velocity_action)
                loyalwingman.update_kinematics()

            for lm in self.loitering_munitions:
                loitering_munition: LoiteringMunition = lm
                self.apply_target_behavior(loitering_munition)
                loitering_munition.update_kinematics()

            p.stepSimulation()

        observation = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeDone()
        info = self._computeInfo()

        self.observation = observation
        #print(observation)
        return observation, reward, terminated, False, info

    ################################################################################

    def close(self):
        """Terminates the environment."""
        # if self.RECORD and self.GUI:
        #    p.stopStateLogging(
        #        self.VIDEO_ID, physicsClientId=self.environment_parameters.client_id
        #    )
        p.disconnect(physicsClientId=self.environment_parameters.client_id)

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

    #def getDroneIds(self):
        """Return the Drone Ids.
        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.
        """
        #return self.DRONE_IDS

    ################################################################################

    def gen_random_position(self) -> np.ndarray:
        x = random.choice([-1, 1]) * random.random() * 2
        y = random.choice([-1, 1]) * random.random() * 2
        z = random.choice([-1, 1]) * random.random() * 2

        return np.array([x, y, z])

    def _housekeeping(self):
        """Housekeeping function.
        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.
        """

        #### Set PyBullet's parameters #############################
        
        p.setGravity(
            0,
            0,
            -self.environment_parameters.G,
            physicsClientId=self.environment_parameters.client_id,
        )
        p.setRealTimeSimulation(
            0, physicsClientId=self.environment_parameters.client_id
        )  # No Realtime Sync
        p.setTimeStep(
            self.environment_parameters.timestep_period,
            physicsClientId=self.environment_parameters.client_id,
        )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.environment_parameters.client_id,
        )

        #TODO: eu devo reposicionar o drone e o alvo a cada episódio, e não recriá-los.
        
        self.loyalwingmen = self.setup_loyalwingmen(1, np.array([0, 0, 3]))
        self.loitering_munitions = self.setup_loiteringmunition(1, np.array([0, 0, 0]))
        
        self.current_timestep = 0

    def setup_drones(self, factory: DroneFactory, position: np.ndarray, quantity: int = 1) -> List:
        drones: List = []

        for _ in range(quantity):
            #random_position = self.gen_random_position()
            factory.set_initial_position(position)
            drone = factory.create()
            drone.update_kinematics()

            drones.append(drone)                            

        return drones

    def setup_loyalwingmen(self, quantity: int = 1, positions: np.ndarray = np.array([0, 0, 0])) -> List[LoyalWingman]:
        drones: List[LoyalWingman] = self.setup_drones(self.lwingman_factory, positions, quantity)
        #drones.astype(LoyalWingman)
        return drones

    def setup_loiteringmunition(self, quantity: int = 1, positions: np.ndarray = np.array([0, 0, 3])) -> List[LoiteringMunition]:
        drones: List[LoiteringMunition] = self.setup_drones(self.lmunition_factory, positions, quantity)
        #drones.astype(LoiteringMunition)
        return drones

    ################################################################################

    def _actionSpace(self):
        """
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
        spherical = (radius, theta, phi)

        the action space is the possible values for unitary vector and intensity.
        To keep the unitary constraint, I choose spherical coordinates with radius equals 1.
        In this way, we got: action = (theta, phi, intensity)

        theta is polar angle and phi is azimuthal angle,
        in which theta is between 0 and pi (0 to 1) and phi is between -pi and +phi (-1 to 1).

        But its used as 0 to 1
        """

        """
        return spaces.Box(
            low=np.array([-1, -1, -1, 0]),
            high=np.array([1, 1, 1, 1]),
            shape=(4,),
            dtype=np.float32,
        )
        """
        
        
        #Direction in spherical coordinates and intensity
        """_return spaces.Box(
            low=np.array([0, -1, 0]),
            high=np.array([1, 1, 1]),
            shape=(3,),
            dtype=np.float32,
        )
        """
        
        #Velocity Vector that will be converted in direction and intensity
        return spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            shape=(3,),
            dtype=np.float32,
        )
        

    def _observationSpace(self):
        """Returns the observation space of the environment.
        Returns
        -------
        ndarray
            A Box() of shape (16,) depending on the observation type.
        """
        # a workaround to work with gymnasium

        return spaces.Box(
            low=np.array([-1, -1, 0, -1, -1, -1, -1, -1,
                         0, -1, -1, -1, -1, -1, -1, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32,
        )

    ################################################################################

    def _computeObs(self) -> np.ndarray:
        #lw: LoyalWingman = self.loyalwingmen[0]
        #lm: LoiteringMunition = self.loitering_munitions[0]
        
        lw: LoyalWingman = self.loyalwingmen[0]
        lm: LoiteringMunition = self.loitering_munitions[0]


        #TODO: kwargs problem.
        # TypeError: lidar_observation() got an unexpected keyword argument 'kwargs'
        return lw.observation(loyalwingmen=self.loyalwingmen, loitering_munitions=self.loitering_munitions, obstacles=[])
    
    import math

    def exponential_with_saturation(self, x, radius, plateau, growth_rate) -> float:
        """
        Exponential function with saturation that approaches zero when x is greater than radius,
        and approaches the plateau value as x approaches 0.

        Params
        -------
        x: float
            The input value.
        radius: float
            The radius at which the function should approach zero.
        plateau: float
            The plateau value that the function approaches as x approaches 0.
        growth_rate: float
            The rate of growth, controlling how fast the function approaches the plateau.

        Returns
        -------
        float
            The value of the exponential function at the given x.
        """

        if x > radius:
            return 0.0
        else:
            return plateau * (1 - math.exp(-growth_rate * (radius - x)))
        
    def rbf_function(self, x, max_value, min_value, sigma, radius, gamma):
        """
        Radial Basis Function (RBF) that satisfies the conditions y = plateau for x = 0,
        and y = 0 for x >= radius.

        Params
        -------
        x: float
            The input value.
        plateau: float
            The value of y when x = 0.
        sigma: float
            Parameter that controls the smoothness of the function.
        radius: float
            The value of x when y = 0.
        gamma: float
            Parameter that controls the smoothness of the transition to zero.

        Returns
        -------
        float
            The value of the RBF function at the given x.
        """
        return max_value * math.exp(-((x - min_value) / sigma)**2) * math.exp(-((x - radius) / gamma)**2)

    def inverted_sigmoid_decay_function(self, x, radius=5, steepness=1, min_value=0, max_value=100):
        """
        Inverted sigmoid decay function that satisfies the conditions min_value for x >= radius,
        and max_value for x <= 0.
        This function is differentiable.

        Params
        -------
        x: float
            The input value.
        max_value: float
            The value of y when x <= 0.
        min_value: float
            The value of y when x >= radius.
        radius: float
            The value of x when y reaches min_value. It is also the parameter used in LiDAR.
            It aims to get the effect of out the LiDAR range, there is no reward impact.
        steepness: float
            Parameter that controls the steepness of the sigmoid decay.

        Returns
        -------
        float
            The value of the inverted sigmoid decay function at the given x.
        """
        return min_value + (max_value - min_value) * math.exp(-steepness * (x - radius))
    
    @staticmethod
    def linear_decay_function(x, min_value=0, max_value=100):
        """
        Linear decay function that satisfies the conditions of outside LiDAR range, the 
        reward is 0, therefore: min_value for x >= radius.
        Max value happens when the target is hitted, therefore: max_value for x <= 0.
        
        It is differentiable when x >= 0 e x <= radius.
        """
        

        a = (-1 * (max_value - min_value)) 
        b = max_value
        
        return  a * x + b
        
      
    @staticmethod
    def exp_decay(current_distance) -> int:
        return int(1000 * math.exp(-current_distance))
 
    
    def _computeReward(self) -> float:
        
        
        lw: LoyalWingman = self.loyalwingmen[0]
        lm: LoiteringMunition = self.loitering_munitions[0]
        lw_position = lw.kinematics.position
        lw_velocity = lw.kinematics.velocity
        
        lm_position = lm.kinematics.position
        
        norm_distance: float = float(np.linalg.norm(lm_position - lw_position))
        velocity:float = float(np.linalg.norm(lw_velocity))
        
        penalty = norm_distance
        bonus = velocity
        
                
        if float(np.linalg.norm(lw_position)) > 200:
            penalty += 1_000_000
            
        if float(np.linalg.norm(lw_position)) < 0.2:
            bonus += 1_000_000    
        
        return bonus - penalty 
        
    
    def _computeDone(self):
        current = time.time()

        if current - self.RESET_TIME > 10: 
            return True
        
        return False    

    def _computeInfo(self):
        return {}

    #####################################################################################################
    # Normalization
    #####################################################################################################

    def _normalizeVelocity(self, velocity: np.ndarray):
        MAX_Velocity = 5
        normalized_velocity = (np.clip(velocity, -MAX_Velocity, MAX_Velocity) / MAX_Velocity) # type: ignore
        return normalized_velocity

    def _normalizePosition(self, position: np.ndarray):
        MAX_X_Y = 100
        MAX_Z = 100

        normalized_position_x_y = np.clip(
            position[0:2], -MAX_X_Y, MAX_X_Y) / MAX_X_Y # type: ignore
        normalized_position_z = np.clip([position[2]], 0, MAX_Z) / MAX_Z

        normalized_position = np.concatenate(
            (normalized_position_x_y, normalized_position_z)
        )

        return normalized_position

    def _normalizeDistance(self, distance):
        MAX_DISTANCE = 100
        normalized_distance = (
            np.clip(distance, -MAX_DISTANCE, MAX_DISTANCE) / MAX_DISTANCE
        )
        return normalized_distance

    

    def get_keymap(self):
        keycode = KeyCode()
        default_action = [0, 0, 0]  # Modify this to your actual default action

        key_map = defaultdict(lambda: default_action)
        key_map.update({
            Key.up: [0, 1, 0],
            Key.down: [0, -1, 0],
            Key.left: [-1, 0, 0],
            Key.right: [1, 0, 0],
            keycode.from_char("w"): [0, 0, 1],
            keycode.from_char("s"): [0, 0, -1],
        })
        
        return key_map

    #####################################################################################################
    # Log
    #####################################################################################################
    """
    def format_list(self, list_of_values):
        return str.join(" ", ["%0.2f".center(5) % i for i in list_of_values])

    def setup_demo_lidar_log(self):
        radius: float = 5
        resolution: float = 0.0045

        self.loyalwingmen[0].set_lidar_parameters(radius, resolution)

        return radius, resolution

    def generate_lidar_log(self):
        lw_kinematics = self.loyalwingmen[0].kinematics
        lm_kinematics = self.loitering_munitions[0].kinematics

        lw_position = lw_kinematics.position
        lm_position = lm_kinematics.position

        distance = np.linalg.norm(lm_position - lw_position)
        direction = (lm_position - lw_position) / distance

        # radius, resolution = self.setup_demo_lidar_log()
        obs = np.round(self.observation, 2)

        text = "The Demo Environment reduces the lidar resolution to be able to log it"
        text += "\n"
        text += f"LiDAR Max distance: {radius}, Resolution: {resolution}"
        text += "\n"
        text += "LoyalWingman position:"
        text += "({:.2f}, {:.2f}, {:.2f})".format(
            lw_position[0], lw_position[1], lw_position[2]
        )
        text += "\n"
        text += "LoiteringMunition position:"
        text += "({:.2f}, {:.2f}, {:.2f})".format(
            lm_position[0], lm_position[1], lm_position[2]
        )
        text += "\n"
        text += "direction: " + self.format_list(direction) + "\n"
        text += "distance: {:.2f}".format(distance) + "\n"
        text += "reward: {:.2f}".format(self.last_reward) + "\n"
        text += "action: " + self.format_list(self.last_action) + "\n"
        text += "Lidar View"
        text += "\n"
        text += "\n"
        text += np.array2string(obs)

        return text

    def show_lidar_log(self):
        # text = self.generate_lidar_log()
        # stdscr = curses.initscr()
        stdscr.clear()

        try:
            stdscr.addstr(0, 0, text)
            stdscr.refresh()
        except curses.error:
            pass

    def init_curses(self):
        if not self.curses_activated:
            stdscr = curses.initscr()
            self.curses_activated = True

        return stdscr
"""

