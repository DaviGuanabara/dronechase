from threatsense.level5.components.normalization import normalize_inertial_data
from threatsense.level5.components.tasks_management.tasks_dispatcher import TasksDispatcher
from threatsense.level5.components.tasks_management.task_progression import TaskProgression
from threatsense.level5.components.level5_simulation import L5AviarySimulation
from threatsense.level5.components.entities_manager import EntitiesManager
from core.entities.quadcopters.quadcopter import Quadcopter
from core.notification_system.topics_enum import Topics_Enum
from core.notification_system.message_hub import MessageHub
from collections import defaultdict
from typing import Dict

import numpy as np
from gymnasium import Env, spaces
try:
    from pynput import keyboard as pynput_keyboard  # type: ignore
    PYNPUT_AVAILABLE = True
except ImportError:
    pynput_keyboard = None  # ✅ define mesmo se der erro
    PYNPUT_AVAILABLE = False


# from .components.tasks_management.stages import L3Stage1 as Stage1

print("importing core on Level 5 Environment")

print("importing level 5 components")

print("importing level 5 components - done")


class Level5Environment(Env):
    """
    This class aims to demonstrate a environment with one Loyal Wingmen (CONTROLLED BY AN THE RL AGENT.
    THIS CODE IS PREPARED TO RECEIVE MULTIPLE LOYALWINGMEN, WITH THE FIRST BEING CONTROLLED BY RL AGENT, AND THE FOLLOWING BEING CONTROLLED BY
    BEHAVIOR TREE). and an linearly increasing number of Loitering Munition, starting by one. It is diveded by waves.
    Each wave has a number of Loitering Munition equal to the wave number.
    The Loyal Wingmen must to destroy all Loitering Munition before the next wave starts. Loyalwingman has a limited number of bullets, 20,
    limiting the number of waves up to 6 (6 waves means 21 loitering munitions engaged).
    It was developed for Stable Baselines 3 2.0.0 or higher.
    Gym Pybullet Drones was an inspiration for this project. For more: https://github.com/utiasDSL/gym-pybullet-drones

    Finally, I tried to do my best inside of my time constraint. Then, sorry for messy code.
    """

    def __init__(
        self,
        dome_radius: float = 20,
        rl_frequency: int = 15,
        use_fused_lidar: bool = False,
        GUI: bool = False,
    ):
        """Initialize the environment."""
        print("Init Level 5 Environment Init ThreatSense level 5 environment")
        print("ThreatSense level 5 environment init")

        print("Level 5 Environment - init constants")
        self.init_constants(dome_radius, rl_frequency, use_fused_lidar, GUI)
        self.init_components(dome_radius, GUI)
        self.frequency_adjustments(rl_frequency)

        self.init_globals()

        self.setup_static_entities()

        # self.step_counter = 0
        self.task_progression.on_env_init()
        self.task_progression.on_episode_start()
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

    def setup_static_entities(self):
        entities_manager = self.entities_manager
        entities_manager.spawn_ground(self.dome_radius)
        entities_manager.spawn_protected_building()

    def setup_messange_hub(self):
        self.message_hub = MessageHub()
        self.message_hub.subscribe(
            topic="Simulation", subscriber=self._subscriber_notifications
        )

    def _subscriber_notifications(self, message, publisher_id):
        if hasattr(message, "notification_type"):
            self.task_progression.on_notification(message, publisher_id)

    #### Initialization ######################################

    def init_constants(self, dome_radius: float, rl_frequency: int, use_fused_lidar: bool, GUI: bool):
        self.dome_radius = dome_radius
        self.debug_on = GUI
        self.show_name_on = GUI
        self.max_step_calls = 20 * rl_frequency
        self.use_fused_lidar = use_fused_lidar

    def init_globals(self):
        self.last_action = np.zeros(4)
        self.step_counter = 0

    def init_components(self, dome_radius, GUI):

        print("Level 5 Environment - init components")
        print("initializing L5AviarySimulation")
        self.simulation = L5AviarySimulation(
            world_scale=dome_radius, render=GUI)
        self.entities_manager = EntitiesManager()
        self.entities_manager.setup_simulation(self.simulation)
        self.entities_manager.setup_debug(self.debug_on)

        print("task progression init - Level 5 Environment")
        self.task_progression = TaskProgression(
            TasksDispatcher.level5_tasks(
                self.dome_radius, self.entities_manager, use_fused_lidar=self.use_fused_lidar)
        )

        self.setup_messange_hub()

        # plane_id = self.simulation.loadURDF("plane.urdf", basePosition=[0, 0, 0])
        # self.simulation.changeDynamics(plane_id, -1, mass=0)

    def frequency_adjustments(self, rl_frequency):
        self.rl_frequency = rl_frequency
        ctrl_hz = self.simulation.ctrl_hz
        self.aggregate_sim_steps = int(ctrl_hz / rl_frequency)

    def manage_debug_text(self, text: str, debug_text_id=None):
        return self.simulation.addUserDebugText(
            text,
            position=np.zeros(3),
            replaceItemUniqueId=debug_text_id,
            textColorRGB=[1, 1, 1],
            textSize=1,
            lifeTime=0,
        )

    #### Close, Reset and Step ######################################
    def close(self):
        """Terminates the environment."""

        self.simulation.close()

    def reset(self, seed=0):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """

        # I want to avoid the overhead of reseting the simulation.
        # self.simulation.reset()

        self.init_globals()
        self.task_progression.on_reset()
        self.step_counter = 0

        observation = self.compute_observation()
        info = self.compute_info()
        return observation, info

    # TODO: I have to work on this method.
    # the action should go only to the RL Agent. And the other entities should be controlled by the task class.
    def step(self, rl_action: np.ndarray = np.array([0, 0, 0, 0])):
        self.last_action: np.ndarray = rl_action

        # TODO: Basically, i have to create a function in entities_manager that returns the RL Agent.
        # And it needs to be separeted from other entities. Maybe, i just need to create a quadcopter_type called RL Agent.
        # Or add another variable. I don't know yet.

        # MAKE A CONSTANT

        rl_agent = self.entities_manager.get_agent()
        rl_agent.drive(rl_action, self.show_name_on)

        self.task_progression.on_step_start()

        self.advance_step()

        reward, terminated = self.task_progression.on_step_middle()
        info = self.task_progression.compute_info()

        observation = self.compute_observation()
        truncated = False
        # info = self.compute_info()

        self.task_progression.on_step_end()
        return observation, reward, terminated, truncated, info

    def advance_step(self):
        for _ in range(self.aggregate_sim_steps):
            self.simulation.step()

        self.step_counter += 1
        self.message_hub.publish(
            Topics_Enum.AGENT_STEP_BROADCAST.value,
            {"step": self.step_counter, "timestep": 1 / self.rl_frequency},
            0,
        )

    # ====================================================================================================
    # observation, truncated, info
    # ====================================================================================================

    def compute_info(self):
        return {}

    def _action_space(self):
        # direction and intensity fo velocity
        return spaces.Box(
            low=np.array([-1, -1, -1, 0]),
            high=np.array([1, 1, 1, 1]),
            shape=(4,),
            dtype=np.float32,
        )

    def compute_observation(self) -> Dict:
        """Return the observation of the simulation.
        The RL AGENT PERSUER is the source of the observation.
        """

        rl_agent: Quadcopter = self.entities_manager.get_agent()

        rl_agent.update_lidar()

        inertial_data: np.ndarray = self.process_inertial_state(rl_agent)
        rl_agent.lidar_data

        lidar: np.ndarray = rl_agent.lidar_data.get(
            "lidar",
            np.zeros(
                rl_agent.lidar_shape,
                dtype=np.float32,
            ),
        )

        gun_state = rl_agent.gun_state

        inertial_gun_concat = np.concatenate((inertial_data, gun_state), axis=0).astype(
            np.float32
        )

        return {
            "lidar": lidar.astype(np.float32),
            # inertial_data.astype(np.float32),
            "inertial_data": inertial_gun_concat,
            "last_action": self.last_action.astype(np.float32),
            # "gun": gun_state.astype(np.float32),
        }

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
        observation_shape = self.observation_shape()
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
                # "gun": spaces.Box(
                #    -1,
                #    1,
                #    shape=observation_shape["gun"],
                #    dtype=np.float32,
                # ),
            }
        )

    def observation_shape(self) -> dict:
        """
        Shape from RL AGENT PERSUER.
        """

        rl_agent = self.entities_manager.get_agent()
        position = 3
        velocity = 3
        attitude = 3
        angular_rate = 3
        gun_state = 3

        inertial_data = position + velocity + attitude + angular_rate + gun_state

        lidar_shape = rl_agent.lidar_shape
        last_action_shape = 4

        gun_state_shape = rl_agent.gun_state_shape

        return {
            "lidar": lidar_shape,
            "inertial_data": inertial_data,
            "last_action": last_action_shape,
            # "gun": gun_state_shape,
        }

    ## Helpers #####################################################################

    def process_inertial_state(self, quadcopter: Quadcopter) -> np.ndarray:
        inertial_data = quadcopter.inertial_data
        max_speed = quadcopter.max_speed
        dome_radius = self.dome_radius
        normalized_inertial_data = normalize_inertial_data(
            inertial_data, max_speed, dome_radius
        )

        return self.convert_dict_to_array(normalized_inertial_data)

    def convert_dict_to_array(self, dictionary: Dict) -> np.ndarray:
        array = np.array([])
        for key in dictionary:
            value: np.ndarray = (
                dictionary[key] if dictionary[key] is not None else np.array([
                ])
            )
            array = np.concatenate((array, value))

        return array

    ################################################################################

    def get_keymap(self):
        key_map = defaultdict(lambda: [0, 0, 0, 1])  # default action

        if not PYNPUT_AVAILABLE or pynput_keyboard is None:
            print("⚠️ pynput indisponível. Algumas funcionalidades estão desativadas.")
            return key_map

        Key = pynput_keyboard.Key
        KeyCode = pynput_keyboard.KeyCode

        key_map |= {
            Key.up: [0, 1, 0, 1],
            Key.down: [0, -1, 0, 1],
            Key.left: [-1, 0, 0, 1],
            Key.right: [1, 0, 0, 1],
            KeyCode.from_char("e"): [0, 0, 1, 1],
            KeyCode.from_char("d"): [0, 0, -1, 1],
        }

        return key_map
