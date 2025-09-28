import math
from typing import Dict

import numpy as np
from threatsense.level5.components.tasks_management.task_progression import TaskProgression
from threatsense.level5.components.tasks_management.tasks_dispatcher import TasksDispatcher
from threatsense.level5.level5_envrionment import Level5Environment
from gymnasium import Env, spaces

class Level5DumbMultiObs(Level5Environment):
    def __init__(self, GUI=True, rl_frequency=15):
        super().__init__(GUI=GUI, rl_frequency=rl_frequency)

    def init_task_progression(self):
        self.task_progression = TaskProgression(
            TasksDispatcher.level5_dumb_multi_object(
                self.dome_radius, self.entities_manager)
        )

        self.task_progression.on_env_init()
        self.task_progression.on_episode_start()

    def _get_pursuers_agent_first(self):
        """Return the first pursuer (the RL agent) and the rest of the pursuers."""
        agent = self.entities_manager.get_agent()
        allies = self.entities_manager.get_allies()
        return [agent] + allies
    
    def _observation_space(self):
        return spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)


    def compute_observation(self):
        return np.zeros(1).astype(np.float32)
    

    def _compute_general_vision(self) -> np.ndarray:
        """Return the observation of the simulation.
        The RL AGENT PERSUER is the source of the observation.
        """

        for pursuer in self.entities_manager.get_all_pursuers():
            pursuer.update_lidar()

        # rl_agent.update_lidar()
        
        observations = []
        #pursuers = self._get_pursuers_agent_first()
        pursuers = self.entities_manager.get_armed_pursuers()

        for pursuer in pursuers:
            inertial_data: np.ndarray = self.process_inertial_state(pursuer)

            stacked_spheres = pursuer.lidar_data.get("stacked_spheres", None)
            validity_mask = pursuer.lidar_data.get("validity_mask", None)

            gun_state = pursuer.gun_state

            

            inertial_gun_concat = np.concatenate((inertial_data, gun_state), axis=0).astype(np.float32)

            # assert lidar is not None, "Lidar data is None"
            assert stacked_spheres is not None, "Stacked spheres data is None"
            assert validity_mask is not None, "Validity mask data is None"

            observations.append( {
                "stacked_spheres": stacked_spheres.astype(np.float32),
                # dummy compatível, #lidar.astype(np.float32),  # só por compatibilidade mesmo
                "lidar": np.zeros((2, 13, 26), dtype=np.float32),
                "validity_mask": validity_mask.astype(bool),
                # inertial_data.astype(np.float32),
                "inertial_data": inertial_gun_concat,
                "last_action": pursuer.last_action.astype(np.float32),
                # "gun": gun_state.astype(np.float32),
            })

        return np.array(observations)
    
    def _action_space(self):
        return spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32)


    def _compute_student_vision(self):
        observations: np.ndarray = self._compute_general_vision()
        new_set = []
        for observation in observations:
            obs = {
                "stacked_spheres": observation["stacked_spheres"],
                "validity_mask": observation["validity_mask"],
                "inertial_data": observation["inertial_data"],
                "last_action": observation["last_action"],
            }
            new_set.append(obs)
        return new_set

    def step(self, action):
        

        self.task_progression.on_step_start()

        self.advance_step()
        observation = self.compute_observation()
        reward, terminated = self.task_progression.on_step_middle()
         
        info = self.compute_info()

        self.task_progression.on_step_end()

        truncated = False
        
        return observation, reward, terminated, truncated, info

    def compute_info(self):

        pursuers = self.entities_manager.get_armed_pursuers()

        for pursuer in pursuers:
            pursuer.update_lidar()

        observations = []
        actions = []


        for pursuer in pursuers:

            stacked_spheres = pursuer.lidar_data.get("stacked_spheres", None)
            validity_mask = pursuer.lidar_data.get("validity_mask", None)

            inertial_data: np.ndarray = self.process_inertial_state(pursuer)
            gun_state = pursuer.gun_state
            inertial_gun_concat = np.concatenate((inertial_data, gun_state), axis=0)
            
            if stacked_spheres is None:
                raise ValueError("Stacked spheres data is None")
            if validity_mask is None:
                raise ValueError("Validity mask data is None")

            observations.append({
                "stacked_spheres": stacked_spheres.astype(np.float32),
                "validity_mask": validity_mask.astype(bool),
                "inertial_data": inertial_gun_concat.astype(np.float32),
                "last_action": pursuer.last_action.astype(np.float32),
            })

            actions.append(pursuer.last_action)

        return {"student_observations": observations, "teacher_actions": actions}
