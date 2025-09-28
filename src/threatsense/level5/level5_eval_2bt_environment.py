import math
from typing import Dict

import numpy as np
from threatsense.level5.components.tasks_management.task_progression import TaskProgression
from threatsense.level5.components.tasks_management.tasks_dispatcher import TasksDispatcher
from threatsense.level5.level5_envrionment import Level5Environment
from gymnasium import Env, spaces


class Level52BTEvaluationEnvironment(Level5Environment):
    def __init__(self, GUI=True, rl_frequency=15):
        super().__init__(GUI=GUI, rl_frequency=rl_frequency, Teacher_Student=False)

    def init_task_progression(self):
        self.task_progression = TaskProgression(
            TasksDispatcher.level5_2bt_evaluation(
                self.dome_radius, self.entities_manager)
        )

        self.task_progression.on_env_init()
        self.task_progression.on_episode_start()

    
    def _observation_space(self):
        return spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)


    def compute_observation(self):
        return np.zeros(1).astype(np.float32)
    

    def _action_space(self):
        return spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32)



    def step(self, action):
        

        self.task_progression.on_step_start()

        self.advance_step()
        
        reward, terminated = self.task_progression.on_step_middle()

        info = self.task_progression.compute_info()

        self.task_progression.on_step_end()

        observation = {}
        reward = 0.0
        truncated = False
        
        return observation, reward, terminated, truncated, info
    

    def reset(self, seed=0):
        """Resets the environment.
        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """


        self.init_globals()
        self.task_progression.on_reset()
        self.reset_step_counter()

        info = self.task_progression.compute_info()


        return {}, info
