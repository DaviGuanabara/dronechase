import math
from typing import Dict

import numpy as np
from threatsense.level5.components.tasks_management.task_progression import TaskProgression
from threatsense.level5.components.tasks_management.tasks_dispatcher import TasksDispatcher
from threatsense.level5.level5_envrionment import Level5Environment

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

    def compute_observation(self) -> np.ndarray:
        """Return the observation of the simulation.
        The RL AGENT PERSUER is the source of the observation.
        """

        for pursuer in self.entities_manager.get_all_pursuers():
            pursuer.update_lidar()

        # rl_agent.update_lidar()
        
        observations = []
        pursuers = self._get_pursuers_agent_first()

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

    def _compute_observation_teacher(self):
        observations: np.ndarray = self.compute_observation()
        for i in range(observations.shape[0]):
            del observations[i]["stacked_spheres"]
            del observations[i]["validity_mask"]
        return observations

    def _compute_observation_student(self):
        observations: np.ndarray = self.compute_observation()
        for i in range(observations.shape[0]):
            del observations[i]["lidar"]
        return observations
    
    def step(self):
        
        #TODO MUNDO RODA BT

        

        # TODO: Basically, i have to create a function in entities_manager that returns the RL Agent.
        # And it needs to be separeted from other entities. Maybe, i just need to create a quadcopter_type called RL Agent.
        # Or add another variable. I don't know yet.

        # MAKE A CONSTANT

        #rl_agent = self.entities_manager.get_agent()
        #rl_agent.drive(rl_action, self.show_name_on)

        self.task_progression.on_step_start()

        self.advance_step()

        reward, terminated = self.task_progression.on_step_middle()
        #info = self.task_progression.compute_info()
        

        observation = self.compute_observation()
        info = self.compute_info()
        truncated = False

        self.task_progression.on_step_end()

        #print(f"[DEBUG] Level5Environment step -> observation: {list(observation.keys())}")
        return observation, reward, terminated, truncated, info
    
    def compute_info(self):
        pursuers = self._get_pursuers_agent_first()
        actions = []
        actions.extend(pursuer.last_action for pursuer in pursuers)
        actions = np.array(actions)
        return {"student_observation": self._compute_observation_student(), "teacher_observation": self._compute_observation_teacher(), "actions": actions}
