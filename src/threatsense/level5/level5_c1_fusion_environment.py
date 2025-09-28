import numpy as np
from threatsense.level5.components.tasks_management.task_progression import TaskProgression
from threatsense.level5.components.tasks_management.tasks_dispatcher import TasksDispatcher
from threatsense.level5.level5_envrionment import Level5Environment
from gymnasium import Env, spaces

class Level5C1FusionEnvironment(Level5Environment):
    def __init__(self, GUI=True, rl_frequency=15):
        super().__init__(GUI=GUI, rl_frequency=rl_frequency)

    def init_task_progression(self):
        self.task_progression = TaskProgression(
            TasksDispatcher.level5_c1_fusion(
                self.dome_radius, self.entities_manager)
        )

        self.task_progression.on_env_init()
        self.task_progression.on_episode_start()

    def compute_observation(self):
        """Return the observation of the simulation.
        The RL AGENT PERSUER is the source of the observation.
        """

        for pursuer in self.entities_manager.get_armed_pursuers():
            pursuer.update_lidar()

        # rl_agent.update_lidar()

        observations = []
        # pursuers = self._get_pursuers_agent_first()
        agent = self.entities_manager.get_agent()

        inertial_data: np.ndarray = self.process_inertial_state(agent)

        stacked_spheres = agent.lidar_data.get("stacked_spheres", None)
        validity_mask = agent.lidar_data.get("validity_mask", None)

        gun_state = agent.gun_state

        inertial_gun_concat = np.concatenate(
                (inertial_data, gun_state), axis=0).astype(np.float32)

            # assert lidar is not None, "Lidar data is None"
        assert stacked_spheres is not None, "Stacked spheres data is None"
        assert validity_mask is not None, "Validity mask data is None"

        return {
                "stacked_spheres": stacked_spheres.astype(np.float32),
                # dummy compatível, #lidar.astype(np.float32),  # só por compatibilidade mesmo
 
                "validity_mask": validity_mask.astype(bool),
                # inertial_data.astype(np.float32),
                "inertial_data": inertial_gun_concat,
                "last_action": agent.last_action.astype(np.float32),
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
                "stacked_spheres": spaces.Box(
                    0,
                    1,
                    shape=observation_shape["stacked_spheres"],
                    dtype=np.float32,
                ),

                "validity_mask": spaces.MultiBinary(observation_shape["validity_mask"]),
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

            }
        )
    
    def compute_info(self):
        return {}