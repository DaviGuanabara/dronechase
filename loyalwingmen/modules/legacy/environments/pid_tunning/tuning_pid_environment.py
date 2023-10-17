import numpy as np
from gymnasium import Env, spaces
from .tuning_pid_simulation import PIDTuningSimulation
from ..helpers.environment_parameters import EnvironmentParameters


class PIDTuningEnvironment(Env):
    def __init__(
        self,
        simulation_frequency: int = 240,
        rl_frequency: int = 30,
        dome_radius: float = 20,
        GUI: bool = False,
        debug: bool = False,
    ):
        self.setup_Parameteres(simulation_frequency, rl_frequency, GUI, debug)
        self.simulation = PIDTuningSimulation(dome_radius, self.environment_parameters)

        #### Create action and observation spaces ##################
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

    def _action_space(self):
        return spaces.Box(low=-10, high=10, shape=(18,), dtype=np.float32)

    def _observation_space(self):
        return spaces.Box(
            low=-1,
            high=1,
            shape=(self.simulation.observation_size(),),
            dtype=np.float32,
        )

    def setup_Parameteres(self, simulation_frequency, rl_frequency, GUI, debug):
        self.environment_parameters = EnvironmentParameters(
            G=9.8,
            NEIGHBOURHOOD_RADIUS=np.inf,
            simulation_frequency=simulation_frequency,
            rl_frequency=rl_frequency,
            timestep=1 / simulation_frequency,
            aggregate_physics_steps=int(simulation_frequency / rl_frequency),
            max_distance=100,
            error=0.5,
            client_id=-1,
            debug=debug,
            GUI=GUI,
        )

    def reset(self):
        return self.simulation.reset()

    def close(self):
        """Terminates the environment."""

        self.simulation.close()

    def step(self, action):
        return self.simulation.step(action)

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.
        Returns
        -------
        int:
            The PyBullet Client Id.
        """
        return self.environment_parameters.client_id
