from abc import ABC
import numpy as np

from typing import Dict


class Navigator(ABC):
    def __init__(
        self, default_state=None, building_position: np.ndarray = np.array([0, 0, 0])
    ):
        self.current_command: np.ndarray = np.array([0, 0, 0, 0])
        self.building_position = building_position
        self.state_registry: Dict = {}
        self.default_state = default_state

    def update(self, agent, offset_handler):
        raise NotImplementedError

    def fetch_command(self, agent, offset_handler):
        self.update(agent, offset_handler)
        return self.current_command

    def reset(self):
        self.state_registry = {}

    # ===============================================================================
    # State Registry
    # ===============================================================================

    def fetch_state(self, agent):
        if agent.id not in self.state_registry:
            self.state_registry[agent.id] = self.default_state

        return self.state_registry[agent.id]

    def register_state(self, agent, state):
        self.state_registry[agent.id] = state
