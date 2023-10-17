import numpy as np
from typing import NamedTuple
from dataclasses import dataclass, field
from gymnasium import spaces


@dataclass
class EnvironmentParameters:
    G: float
    NEIGHBOURHOOD_RADIUS: float
    simulation_frequency: int
    rl_frequency: int
    timestep: float
    aggregate_physics_steps: int
    client_id: int
    max_distance: float
    error: float
    debug: bool = False
    max_episode_time: float = 20
    GUI: bool = False
