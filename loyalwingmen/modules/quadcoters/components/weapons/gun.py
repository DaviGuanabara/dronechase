from ....events.message_hub import MessageHub
from typing import Dict, Optional
import random
import numpy as np


class Gun:
    def __init__(
        self,
        parent_id: int,
        munition=10,
        cooldown_seconds=10,
        timestep=1 / 15,
        shoot_range=5,
        fire_probability=0.9,
    ):
        self.parent_id = parent_id
        self.timestep = timestep

        self.max_munition = munition
        self.munition = munition
        self.cooldown_steps = cooldown_seconds / self.timestep  # Cooldown in steps
        print(self.cooldown_steps, cooldown_seconds, self.timestep)
        self.available = True

        self.last_fired_step = -self.cooldown_steps
        self.current_step = 0

        self.messageHub = MessageHub()

        self.shoot_range = shoot_range
        self.fire_probability = fire_probability

        self.messageHub.subscribe(
            topic="SimulationStep", subscriber=self._subscriber_simulation_step
        )

    def _subscriber_simulation_step(self, message: Dict, publisher_id: int):
        self.current_step = message.get("step", 0)
        self.timestep = message.get("timestep", 0)
        self.available = self.is_available()

    def is_available(self):
        # print("self.cooldown_steps",self.cooldown_steps,"self.current_step",self.current_step,"self.last_fired_step",self.last_fired_step,"is available",self.cooldown_steps <= self.current_step - self.last_fired_step,)
        return self.cooldown_steps <= self.current_step - self.last_fired_step

    def has_munition(self):
        return self.munition > 0

    def can_fire(self) -> bool:
        return bool(self.is_available() and self.has_munition())

    def target_is_valid(self, target_acquired_step) -> bool:
        return target_acquired_step >= self.current_step

    def shoot(self) -> bool:
        if not self.can_fire():
            return False

        self.munition -= 1
        self.last_fired_step = self.current_step
        self.available = False

        if random.random() >= self.fire_probability:
            return False

        return True

    def get_state(self) -> np.ndarray:
        wait_time = max(
            self.cooldown_steps - (self.current_step - self.last_fired_step),
            0,
        )

        return np.array(
            [
                self.munition / self.max_munition,
                wait_time / self.cooldown_steps,
                int(self.available),
            ]
        )

    def get_state_shape(self):
        state = self.get_state()
        return state.shape

    def reset(self):
        self.munition = self.max_munition
        self.last_fired_step = -self.cooldown_steps
        self.current_step = 0
        self.available = True
