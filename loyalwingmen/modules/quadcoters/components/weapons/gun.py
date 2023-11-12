from ....events.message_hub import MessageHub
from typing import Dict, Optional
import random
import numpy as np

class Gun():
    
    
    def __init__(self, parent_id: int, munition = 20, cooldown = 3, timestep = 1/15, target_range = 5, fire_probability = 0.9):
        
        self.parent_id = parent_id

        self.max_munition = munition
        self.munition = munition
        self.cooldown = cooldown
        self.available = True

        self.timestep = timestep
        self.last_fired_step = 0
        self.current_step = 0

        self.messageHub = MessageHub()
        
        self.target_range = target_range
        self.fire_probability = fire_probability
        
        self.messageHub.subscribe(
            topic="SimulationStep", subscriber=self._subscriber_simulation_step
        )
    
    def _subscriber_simulation_step(self, message: Dict, publisher_id: int):
        self.current_step = message.get("step", 0)
        self.timestep = message.get("timestep", 0)
        self.available = self.is_available()
        
        
    def is_available(self):
        return self.cooldown <= self.current_step * self.timestep - self.last_fired_step * self.timestep
    
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
        
        if random.random() >= self.fire_probability:
            return False
        
        return True

    
    def get_state(self) -> np.ndarray:
        wait_time = max(self.cooldown - (self.current_step * self.timestep - self.last_fired_step * self.timestep), 0) 
        
        
        return np.array([self.munition / self.max_munition, wait_time / self.cooldown, int(self.available)])
    
    def get_state_shape(self):
        state = self.get_state()
        return state.shape