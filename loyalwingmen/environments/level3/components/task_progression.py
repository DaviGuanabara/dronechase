from typing import List, Tuple
from abc import ABC
from enum import Enum, auto

class StageStatus(Enum):
    
    FAILURE = auto()
    RUNNING = auto()
    SUCCESS = auto()
    
class Stage(ABC):
#===============================================================================
# On Calls
#===============================================================================
    def on_reset(self):
        pass

    def on_notification(self, message, publisher_id):
        pass

    def on_env_init(self):
        pass
        
    def on_episode_start(self):
        pass
    
    def on_episode_end(self):
        pass
        
    def on_step_start(self):
        """
        Here lies the methods that should be executed BEFORE the STEP.
        It aims to set the environment to the simulation step execution.
        """
        pass
        
    def on_step_middle(self):
        """
        Here lies the methods that should be executed BEFORE observation and reward.
        So, pay attention to the order of the methods. Do not change the position of the quadcopters
        before the observation and reward.
        """
        reward, termination = 0, False
        return reward, termination
        
    def on_step_end(self):
        
        pass
    
    @property
    def status(self):
        return StageStatus.FAILURE
    
class TaskProgression:
    """Curriculum Learning for the Level 3 Problem."""

    def __init__(self, stage_dispatch: List[Stage]):
        """Initialize the task progression manager."""
        self._current_stage_number = 0

        self.stage_dispatch = stage_dispatch
        self.update_current_stage()

    def on_notification(self, message, publisher_id):
        self.current_stage.on_notification(message, publisher_id)


    def update_current_stage(self):
        """Update the current stage based on the stage number."""
        self.current_stage = self.stage_dispatch[self._current_stage_number]
        
    def on_env_init(self):
        print("task progression - on env init")
        self.current_stage.on_env_init()

    def on_episode_start(self):
        """Called at the start of each episode."""
        self.current_stage.on_episode_start()

    def on_reset(self):
        """Reset the current stage."""
        self.current_stage.on_reset()

    def on_episode_end(self):
        """Called at the end of each episode."""
        self.current_stage.on_episode_end()

    def on_step_start(self):
        """Called at the start of each step."""
        self.current_stage.on_step_start()

    def on_step_middle(self) -> Tuple[float, bool]:
        """Execute actions before observation and reward."""
        reward, termination = self.current_stage.on_step_middle()
        
        return reward, termination

    def on_step_end(self):
        """Called at the end of each step."""
        self.current_stage.on_step_end()
        self.update_stage()

    def update_stage(self):
        """Advance to the next stage."""
        
        status = self.current_stage.status
        if status == StageStatus.SUCCESS:
            self.advance_stage()
            
        if status == StageStatus.FAILURE:
            self.reset_progression()
            
    def advance_stage(self):
        self._current_stage_number += 1
        if self._current_stage_number >= len(self.stage_dispatch):
            self._current_stage_number = 0
        self.update_current_stage()

    def reset_progression(self):
        """Reset progression to the beginning."""
        self._current_stage_number = 0
        self.update_current_stage()

    def get_current_stage_number(self) -> int:
        """Get the current stage number."""
        return self._current_stage_number



