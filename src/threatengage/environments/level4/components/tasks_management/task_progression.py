from typing import List, Tuple
from abc import ABC
from enum import Enum, auto


class TaskStatus(Enum):
    FAILURE = auto()
    RUNNING = auto()
    SUCCESS = auto()


class Task(ABC):
    # ===============================================================================
    # On Calls
    # ===============================================================================

    def __init__(self, simulation):
        pass

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

    def compute_info(self):
        return {}

    def update_model_path(self, model_params):
        pass

    def update_model(self, model):
        pass

    @property
    def status(self):
        return TaskStatus.FAILURE


class TaskProgression:
    """Curriculum Learning for the Level 3 Problem."""

    def __init__(self, tasks_dispatch: List[Task]):
        """Initialize the task progression manager."""
        self._current_stage_number = 0
        self.tasks_dispatch = tasks_dispatch
        self.update_current_stage()

    def compute_info(self):
        return self.current_stage.compute_info()

    def on_notification(self, message, publisher_id):
        self.current_stage.on_notification(message, publisher_id)

    def update_current_stage(self):
        """Update the current stage based on the stage number."""
        self.current_stage = self.tasks_dispatch[self._current_stage_number]

    def on_env_init(self):
        print("[ThreatEngage] Level 4 - task progression - on env init")
        print(f"Current stage: {self._current_stage_number}")
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
        if status == TaskStatus.SUCCESS:
            self.advance_stage()

        if status == TaskStatus.FAILURE:
            self.reset_progression()

    def advance_stage(self):
        self._current_stage_number += 1
        if self._current_stage_number >= len(self.tasks_dispatch):
            self._current_stage_number = 0
        self.update_current_stage()

    def reset_progression(self):
        """Reset progression to the beginning."""
        self._current_stage_number = 0
        self.update_current_stage()

    def get_current_stage_number(self) -> int:
        """Get the current stage number."""
        return self._current_stage_number

    def update_model_path(self, model_params):
        self.current_stage.update_model_path(model_params)

    def update_model(self, model):
        self.current_stage.update_model(model)
