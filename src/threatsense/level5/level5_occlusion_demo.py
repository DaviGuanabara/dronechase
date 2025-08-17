from threatsense.level5.components.tasks_management.task_progression import TaskProgression
from threatsense.level5.components.tasks_management.tasks_dispatcher import TasksDispatcher
from threatsense.level5.level5_envrionment import Level5Environment

class Level5OcclusionDemo(Level5Environment):
    def __init__(self, GUI=True, rl_frequency=15):
        super().__init__(GUI=GUI, rl_frequency=rl_frequency)

    def init_task_progression(self):
        self.task_progression = TaskProgression(
            TasksDispatcher.level5_occlusion_demo(
                self.dome_radius, self.entities_manager)
        )

        self.task_progression.on_env_init()
        self.task_progression.on_episode_start()
