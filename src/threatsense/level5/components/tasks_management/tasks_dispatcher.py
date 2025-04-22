# from .tasks import Task, L4Task1


from threatsense.level5.components.tasks_management.task_progression import Task
from threatsense.level5.components.tasks_management.tasks.level5_task import Level5_Task

class TasksDispatcher:
   

    @staticmethod
    def level5_tasks(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Level5_Task(quadcopter_manager, dome_radius)]

