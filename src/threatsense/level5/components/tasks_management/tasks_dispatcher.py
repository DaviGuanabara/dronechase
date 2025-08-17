# from .tasks import Task, L4Task1


from threatsense.level5.components.tasks_management.task_progression import Task
from threatsense.level5.components.tasks_management.tasks.level5_task import Level5_Task
from threatsense.level5.components.tasks_management.tasks.level5_occlusion_demo_task import Level5_Occlusion_Demo_Task


class TasksDispatcher:

    @staticmethod
    def level5_tasks(dome_radius, quadcopter_manager) -> "list[Task]":
        print("[TasksDispatcher] Dispatching Level 5 tasks")

        return [Level5_Task(quadcopter_manager, dome_radius)]
    
    @staticmethod
    def level5_occlusion_demo(dome_radius, quadcopter_manager) -> "list[Task]":
        print("[TasksDispatcher] Dispatching Level 5 occlusion demo tasks")

        return [Level5_Occlusion_Demo_Task(quadcopter_manager, dome_radius)]
