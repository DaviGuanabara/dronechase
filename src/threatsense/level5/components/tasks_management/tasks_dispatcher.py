# from .tasks import Task, L4Task1


from threatsense.level5.components.tasks_management.task_progression import Task
from threatsense.level5.components.tasks_management.tasks.level5_dumb_multiobject_task import Level5DumbMultiObjectTask
from threatsense.level5.components.tasks_management.tasks.level5_task import Level5_Task
from threatsense.level5.components.tasks_management.tasks.level5_occlusion_demo_task import Level5_Occlusion_Demo_Task
from threatsense.level5.components.tasks_management.tasks.level5_fusion_task import Level5FusionTask
from threatsense.level5.components.tasks_management.tasks.level5_c1_fusion_task import Level5C1FusionTask
class TasksDispatcher:

    @staticmethod
    def level5_tasks(dome_radius, quadcopter_manager) -> "list[Task]":
        print("[TasksDispatcher] Dispatching Level 5 tasks")

        return [Level5_Task(quadcopter_manager, dome_radius)]
    
    @staticmethod
    def level5_occlusion_demo(dome_radius, quadcopter_manager) -> "list[Task]":
        print("[TasksDispatcher] Dispatching Level 5 occlusion demo tasks")

        return [Level5_Occlusion_Demo_Task(quadcopter_manager, dome_radius)]
    

    @staticmethod
    def level5_fusion(dome_radius, quadcopter_manager) -> "list[Task]":
        print("[TasksDispatcher] Dispatching Level 5 fusion tasks")

        return [Level5FusionTask(quadcopter_manager, dome_radius)]
    
    @staticmethod
    def level5_c1_fusion(dome_radius, quadcopter_manager) -> "list[Task]":
        print("[TasksDispatcher] Dispatching Level 5 fusion tasks")

        return [Level5C1FusionTask(quadcopter_manager, dome_radius)]

    @staticmethod
    def level5_dumb_multi_object(dome_radius, quadcopter_manager) -> "list[Task]":
        print("[TasksDispatcher] Dispatching Level 5 dumb multi-object tasks")

        return [Level5DumbMultiObjectTask(quadcopter_manager, dome_radius)]