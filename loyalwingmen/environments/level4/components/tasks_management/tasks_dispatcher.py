# from .tasks import Task, L4Task1
from .tasks.exp01_task import Task, Exp01_Task
from .tasks.exp02_task import Exp02_Task
from .tasks.exp02_v2_task import Exp02_V2_Task
from .tasks.exp021_task import Exp021_Task


class TasksDispatcher:
    @staticmethod
    def exp01(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp01_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp02(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp02_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp02_v2(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp02_V2_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp021(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp021_Task(quadcopter_manager, dome_radius)]
