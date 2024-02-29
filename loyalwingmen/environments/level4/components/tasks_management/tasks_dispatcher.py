# from .tasks import Task, L4Task1
from .tasks.exp01_task import Task, Exp01_Task
from .tasks.exp02_task import Exp02_Task
from .tasks.exp02_v2_task import Exp02_V2_Task
from .tasks.exp02_v3_task import Exp02_V3_Task
from .tasks.exp02_v2_full_task import Exp02_V2_Full_Task

from .tasks.exp02_v4_single_air_combat_only_task import (
    Exp02_V4_Single_air_combat_only_Task,
)

from .tasks.exp021_task import Exp021_Task
from .tasks.exp03_cooperative_only_task import Exp03_Cooperative_Only_Task
from .tasks.exp04_cooperative_only_coop_bt_stopped_task import (
    Exp04_Cooperative_Only_BT_Stopped_Task,
)

from .tasks.exp05_cooperative_only_2_RL_task import Exp05_Cooperative_Only_2_RL_Task
from .tasks.exp05_cooperative_only_2_RL_V2_task import (
    Exp05_Cooperative_Only_2_RL_V2_Task,
)

from .tasks.exp03_v2_cooperative_only_task import Exp03_V2_Cooperative_Only_Task

from .tasks.exp02_vFinal_task import Exp02_vFinal_Task
from .tasks.exp03_vFinal_task import Exp03_vFinal_Task
from .tasks.exp04_vFinal_task import Exp04_vFinal_Task
from .tasks.exp05_vFinal_task import Exp05_vFinal_Task

from .tasks.evaluation_task import Evaluation_Task


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
    def exp02_v2_full(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp02_V2_Full_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp021(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp021_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp02_v3(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp02_V3_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp02_v4(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp02_V4_Single_air_combat_only_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp03(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp03_Cooperative_Only_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp03_v2(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp03_V2_Cooperative_Only_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp04(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp04_Cooperative_Only_BT_Stopped_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp05(dome_radius, quadcopter_manager, model_path) -> "list[Task]":
        return [
            Exp05_Cooperative_Only_2_RL_Task(
                quadcopter_manager, dome_radius, model_path
            )
        ]

    @staticmethod
    def exp05_v2(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp05_Cooperative_Only_2_RL_V2_Task(quadcopter_manager, dome_radius)]

    # @staticmethod
    # def evaluation_exp03(dome_radius, quadcopter_manager) -> "list[Task]":
    #    return [Evaluation_Exp03_Cooperative_Only_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp02_vFinal(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp02_vFinal_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp03_vFinal(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp03_vFinal_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp04_vFinal(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp04_vFinal_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def exp05_vFinal(dome_radius, quadcopter_manager) -> "list[Task]":
        return [Exp05_vFinal_Task(quadcopter_manager, dome_radius)]

    @staticmethod
    def evaluation_vFinal(
        dome_radius, quadcopter_manager, configuration
    ) -> "list[Task]":
        return [Evaluation_Task(quadcopter_manager, dome_radius, configuration)]
