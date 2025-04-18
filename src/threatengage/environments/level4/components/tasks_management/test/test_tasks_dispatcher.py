from .test_level4_kamikaze_fsm_task import (
    Task,
    TEST_LEVEL4_KAMIKAZE_FSM_TASK,
)

from .test_level4_wingman_bh_task import TEST_LEVEL4_WNGMAN_BH_TASK
from .test_lw_vs_lm_task import TEST_LW_VS_LM_TASK


class TESTTasksDispatcher:
    @staticmethod
    def test_fsm_level4_task(dome_radius, quadcopter_manager) -> "list[Task]":
        return [TEST_LEVEL4_KAMIKAZE_FSM_TASK(quadcopter_manager, dome_radius)]

    @staticmethod
    def test_bh_level4_task(dome_radius, quadcopter_manager) -> "list[Task]":
        return [TEST_LEVEL4_WNGMAN_BH_TASK(quadcopter_manager, dome_radius)]

    @staticmethod
    def test_lw_vs_lm_task(dome_radius, quadcopter_manager) -> "list[Task]":
        return [TEST_LW_VS_LM_TASK(quadcopter_manager, dome_radius)]
