# from __future__ import annotations
from enum import Enum, auto
from core.entities.navigators.navigator import Navigator

# from ..offsets_handler import OffsetHandler
# from loyalwingmen.environments.level4.components.entities_management.offsets_handler import (
#    OffsetHandler,
#    Quadcopter,
# )

from core.entities.navigators.geometry_utils import GeometryUtils
from typing import Optional, Tuple, List, Dict
import numpy as np


# ===============================================================================
# Loitering Munition Navigator
#
# Finite State Machine for Loitering Munition Navigator
# -----------------------------------------------------
#
# State Behavior:
# - WaitState: No movement, waiting for condition to change state.
# - CollideWithWingmanState: Navigate towards the closest wingman.
# - CollideWithBuildingState: Navigate directly towards the building.
#
#      (Building path clear)
#   | <-------------------       WaitState
#   |                                | (Building path not clear)
#   |  (are wingmen alive)           v
#   | <-------------------  CollideWithWingmanState
#   |                                ^
#   |                                | (Building path not clear)
#   | ------------------->  CollideWithBuildingState
#
#   Hysteresis Switch Concept:
#   - Used to prevent rapid state toggling (e.g., in CollideWithBuildingState).
#   - Transitions use different thresholds for entering and exiting states.
#
# ===============================================================================


# TODO: esse navigator só está funcionando para 1 kamikaze. Eu tenho que fazer ele funcionar para vários kamikazes.
# Então, pode-se utilizar duas estraégias:
# 1) Criar um navigator para cada kamikaze, assim o estado ficará armazenado até chamar o reset.
# 2) Armazenar os estados em um dicionário, onde a chave é o id do kamikaze e o valor é o estado, dentro do Navigator.
# Assim, o execute terá que ter uma função a mais, que é a de identificar o kamikaze que está sendo executado (talvez com o nome setup).
# O Dois será bem mais fácil de implementar. Assim, vou com ele.
class KamikazeNavigator(Navigator):
    """
    A finite state machine for Loitering Munition.
    """

    def __init__(self, building_position: np.ndarray):
        super().__init__(WaitState(), building_position)

        self.action: np.ndarray = np.array([0, 0, 0, 0])
        self.velocity = 0.4

    # ===============================================================================
    # State Machine
    # ===============================================================================

    def change_state(self, agent, new_state):
        self.register_state(agent, new_state)
        # self.state = new_state

    def update(self, agent, offset_handler):
        state: State = self.fetch_state(agent)
        state.check_transition(agent, self, offset_handler)
        state.execute(agent, self, offset_handler)

    # ===============================================================================
    # Helpers
    # ===============================================================================
    def _is_building_path_clear(self, offset_handler, agent, degrees: int = 60) -> bool:
        invader_position = agent.inertial_data["position"]
        pursuers_positions = offset_handler.get_purusers_positions()

        return next(
            (
                False
                for point in pursuers_positions
                if GeometryUtils.is_point_inside_cone(
                    point, invader_position, self.building_position, degrees
                )
            ),
            False,
        )

    def _are_wingmen_alive(self, offset_handler) -> bool:
        return len(offset_handler.get_purusers_positions()) > 0

    def _get_closest_pursuers_position(self, offset_handler, invader_id):
        closest_pursuer_id = offset_handler.identify_closest_pursuer(invader_id)
        if closest_pursuer_id == -1:
            return np.array([0, 0, 0])

        return offset_handler.get_pursuer_position(closest_pursuer_id)


class State:
    """
    Abstract state class for LoiteringMunitionNavigator.
    """

    def __init__(self, name):
        self.name = name

    def check_transition(
        self,
        agent,
        navigator: KamikazeNavigator,
        offset_handler,
    ):
        raise NotImplementedError(
            "This method is abstract and must be implemented in derived classes"
        )

    def execute(
        self,
        agent,
        navigator: KamikazeNavigator,
        offset_handler,
    ):
        raise NotImplementedError(
            "This method is abstract and must be implemented in derived classes"
        )


class WaitState(State):
    def __init__(self):
        super().__init__("WaitState")

    def check_transition(
        self,
        agent,
        navigator: KamikazeNavigator,
        offset_handler,
    ):
        if navigator._is_building_path_clear(offset_handler, agent, degrees=60):
            navigator.change_state(agent, CollideWithBuildingState())
        elif navigator._are_wingmen_alive(offset_handler):
            if closest_pursuer := offset_handler.identify_closest_pursuer(
                agent.id
            ):
                navigator.change_state(agent, CollideWithWingmanState())

    def execute(
        self,
        agent,
        navigator: KamikazeNavigator,
        offset_handler,
    ):
        # No movement
        agent.drive(np.array([0, 0, 0, 0.4]))


class CollideWithWingmanState(State):
    def __init__(self):
        super().__init__("CollideWithWingman")

    def check_transition(
        self,
        agent,
        navigator: KamikazeNavigator,
        offset_handler,
    ):
        if not navigator._are_wingmen_alive(offset_handler):
            navigator.change_state(agent, CollideWithBuildingState())

    def execute(
        self,
        agent,
        navigator: KamikazeNavigator,
        offset_handler,
    ):
        target_position = navigator._get_closest_pursuers_position(
            offset_handler, agent.id
        )

        invader_position = agent.inertial_data["position"]
        vector = target_position - invader_position
        direction = (
            vector / np.linalg.norm(vector)
            if float(np.linalg.norm(vector)) > 0
            else vector
        )

        agent.drive(np.array([*direction, navigator.velocity]))


class CollideWithBuildingState(State):

    """
    Collide with building state.
    It will be triggered when the path to the building is clear.
    It can be changed if the path is not clear anymore.

    To avoid the hysteresis switch problem, it will be defined as 60 degrees cone to enter the Collide with building state, and
    45 degrees cone to exit the Collide with building state.

    The hysteresis switch problem in a badly designed Finite State Machine (FSM) refers
    to an issue where the FSM experiences undesirable oscillations between states due to small fluctuations in input.
    This problem is particularly evident in systems that are sensitive to input changes and where the thresholds
    for state transitions are not well-defined or are too close to each other.
    """

    def __init__(self):
        super().__init__("CollideWithBuilding")

    def check_transition(
        self,
        agent,
        navigator: KamikazeNavigator,
        offset_handler,
    ):
        if not navigator._is_building_path_clear(offset_handler, agent, degrees=45):
            navigator.change_state(agent, CollideWithWingmanState())

    def execute(
        self,
        agent,
        navigator: KamikazeNavigator,
        offset_handler,
    ):
        invader_position: np.ndarray = agent.inertial_data["position"]
        target_position: np.ndarray = navigator.building_position

        vector = target_position - invader_position
        direction = (
            vector / np.linalg.norm(vector)
            if float(np.linalg.norm(vector)) > 0
            else vector
        )

        # print("CollideWithBuilding Execute - direction", direction)
        # print("invader_position", invader_position, "target_position", target_position)
        agent.drive(np.array([*direction, navigator.velocity]))
