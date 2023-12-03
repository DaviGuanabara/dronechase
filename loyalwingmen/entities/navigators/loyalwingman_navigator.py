from loyalwingmen.entities.quadcoters.quadcopter import (
    Quadcopter,
)
from loyalwingmen.environments.level4.components.entities_management.offsets_handler import (
    OffsetHandler,
)


from loyalwingmen.entities.navigators.navigator import Navigator
from enum import Enum, auto
from typing import Optional, Tuple, List
import random
import numpy as np

# ===============================================================================
# Behavior tree implementation
#
# SelectorNode (?)
# SequenceNode (->)
#                                                       [(?): root]
#                                                /                                                            \
#                                       	    /                                                              \
#                         [(?): EngageOrFormation]                                                          [(->): Sacrifice]
#                         /                                           \                                       /          \
#                        /                                             \                                     /            \
#                   [(->): EngageThreat]                           [(->) Formation]                    [OutOfMunition] [SacrificeAttack]
#                   /                 \                      /                              \
#                  /                   \                    /                                \
#   [ThreatInRangeAndGunAvailable]  [ChaseThreat]  [Inverse: ThreatInRangeAndGunAvailable] [MoveToFormation]

#
# ===============================================================================


class LoyalWingmanBehaviorTree(Navigator):
    def __init__(self, building_position: np.ndarray):
        self.root = SelectorNode("root")
        super().__init__(self.root, building_position)

        self.velocity = 0.6
        self._build_tree()
        # self.current_tree_node = None

    def set_current_tree_node(self, node):
        self.current_tree_node = node

    def _build_tree(self):
        # Path 1: Engage threat or maintain formation
        engage_or_formation = SelectorNode("EngageOrFormation")
        self.root.add_child(engage_or_formation)

        # Engage threat
        engage_threat_seq = SequenceNode("EngageThreat")
        engage_threat_seq.add_child(ThreatInRangeAndGunAvailable())
        engage_threat_seq.add_child(ChaseThreat())
        engage_or_formation.add_child(engage_threat_seq)

        # Maintain formation

        formation_seq = SequenceNode("Formation")

        inverse_node = InverseNode("inverse")
        inverse_node.add_child(ThreatInRangeAndGunAvailable())
        formation_seq.add_child(inverse_node)

        inverse_out_of_munition = InverseNode("has_munition")
        inverse_out_of_munition.add_child(OutOfMunition())

        formation_seq.add_child(inverse_out_of_munition)

        formation_seq.add_child(MoveToFormation())

        engage_or_formation.add_child(formation_seq)

        # Path 2: Sacrifice attack
        sacrifice_seq = SequenceNode("Sacrifice")
        sacrifice_seq.add_child(OutOfMunition())
        sacrifice_seq.add_child(SacrificeAttack())

        self.root.add_child(sacrifice_seq)

    def update(
        self,
        agent: Quadcopter,
        offset_handler: OffsetHandler,
    ):
        # It will always be the root node.
        state: TreeNode = self.fetch_state(agent)
        state.execute(agent, self, offset_handler)

        # print(f"Root:{state.name} Current tree node:", self.current_tree_node.name)

        # self.root.execute(agent, self, offset_handler)


# ===============================================================================
# Behavior Tree Elements
#
# TreeNode: Base class for all behavior tree nodes
# SequenceNode: Tree node that executes children sequentially until one fails
# SelectorNode: Tree node that executes children until one succeeds
# ===============================================================================


class ExecutionStatus(Enum):
    """
    Represents the execution status of a behavior tree node.
    """

    SUCCESS = 0
    FAILURE = 1
    RUNNING = 2


# Base class for all behavior tree nodes
class TreeNode:
    def __init__(self, name):
        self.name = name

    def enter(
        self,
        agent: Quadcopter,
        navigator: LoyalWingmanBehaviorTree,
        offset_handler: OffsetHandler,
    ):
        pass

    def execute(
        self,
        agent: Quadcopter,
        navigator: LoyalWingmanBehaviorTree,
        offset_handler: OffsetHandler,
    ):
        raise NotImplementedError("Execute method must be implemented by child classes")


# tree node that executes children sequentially until one fails
class SequenceNode(TreeNode):
    def __init__(self, name):
        super().__init__(name)
        self.children = []
        self.current_child = None

    def execute(
        self,
        agent: Quadcopter,
        navigator: LoyalWingmanBehaviorTree,
        offset_handler: OffsetHandler,
    ):
        # print(f"SequenceNode: {self.name}")
        # if not self.current_child:
        #    self.current_child = iter(self.children)

        for running_child in self.children:
            status = running_child.execute(agent, navigator, offset_handler)
            if status != ExecutionStatus.SUCCESS:
                return status
        return ExecutionStatus.SUCCESS

    def add_child(self, child):
        self.children.append(child)


class InverseNode(TreeNode):
    def __init__(self, name):
        super().__init__(name)
        self.children = []
        self.current_child = None

    def execute(
        self,
        agent: Quadcopter,
        navigator: LoyalWingmanBehaviorTree,
        offset_handler: OffsetHandler,
    ):
        # print(f"InverseNode: {self.name}")
        status = self.children[0].execute(agent, navigator, offset_handler)
        if status == ExecutionStatus.SUCCESS:
            return ExecutionStatus.FAILURE

        if status == ExecutionStatus.FAILURE:
            return ExecutionStatus.SUCCESS

        if status == ExecutionStatus.RUNNING:
            return ExecutionStatus.RUNNING

    def add_child(self, child):
        self.children.append(child)


# tree node that executes children until one succeeds
class SelectorNode(TreeNode):
    def __init__(self, name):
        super().__init__(name)
        self.children = []
        self.current_child = None

    def execute(
        self,
        agent: Quadcopter,
        navigator: LoyalWingmanBehaviorTree,
        offset_handler: OffsetHandler,
    ):
        # print(f"SelectorNode: {self.name}")
        # print("begin")
        # print(self.children)
        # if not self.current_child:
        #    self.current_child = iter(self.children)

        # running_child = self.children[0]
        # print(f"for child {self.current_child}")

        for running_child in self.children:
            # print(f"running_child: {running_child.name}")
            status = running_child.execute(agent, navigator, offset_handler)
            if status != ExecutionStatus.FAILURE:
                return status

        # print("end - Failure")
        return ExecutionStatus.FAILURE

    def add_child(self, child):
        self.children.append(child)


# ===============================================================================
# Leafs of the Behavior Tree
#
# 1. Conditions
#    ThreatInRangeAndGunAvailable: Check if threat is in range and gun is available
#    OutOfMunition: Check if out of munition
#
# 2. Behaviors
#    ChaseThreat: Chase the threat
#    MoveToFormation: Move to formation
#    SacrificeAttack: Sacrifice attack
#
# ===============================================================================


class ThreatInRangeAndGunAvailable(TreeNode):
    def __init__(self):
        super().__init__("ThreatInRangeAndGunAvailable")

    def execute(
        self,
        agent: Quadcopter,
        navigator: LoyalWingmanBehaviorTree,
        offset_handler: OffsetHandler,
    ):
        # Implement check for threat in range and gun availability
        # TODO: Range of threat
        # I am considering that all the threats is in range.
        # print("ThreatInRangeAndGunAvailable")
        if agent.is_gun_available:
            return ExecutionStatus.SUCCESS

        else:
            return ExecutionStatus.FAILURE


class OutOfMunition(TreeNode):
    def __init__(self):
        super().__init__("OutOfMunition")

    def execute(
        self,
        agent: Quadcopter,
        navigator: LoyalWingmanBehaviorTree,
        offset_handler: OffsetHandler,
    ):
        # print("OutOfMunition")
        # Implement check for munition status
        if agent.is_munition_available:
            return ExecutionStatus.FAILURE

        return ExecutionStatus.SUCCESS


class ChaseThreat(TreeNode):
    def __init__(self):
        super().__init__("ChaseThreat")

    def execute(
        self,
        agent: Quadcopter,
        navigator: LoyalWingmanBehaviorTree,
        offset_handler: OffsetHandler,
    ):
        # Implement chasing logic
        # print("ChaseThreat")

        navigator.set_current_tree_node(self)

        closest_invader_id = offset_handler.identify_closest_invader(agent.id)
        target_position = offset_handler.get_invader_position(closest_invader_id)

        vector = target_position - agent.inertial_data["position"]
        direction = (
            vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
        )

        # print(f"ChaseThreat: {direction}")
        agent.drive(np.array([*direction, navigator.velocity]))
        return ExecutionStatus.SUCCESS


class MoveToFormation(TreeNode):
    def __init__(self):
        super().__init__("MoveToFormation")

    def execute(
        self,
        agent: Quadcopter,
        navigator: LoyalWingmanBehaviorTree,
        offset_handler: OffsetHandler,
    ):
        # Implement formation logic
        # print("MoveToFormation")
        navigator.set_current_tree_node(self)

        target_position = agent.formation_position

        vector = target_position - agent.inertial_data["position"]
        direction = (
            vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
        )
        agent.drive(np.array([*direction, navigator.velocity]))
        return ExecutionStatus.SUCCESS


class SacrificeAttack(TreeNode):
    def __init__(self):
        super().__init__("SacrificeAttack")

    def execute(
        self,
        agent: Quadcopter,
        navigator: LoyalWingmanBehaviorTree,
        offset_handler: OffsetHandler,
    ):
        # Implement sacrifice attack logic
        # print("SacrificeAttack")
        navigator.set_current_tree_node(self)

        closest_invader_id = offset_handler.identify_closest_invader(agent.id)
        target_position = offset_handler.get_invader_position(closest_invader_id)

        vector = target_position - agent.inertial_data["position"]
        direction = (
            vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
        )
        agent.drive(np.array([*direction, navigator.velocity]))

        return ExecutionStatus.SUCCESS
