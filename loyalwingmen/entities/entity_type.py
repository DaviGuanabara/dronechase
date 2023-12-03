from enum import Enum, auto


class EntityType(Enum):
    QUADCOPTER = auto()
    LOYALWINGMAN = auto()
    LOITERINGMUNITION = auto()
    # OBSTACLE = auto()
    PROTECTED_BUILDING = auto()
    GROUND = auto()
