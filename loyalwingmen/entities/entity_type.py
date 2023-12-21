from enum import Enum, auto


class EntityType(Enum):
    LOITERINGMUNITION = auto()
    QUADCOPTER = auto()
    LOYALWINGMAN = auto()

    # OBSTACLE = auto()
    PROTECTED_BUILDING = auto()
    GROUND = auto()
