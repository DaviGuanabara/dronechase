from enum import Enum, auto


class EntityType(Enum):
    #TODO: DO NOT CHANGE OR ADD ANYTHING HERE.
    #ITS SIZE IS USED IN LIDAR.
    #auto() begins in 1
    LOITERINGMUNITION = auto()
    QUADCOPTER = auto()
    LOYALWINGMAN = auto()

    # OBSTACLE = auto()
    PROTECTED_BUILDING = auto()
    GROUND = auto()
