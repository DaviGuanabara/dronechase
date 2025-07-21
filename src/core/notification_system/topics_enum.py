from enum import IntEnum, auto


class TopicsEnum(IntEnum):
    AGENT_STEP_BROADCAST = 0 # "Agent_Step_Broadcast"
    INERTIAL_DATA_BROADCAST = auto()  # "Inertial_Data_Broadcast"
    LIDAR_DATA_BROADCAST = auto()  # "Lidar_Data_Broadcast"
    SIMULATION = auto()  # "Simulation"
