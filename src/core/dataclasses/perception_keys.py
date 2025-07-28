from dataclasses import dataclass
from core.notification_system import TopicsEnum

@dataclass
class PerceptionKeys:
    imu: str = TopicsEnum.INERTIAL_DATA_BROADCAST.name
    lidar: str = TopicsEnum.LIDAR_DATA_BROADCAST.name
    step: str = TopicsEnum.AGENT_STEP_BROADCAST.name
