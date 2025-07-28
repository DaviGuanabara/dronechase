from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from core.notification_system.topics_enum import TopicsEnum


@dataclass
class PerceptionSnapshot():
    topics: Dict[TopicsEnum, Dict]
    publisher_id: int
    step: int

    @property
    def imu(self) -> Optional[Dict]:
        return self.topics.get(TopicsEnum.INERTIAL_DATA_BROADCAST)

    @property
    def lidar(self) -> Optional[Dict]:
        return self.topics.get(TopicsEnum.LIDAR_DATA_BROADCAST)
    
    def update(self, topic: TopicsEnum, data: Dict) -> None:
        """
        Update the snapshot with new data for a specific topic.
        If the topic does not exist, it will be created.
        """
        if topic not in self.topics:
            self.topics[topic] = {}
        self.topics[topic].update(data)
