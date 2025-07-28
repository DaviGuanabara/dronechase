from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from core.entities.entity_type import EntityType
from core.notification_system.topics_enum import TopicsEnum


@dataclass
class PerceptionSnapshot():
    topics: Dict[TopicsEnum, Dict]
    publisher_id: int
    step: int
    entity_type: EntityType

    @property
    def imu(self) -> Optional[Dict]:
        return self.topics.get(TopicsEnum.INERTIAL_DATA_BROADCAST)

    @property
    def lidar(self) -> Optional[Dict]:
        return self.topics.get(TopicsEnum.LIDAR_DATA_BROADCAST)

    @property
    def sphere(self) -> Optional[np.ndarray]:
        lidar = self.lidar
        return lidar.get("sphere") if lidar and "sphere" in lidar else None
    
    def update(self, topic: TopicsEnum, data: Dict) -> None:
        """
        Update the snapshot with new data for a specific topic.
        If the topic does not exist, it will be created.
        """
        if topic not in self.topics:
            self.topics[topic] = {}
        self.topics[topic].update(data)


    @property
    def position(self) -> Optional[np.ndarray]:
        imu = self.imu
        if imu and "position" in imu:
            return np.asarray(imu["position"], dtype=np.float32)
        return None

    @property
    def velocity(self) -> Optional[np.ndarray]:
        imu = self.imu
        return np.asarray(imu["velocity"], dtype=np.float32) if imu and "velocity" in imu else None

    @property
    def attitude(self) -> Optional[np.ndarray]:
        imu = self.imu
        return np.asarray(imu["attitude"], dtype=np.float32) if imu and "attitude" in imu else None

    @property
    def quaternion(self) -> Optional[np.ndarray]:
        imu = self.imu
        return np.asarray(imu["quaternion"], dtype=np.float32) if imu and "quaternion" in imu else None

    @property
    def angular_rate(self) -> Optional[np.ndarray]:
        imu = self.imu
        return np.asarray(imu["angular_rate"], dtype=np.float32) if imu and "angular_rate" in imu else None

