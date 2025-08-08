from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from core.entities.entity_type import EntityType
from core.enums.channel_index import LidarChannels
from core.notification_system.topics_enum import TopicsEnum


@dataclass
class PerceptionSnapshot():
    topics: Dict[TopicsEnum, Dict]
    publisher_id: int
    step: int
    entity_type: EntityType
    max_delta_step: int


    def relative_step(self, current_absolute_step: int) -> float:
        delta = (self.step - current_absolute_step) / self.max_delta_step
        return np.clip(delta, 0.0, 1.0)

    @property
    def imu(self) -> Optional[Dict]:
        return self.topics.get(TopicsEnum.INERTIAL_DATA_BROADCAST)

    @property
    def lidar(self) -> Optional[Dict]:
        return self.topics.get(TopicsEnum.LIDAR_DATA_BROADCAST)

    @property
    def sphere(self) -> Optional[np.ndarray]:
        lidar = self.lidar
        return lidar.get("sphere") if lidar else None
    
    @property
    def lidar_features(self) -> Optional[np.ndarray]:
        lidar = self.lidar
        return lidar.get("features") if lidar else None
    
    def build_temporal_sphere(self, current_absolute_step: int) -> Optional[np.ndarray]:
        """
        Returns a LiDAR sphere with an added time channel representing delta_step,
        calculated as (current_step - self.step).
        """
        if self.sphere is None:
            return None

        TEMPORAL_N_CHANNELS = 3
        sphere_raw = self.sphere
        n_channels, n_theta, n_phi = sphere_raw.shape

        if n_channels != 3:
            raise ValueError(f"Expected 3 channels (distance, flag, time), got {n_channels}")

        t_sphere = np.zeros((TEMPORAL_N_CHANNELS, n_theta, n_phi), dtype=sphere_raw.dtype)
        t_sphere[LidarChannels.distance.value] = sphere_raw[LidarChannels.distance.value]
        t_sphere[LidarChannels.flag.value] = sphere_raw[LidarChannels.flag.value]

        delta_step = self.relative_step(current_absolute_step)
        t_sphere[LidarChannels.time.value].fill(delta_step)

        return t_sphere


    
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

