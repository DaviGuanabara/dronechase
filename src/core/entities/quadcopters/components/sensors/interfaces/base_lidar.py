import math
from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod
from typing import Dict
from core.dataclasses.angle_grid import LIDARSpec
from core.dataclasses.message_context import MessageContext
from core.dataclasses.perception_snapshot import PerceptionSnapshot
from core.entities.quadcopters.components.sensors.interfaces.sensor_interface import Sensor
from core.enums.channel_index import LidarChannels
from core.notification_system.topics_enum import TopicsEnum
from enum import Enum, auto
import numpy as np
import random
from math import ceil
from core.entities.quadcopters.components.sensors.components.lidar_buffer import LiDARBufferManager
from core.entities.quadcopters.components.sensors.components.lidar_math import LidarMath

class BaseLidar(Sensor):
    
    def __init__(self, parent_id, client_id, debug = False, radius: float = 1.0, resolution: float = 1.0, n_neighbors_min: int = 1, n_neighbors_max: int = 10):
        
        self.parent_id = parent_id
        self.client_id = client_id
        self.debug = debug

        self.n_neighbors_min: int = n_neighbors_min
        self.n_neighbors_max: int = n_neighbors_max

        self.lidar_spec = LIDARSpec(theta_initial_radian=0, theta_final_radian=np.pi,
                                    phi_initial_radian=-np.pi, phi_final_radian=np.pi,
                                    resolution=resolution, n_channels=len(LidarChannels), max_radius=radius)
        
        self.math = LidarMath(self.lidar_spec)
    
        self.sphere: np.ndarray = self.lidar_spec.empty_sphere()
        self.buffer_manager = LiDARBufferManager(current_step=0, max_buffer_size=10)
        
    def buffer_lidar_data(self, message: Dict, message_context: MessageContext):
        topic: TopicsEnum = TopicsEnum.LIDAR_DATA_BROADCAST
        self._buffer_data(message, message_context, topic)

    def buffer_inertial_data(self, message: Dict, message_context: MessageContext):
        topic: TopicsEnum = TopicsEnum.INERTIAL_DATA_BROADCAST
        self._buffer_data(message, message_context, topic)

    def buffer_step_broadcast(self, message: Dict, message_context: MessageContext):
        self.buffer_manager.update_current_step(message.get("step", self.buffer_manager.current_step))


    def _buffer_data(self, message: Dict, message_context: MessageContext, topic: TopicsEnum):
        """
        Método genérico para bufferizar dados de diferentes tópicos.
        """

        message["message_context"] = message_context
        if "termination" in message:
            self.buffer_manager.close_buffer(message_context.publisher_id, topic)
        else:
            step = message_context.step if message_context.step is not None else -1
            self.buffer_manager.buffer_message(message, message_context=message_context, topic=topic)


    def _reset_sphere(self):
        if self.sphere:
            self.lidar_spec.reset_sphere(self.sphere)


    def get_agent_snapshot(self) -> Optional[PerceptionSnapshot]:
        return self.buffer_manager.get_latest_snapshot(self.parent_id)
    
    @abstractmethod
    def get_data_shape(self) -> Tuple:
        pass

