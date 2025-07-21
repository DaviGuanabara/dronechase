from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from typing import Dict
from core.entities.quadcopters.components.sensors.interfaces.sensor_interface import Sensor
from core.notification_system.topics_enum import TopicsEnum
from enum import Enum, auto
import numpy as np
import random
from math import ceil

class BaseLidar(Sensor):
    
    def buffer_lidar_data(self, message: Dict, publisher_id: int):
        pass

    def buffer_inertial_data(self, message: Dict, publisher_id: int):
        pass

    def buffer_step_broadcast(self, message: Dict, publisher_id: int):
        pass


class LiDARBufferManager:
    def __init__(self, current_step=0, max_buffer_size: int = 1):

        self.buffer: Dict[int, Dict[TopicsEnum, List[Optional[Dict]]]] = {}
        self.max_buffer_size = max_buffer_size
        self.current_step = current_step

    def _add_publisher(self, publisher_id: int) -> None:
        if publisher_id not in self.buffer:
            self.buffer[publisher_id] = {}

    def _add_topic(self, publisher_id: int, topic: TopicsEnum) -> None:
        self._add_publisher(publisher_id)
        if topic not in self.buffer[publisher_id]:
            self.buffer[publisher_id][topic] = [None] * self.max_buffer_size


    def update_current_step(self, step: int) -> None:
        self.current_step = step
        # slide buffer window
        for publisher_id, topics in self.buffer.items():
            for topic, messages in topics.items():
                self.buffer[publisher_id][topic] = [None] + messages[1:]


    def _buffer_temporal_message(self, message: Dict, topic: TopicsEnum,publisher_id: int, step: int) -> None:
        delta_step = self.current_step - step
        if delta_step >= self.max_buffer_size:
            return  #avoid overflow
        self.buffer[publisher_id][topic][delta_step] = message

    def buffer_message(self, message: Dict, publisher_id: int, topic: TopicsEnum, step: int = -1) -> None:
        self._add_topic(publisher_id, topic)

        # TODO: what if step explodes int range?
        # So, the delta step would be negative, and the message would not be buffered.
        if step < 0: 
            self._buffer_temporal_message(message, topic, publisher_id, self.current_step)
        elif step > self.current_step:
            raise ValueError(
                "Step must be less than or equal to the current step.")
        else:
            self._buffer_temporal_message(message, topic, publisher_id, step)


    def close_buffer(self, publisher_id: int, topic: TopicsEnum) -> None:
        if publisher_id in self.buffer and topic in self.buffer[publisher_id]:
            del self.buffer[publisher_id][topic]
            if not self.buffer[publisher_id]:
                del self.buffer[publisher_id]


    def _random_publisher(self) -> Optional[int]:
        return random.choice(list(self.buffer.keys())) if self.buffer else None
    
    def _random_topic(self, publisher_id: int) -> Optional[TopicsEnum]:
        if publisher_id not in self.buffer or not self.buffer[publisher_id]:
            return None
        return random.choice(list(self.buffer[publisher_id].keys()))


    def sample(self, n_messages: int) -> List[Dict]:
        messages = []
        for _ in range(n_messages):
            publisher_id = self._random_publisher()
            if publisher_id is None:
                break

            topic = self._random_topic(publisher_id)
            if topic is None:
                break

            if valid_messages := [
                m for m in self.buffer[publisher_id][topic] if m is not None
            ]:
                messages.append(random.choice(valid_messages))

        return messages
    
    def sample_from_publisher(self, publisher_id: int, n_messages: int) -> List[Dict]:
        if publisher_id not in self.buffer:
            return []

        topic_buffers = self.buffer[publisher_id]
        if not topic_buffers:
            return []

        buffer_length = self.max_buffer_size
        samples = []

        for _ in range(n_messages):
            # Keep delta_step (index) so we can use it later
            candidate_indices = []
            candidate_indices.extend(
                idx
                for idx in range(buffer_length)
                if all(
                    messages[idx] is not None
                    for messages in topic_buffers.values()
                )
            )
            if not candidate_indices:
                break  # no valid snapshot

            delta_step = random.choice(candidate_indices)
            absolute_step = self.current_step - delta_step

            sample = {"absolute_step": absolute_step}
            for topic, messages in topic_buffers.items():
                sample[topic] = messages[delta_step]

            samples.append(sample)

        return samples


    def get_latest_message(self, publisher_id: int, topic: TopicsEnum) -> Optional[Dict]:
        return self.buffer.get(publisher_id, {}).get(topic, [None])[0]
    
    def get_latest_from_publisher(self, publisher_id: int) -> Dict[TopicsEnum, Optional[Dict]]:
        latest_messages = {}
        if publisher_id in self.buffer:
            for topic in self.buffer[publisher_id]:
                latest_message = self.get_latest_message(publisher_id, topic)
                if latest_message is not None:
                    latest_messages[topic] = latest_message
        return latest_messages

    


    def distribute_samples(self, n_total: int, k_publishers: int) -> List[int]:
        base, remainder = divmod(n_total, k_publishers)
        return [base + (1 if i < remainder else 0) for i in range(k_publishers)]

    def get_neighbours(self, n: int, agent_publisher_id: int) -> List[Dict[TopicsEnum, Optional[Dict]]]:
        neighbour_ids = [
            publisher_id for publisher_id in self.buffer if publisher_id != agent_publisher_id]
        if not neighbour_ids:
            return []

        sample_distribution = self.distribute_samples(n, len(neighbour_ids))

        neighbours = []
        for publisher_id, n_samples in zip(neighbour_ids, sample_distribution):
            samples = self.sample_from_publisher(publisher_id, n_samples)
            neighbours.extend(samples)

        return neighbours


    def get_latest_from_topic(self, topic: TopicsEnum) -> Dict:
        latest_messages = {}
        for publisher_id in self.buffer:
            if topic in self.buffer[publisher_id]:
                latest_message = self.get_latest_message(publisher_id, topic)
                if latest_message is not None:
                    latest_messages[publisher_id] = latest_message
        return latest_messages



class Channels(Enum):
    DISTANCE_CHANNEL = 0
    FLAG_CHANNEL = 1


class CoordinateConverter:
    @staticmethod
    def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
        radius, theta, phi = spherical
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.array([x, y, z])

    @staticmethod
    def cartesian_to_spherical(cartesian: np.ndarray) -> np.ndarray:
        x, y, z = cartesian
        radius = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / radius)
        phi = np.arctan2(y, x)
        return np.array([radius, theta, phi])
