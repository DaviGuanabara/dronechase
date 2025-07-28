import random
from typing import Dict, List, Optional

from core.dataclasses.perception_snapshot import PerceptionSnapshot
from core.notification_system.topics_enum import TopicsEnum


class LiDARBufferManager:
    def __init__(self, current_step=0, max_buffer_size: int = 1):

        self.buffer: Dict[int, List[Optional[PerceptionSnapshot]]] = {}
        self.max_buffer_size = max_buffer_size
        self.current_step = current_step

    def _add_publisher(self, publisher_id: int) -> None:
        if publisher_id not in self.buffer:
            self.buffer[publisher_id] = [None] * self.max_buffer_size

    def _slide_buffer(self) -> None:
        """
        Slide the buffer for a specific publisher and topic.
        This will remove the oldest message and shift the rest.
        """
        for publisher_id, snapshots in self.buffer.items():
            # Shift buffer window to the left (oldest removed)
            self.buffer[publisher_id] = [None] + snapshots[:-1]

    def update_current_step(self, step: int) -> None:

        delta_step = step - self.current_step
        for _ in range(delta_step):
            self._slide_buffer()

        self.current_step = step


    def _get_snapshot(self, publisher_id: int, step: int) -> Optional[PerceptionSnapshot]:
        if publisher_id not in self.buffer:
            return None

        if step < 0:
            step = self.current_step
        delta_step = self.current_step - step
        if delta_step < 0 or delta_step >= self.max_buffer_size:
            return None

        return self.buffer[publisher_id][delta_step]

    def close_buffer(self, publisher_id: int, topic: TopicsEnum) -> None:
        """
        I am not sure if this is necessary to do anything.
        Needed for compatibility with the old code (LIDAR).
        """
        pass

    def get_latest_from_topic(self, topic: TopicsEnum) -> Dict[int, Dict]:
        """
        returns buffer_data = {publisher_id: topic_message}
        something like:
        {
            1: message1,
            2: message2,
            3: message3
        }
        """

        buffer_data: Dict[int, Dict] = {}
        for publisher_id, snapshots in self.buffer.items():
            snapshot = snapshots[0]  # Get the latest snapshot for this publisher
            if snapshot and topic in snapshot.topics:
                buffer_data[publisher_id] = snapshot.topics[topic]
        return buffer_data


    def buffer_message(self, message: Dict, publisher_id: int, topic: TopicsEnum, step:int = -1) -> None:

        self._add_publisher(publisher_id)
        if step < 0:
            step = self.current_step
        delta_step = self.current_step - step
        if delta_step < 0 or delta_step >= self.max_buffer_size:
            return None

        snapshot = self.buffer[publisher_id][delta_step]
        if snapshot is None:
            snapshot = PerceptionSnapshot(
                topics={}, publisher_id=publisher_id, step=step)
            self.buffer[publisher_id][delta_step] = snapshot

        snapshot.update(topic=topic, data=message)

    def _random_publisher_id(self, exclude_publisher_id: int) -> Optional[int]:
        eligible_ids = [
            publisher_id for publisher_id in self.buffer if publisher_id != exclude_publisher_id]
        return random.choice(eligible_ids) if eligible_ids else None

    
    def _random_snapshot(self, publisher_id: int) -> Optional[PerceptionSnapshot]:
        if publisher_id not in self.buffer:
            return None
        return random.choice(self.buffer[publisher_id])

    def get_neighborhood(self, n_neighbors: int, exclude_publisher_id: int) -> List[PerceptionSnapshot]:
        neighborhood = []
        for _ in range(n_neighbors):
            publisher_id = self._random_publisher_id(exclude_publisher_id)
            if publisher_id is None:
                raise ValueError("No publishers available in the buffer.")
            neighborhood.append(self._random_snapshot(publisher_id))
        return neighborhood

    def get_latest_snapshot(self, publisher_id: int) -> Optional[PerceptionSnapshot]:
        """Get the latest snapshot for a specific publisher at the current step."""
        return self._get_snapshot(publisher_id, self.current_step)