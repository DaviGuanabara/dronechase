import random
from typing import Dict, List, Optional

from core.dataclasses.perception_snapshot import PerceptionSnapshot
from core.entities.entity_type import EntityType
from core.notification_system.topics_enum import TopicsEnum


class LiDARBufferManager:
    def __init__(self, current_step=0, max_buffer_size: int = 1):

        self.buffer: Dict[int, List[Optional[PerceptionSnapshot]]] = {}
        self.publisher_entity_map: Dict[int, EntityType] = {}
        self.max_buffer_size = max_buffer_size
        self.current_step = current_step

    def _add_publisher(self, publisher_id: int, entity_type: EntityType) -> None:
        if publisher_id not in self.buffer:
            self.buffer[publisher_id] = [None] * self.max_buffer_size
            self.publisher_entity_map[publisher_id] = entity_type

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


    def _get_snapshot(self, publisher_id: int, delta_step: int) -> Optional[PerceptionSnapshot]:
        if publisher_id not in self.buffer:
            return None

        delta_step = min(max(delta_step, 0), self.max_buffer_size - 1)
        return self.buffer[publisher_id][delta_step]


    def retrieve_snapshots(
        self,
        publisher_ids: Optional[List[int]] = None,
        topic: Optional[TopicsEnum] = None,
        entity_type: Optional[EntityType] = None,
        delta_step: Optional[int] = None,
        exclude_publisher_id: Optional[int] = None
    ) -> List[PerceptionSnapshot]:
        """
        Returns a list of PerceptionSnapshot filtered by:
        - publisher_ids (if specified)
        - exclude_publisher_id (if specified)
        - topic (if specified, only snapshots that contain this topic are returned)
        - entity_type (if specified)
        - delta_step from the current step (default is None, which means random)

        Only non-None, valid snapshots are included.
        """
        snapshots: List[PerceptionSnapshot] = []
        target_publishers = publisher_ids if publisher_ids is not None else list(self.buffer.keys())

        for publisher_id in target_publishers:
            if publisher_id == exclude_publisher_id:
                continue
            if publisher_id not in self.buffer:
                continue
            if entity_type is not None and self.publisher_entity_map.get(publisher_id) != entity_type:
                continue

            # Choose delta_step
            if delta_step is None:
                max_valid = min(self.max_buffer_size - 1, self.current_step)
                random_delta = random.randint(0, max_valid)
            else:
                random_delta = delta_step

            snapshot = self._get_snapshot(publisher_id, random_delta)
            if snapshot is None:
                continue
            if topic is not None and topic not in snapshot.topics:
                continue

            snapshots.append(snapshot)

        return snapshots





    def close_buffer(self, publisher_id: int, topic: TopicsEnum) -> None:
        """
        I am not sure if this is necessary to do anything.
        Needed for compatibility with the old code (LIDAR).
        """
        pass

  



    def buffer_message(
        self,
        message: Dict,
        publisher_id: int,
        topic: TopicsEnum,
        step: int = -1
    ) -> None:
        """
        Buffers a message into the appropriate snapshot.
        Registers entity_type if it's a new publisher.
        """
        entity_type = message.get("entity_type")

        # Try to fallback to already known entity type
        if entity_type is None:
            entity_type = self.publisher_entity_map.get(publisher_id)

        # If we still don't know the entity type, we can't proceed
        if entity_type is None:
            print(f"[WARNING] Entity type missing for publisher {publisher_id}. Message discarded.")
            return

        self._add_publisher(publisher_id, entity_type)

        if step < 0:
            step = self.current_step

        delta_step = self.current_step - step
        if delta_step < 0 or delta_step >= self.max_buffer_size:
            return

        snapshot = self.buffer[publisher_id][delta_step]
        if snapshot is None:
            snapshot = PerceptionSnapshot(topics={}, publisher_id=publisher_id, step=step, entity_type=entity_type, max_delta_step=self.max_buffer_size)
            self.buffer[publisher_id][delta_step] = snapshot

        snapshot.update(topic=topic, data=message)


    def _random_publisher_id(
        self,
        exclude_publisher_id: int,
        quantity: int = 1,
        entity_type: Optional[EntityType] = None,
    ) -> List[int]:
        """
        Returns a list of one or more random publisher IDs that match the given filters.
        
        - Always returns a list (possibly empty).
        - If quantity == 1, the list contains at most one ID.
        - If quantity > 1, the list contains up to `quantity` IDs.
        """
        if eligible_ids := [
            publisher_id
            for publisher_id in self.buffer
            if publisher_id != exclude_publisher_id
            and (
                entity_type is None
                or self.publisher_entity_map.get(publisher_id) == entity_type
            )
        ]:
            return (
                random.sample(eligible_ids, min(quantity, len(eligible_ids)))
            )
        else:
            return []



    
    def _random_snapshot(self, publisher_id: int) -> Optional[PerceptionSnapshot]:
        if publisher_id not in self.buffer:
            return None
        return random.choice(self.buffer[publisher_id])

    def get_random_neighborhood(self, n_neighbors: int, exclude_publisher_id: int) -> List[PerceptionSnapshot]:

        random_publisher_ids = self._random_publisher_id(exclude_publisher_id, quantity=n_neighbors, entity_type=EntityType.LOYALWINGMAN)
        return self.retrieve_snapshots(publisher_ids=random_publisher_ids, delta_step=None)

    def get_latest_snapshot(self, publisher_id: int) -> Optional[PerceptionSnapshot]:
        """Get the latest snapshot for a specific publisher at the current step."""
        return self._get_snapshot(publisher_id, self.current_step)


    def get_latest_neighborhood(self, exclude_publisher_id: int) -> List[PerceptionSnapshot]:
        """
        Get the latest snapshot for all LOYALWINGMANs except the one excluded,
        using the current step (delta_step = 0).
        """
        return self.retrieve_snapshots(
            entity_type=EntityType.LOYALWINGMAN,
            delta_step=0,
            exclude_publisher_id=exclude_publisher_id
        )


    def get_latest_threats(self) -> List[PerceptionSnapshot]:
        """
        Get the latest snapshot (delta_step = 0) for all LOITERINGMUNITIONs.
        """
        return self.retrieve_snapshots(
            entity_type=EntityType.LOITERINGMUNITION,
            delta_step=0
        )

    def get_latests(self, exclude_publisher_id:Optional[int]=None):
        return self.retrieve_snapshots(
            entity_type=None,
            delta_step=0,
            exclude_publisher_id=exclude_publisher_id
        )
    
    def get_latest_messages_from_topic(
        self,
        topic: TopicsEnum,
        entity_type: Optional[EntityType] = None,
        exclude_publisher_id: Optional[int] = None
    ) -> Dict[int, Dict]:
        """
        Returns the latest messages (step = current_step) that contain the specified topic,
        structured as a dictionary: {publisher_id: message}

        Parameters:
        - topic: the topic to filter messages by
        - entity_type: optional filter for the entity type (e.g., LOYALWINGMAN)
        - exclude_publisher_id: optional ID to exclude from results

        Returns:
        - Dict[int, Dict]: A mapping of publisher_id to the topic-specific message
        """
        snapshots = self.retrieve_snapshots(
            topic=topic,
            entity_type=entity_type,
            delta_step=0,
            exclude_publisher_id=exclude_publisher_id
        )

        return {
            snapshot.publisher_id: snapshot.topics[topic]
            for snapshot in snapshots
            if topic in snapshot.topics
        }


