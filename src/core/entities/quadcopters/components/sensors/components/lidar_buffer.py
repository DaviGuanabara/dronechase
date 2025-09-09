import random
from typing import Dict, List, Optional

from core.dataclasses.message_context import MessageContext
from core.dataclasses.perception_snapshot import PerceptionSnapshot
from core.entities.entity_type import EntityType
from core.notification_system.topics_enum import TopicsEnum


class SnapshotBuffer:
    """
    Fixed-size temporal buffer for storing and retrieving PerceptionSnapshots.

    Each publisher (e.g., drone or sensor) is associated with its own sliding window
    of snapshots, capped at `max_size`. Snapshots are updated step-by-step, where
    newer entries overwrite the oldest ones (FIFO behavior).

    This class enforces encapsulation: snapshots can only be accessed or updated via
    its API, ensuring consistent handling of temporal normalization and entity mapping.
    """
    def __init__(self, max_size: int):
        """
        Initialize a new SnapshotBuffer.

        Args:
            max_size (int): Maximum number of snapshots to store per publisher.
                            This defines the temporal depth of the buffer.
        """
        self.max_size = max_size
        self._buffer: Dict[int, List[Optional[PerceptionSnapshot]]] = {}
        self.publisher_entity_map: Dict[int, EntityType] = {}
        self.current_step = 0

    def add_publisher(self, publisher_id: int, entity_type: EntityType):
        """
        Register a new publisher with an empty snapshot buffer.

        Args:
            publisher_id (int): Unique identifier of the publisher (e.g., drone ID).
            entity_type (EntityType): Semantic type of the publisher (agent, ally, etc.).

        Notes:
            - If the publisher already exists, no action is taken.
            - Initializes a buffer of length `max_size` filled with `None`.
        """
        if publisher_id not in self._buffer:
            self._buffer[publisher_id] = [None] * self.max_size
            self.publisher_entity_map[publisher_id] = entity_type

    @property
    def publisher_ids(self):
        return list(self._buffer.keys())

    def _slide(self):
        """
        Advance the buffer by one step for all publishers.

        Behavior:
            - Oldest snapshot is discarded.
            - Remaining snapshots are shifted right by one position.
            - Index 0 is left empty for insertion of the newest snapshot.
        """
        for publisher_id, snapshots in self._buffer.items():
            self._buffer[publisher_id] = [None] + snapshots[:-1]

    def update_current_step(self, step: int) -> None:
        
        if step < self.current_step:
            return  # Ignore updates to past steps

        delta_step = step - self.current_step
        for _ in range(delta_step):
            self._slide()

        self.current_step = step


    def get_snapshot(self, publisher_id: int, delta_step: int) -> Optional[PerceptionSnapshot]:
        """
        Retrieve a snapshot at a given temporal offset.
        Snapshots must be treated as immutable once retrieved.
        Args:
            publisher_id (int): ID of the publisher to query.
            delta_step (int): Temporal offset into the buffer (0 = most recent).
                              Values outside the buffer range are clipped.
            current_step (int): Current absolute simulation step, used to normalize time.

        Returns:
            Optional[PerceptionSnapshot]: The requested snapshot if available, otherwise None.

        Behavior:
            - If the snapshot exists, its temporal information is normalized using
              `snapshot.update_normalized_delta()`.
            - If the publisher or snapshot does not exist, returns None.
        """
        snapshot = self._get_raw_snapshot(publisher_id, delta_step)
        if snapshot is not None:
            snapshot.update_normalized_delta(current_absolute_step=self.current_step,
                                            max_delta_step=self.max_size)

            #print(f"[DEBUG] LIDARBUFFER GET_SNAPSHOT() - normalized delta: {snapshot.normalized_delta}")
        return snapshot
    
    def get_random_snapshot(self, publisher_id: int, fresher_delta_step: int) -> Optional[PerceptionSnapshot]:
        if publisher_id not in self._buffer:
            return None

        random_delta_step = random.randint(fresher_delta_step, self.max_size - 1)
        return self.get_snapshot(publisher_id, random_delta_step)
    
    def get_random_neighborhood(
        self,
        n_neighbors: int,
        exclude_publisher_id: int = -1,
        entity_type: Optional[EntityType] = None,
        min_delta_step: int = 1
    ) -> List[PerceptionSnapshot]:
        """
        Retrieve a random set of neighbor snapshots from publishers.

        Args:
            n_neighbors (int): Number of neighbor snapshots to retrieve.
            exclude_publisher_id (int): Publisher to exclude (default: -1).
            entity_type (Optional[EntityType]): Filter by entity type (if provided).
            fresher_delta_step (int): Minimum freshness (default = 0 → freshest allowed).

        Returns:
            List[PerceptionSnapshot]: Randomly chosen snapshots (may contain fewer than n_neighbors if unavailable).
        """
        candidates = [
            pid for pid, et in self.publisher_entity_map.items()
            if pid != exclude_publisher_id and (entity_type is None or et == entity_type)
        ]

        if not candidates:
            return []

        chosen_ids = random.sample(candidates, min(n_neighbors, len(candidates)))

        snaps = []
        for pid in chosen_ids:
            if snapshot := self.get_random_snapshot(pid, min_delta_step):
                snaps.append(snapshot)

        return snaps

            


    def _get_raw_snapshot(self, publisher_id: int, delta_step: int) -> Optional[PerceptionSnapshot]:
        """
        Internal accessor: returns raw snapshot from buffer, without normalization.
        """
        if publisher_id not in self._buffer:
            return None
        delta_step = min(max(delta_step, 0), self.max_size - 1)
        return self._buffer[publisher_id][delta_step]

    def _insert_snapshot(self, snapshot: PerceptionSnapshot, delta_step: int, track_snapshot: bool = False) -> None:
        """
        Insert an existing snapshot into the buffer at the specified delta_step.

        Args:
            snapshot (PerceptionSnapshot): Snapshot to be inserted. Its publisher_id
                will be used as the buffer key.
            delta_step (int): Relative index in the buffer where the snapshot
                should be stored. Clipped to [0, max_size-1].
        """
        publisher_id = snapshot.publisher_id

        if publisher_id not in self._buffer:
            # Garantir consistência — adiciona publisher se ainda não existir
            self.add_publisher(publisher_id, snapshot.entity_type)

        delta_step = min(max(delta_step, 0), self.max_size - 1)
        self._buffer[publisher_id][delta_step] = snapshot

        if track_snapshot:
            if not hasattr(self, '_tracked_snapshots'):
                self._tracked_snapshots = set()
            self._tracked_snapshots.add((publisher_id, delta_step, snapshot))

    def _ensure_snapshot(
        self,
        publisher_id: int,
        delta_step: int,
        entity_type: EntityType,
        track_snapshot: bool = False
    ) -> PerceptionSnapshot:
        """
        Guarantee that a snapshot exists for (publisher, delta_step).
        If none exists, create a new PerceptionSnapshot and store it.
        Never overwrites an existing snapshot.
        """
        snapshot = self._get_raw_snapshot(publisher_id, delta_step)
        if snapshot is None:
            snapshot = PerceptionSnapshot(
                topics={},
                publisher_id=publisher_id,
                step=self.current_step - delta_step,
                entity_type=entity_type
            )
            self._insert_snapshot(snapshot, delta_step)
            #print(f"[DEBUG] Snapshot created for publisher {publisher_id} at delta_step {delta_step}.")

        return snapshot

    def record_message(
        self,
        publisher_id: int,
        delta_step: int,
        entity_type: EntityType,
        topic: TopicsEnum,
        message: Dict,
        debug: bool = False
    ) -> None:
        """
        Ensure a snapshot exists for (publisher, delta_step), and
        update it with a new broadcast message under the given topic.
        Uses `_ensure_snapshot` to safely retrieve the snapshot if missing.
        """
        
        snapshot = self._ensure_snapshot(publisher_id, delta_step, entity_type, debug)

        #if hasattr(self, '_tracked_snapshots') and (publisher_id, delta_step, snapshot) in self._tracked_snapshots:
        #    print(f"[DEBUG] Attempt to modify Snapshot tracked for {publisher_id} at delta_step {delta_step}.")
        #    print("before, topic lidar data broadcast")
        #    print(snapshot.topics[TopicsEnum.LIDAR_DATA_BROADCAST])
        #    print("message:")
        #    print(message)

        snapshot.update(topic=topic, data=message)

        #if debug:
        #    print(f"[DEBUG] Recorded message for {publisher_id} at delta_step {delta_step}: {message}")
            #print(snapshot.topics[TopicsEnum.LIDAR_DATA_BROADCAST])

        #if snapshot.topics[topic] is None:
        #    snapshot.topics[topic] = message



    def remove_publisher(self, publisher_id: int) -> None:
        """
        Remove a publisher and all its stored snapshots from the buffer.

        Args:
            publisher_id (int): Unique identifier of the publisher to be removed.
        """
        if publisher_id in self._buffer:
            del self._buffer[publisher_id]
        if publisher_id in self.publisher_entity_map:
            del self.publisher_entity_map[publisher_id]

    def print_status(self, freshest_data: int) -> None:
        print("\n📦 LiDARBufferManager Status Snapshot:")
        print(f"  ➤ Current Step: {self.current_step}")
        print(f"  ➤ Max Buffer Size: {self.max_size}\n")
        print(f"  ➤ Using freshest snapshot at Δ={freshest_data}")

        for publisher_id, snapshots in self._buffer.items():
            entity = self.publisher_entity_map.get(publisher_id, "UNKNOWN")
            valid_snapshots = [s for s in snapshots if s is not None]
            print(
                f"🛰️  Publisher ID: {publisher_id} | Entity: {entity} | Valid Snapshots: {len(valid_snapshots)}")

            for i, snapshot in enumerate(snapshots):
                if snapshot is None:
                    print(f"   └─ [Δ={i}] ⛔ Empty")
                else:
                    step_info = f"Step: {snapshot.step}" if snapshot.step is not None else "No Step"
                    topic_list = ", ".join(
                        t.name for t in snapshot.topics.keys()) if snapshot.topics else "No Topics"
                    print(
                        f"   └─ [Δ={i}] ✅ {step_info} | Topics: {topic_list}")
            print()

class LiDARBufferManager:
    def __init__(self, current_step=0, max_buffer_size: int = 1, verbose:bool = False):


        self._buffer:SnapshotBuffer = SnapshotBuffer(max_buffer_size)
        self._publisher_entity_map: Dict[int, EntityType] = {}

        self.STABLE_DELTA_STEP = 1
        self.verbose = verbose


    def reset(self, new_step: int = 0):
        self._buffer = SnapshotBuffer(self._buffer.max_size)
        #self._publisher_entity_map.clear() # the entities will kept same id.
        self._buffer.current_step = new_step


    @property
    def current_step(self) -> int:
        return self._buffer.current_step

    def update_current_step(self, step: Optional[int]) -> None:
        if step is not None:
            self._buffer.update_current_step(step)

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
        target_publishers = publisher_ids if publisher_ids is not None else self._buffer.publisher_ids

        for publisher_id in target_publishers:
            if publisher_id == exclude_publisher_id:
                continue
            if publisher_id not in self._buffer.publisher_ids:
                continue
            if entity_type is not None and self._publisher_entity_map.get(publisher_id) != entity_type:
                continue

            # Choose delta_step
            if delta_step is None:
                max_valid = min(self._buffer.max_size - 1, self._buffer.current_step) #If current_step is 1, max_valid will be 1
                random_delta = 0 if max_valid < self.STABLE_DELTA_STEP else random.randint(self.STABLE_DELTA_STEP, max_valid)
            else:
                random_delta = delta_step

            snapshot = self._buffer.get_snapshot(publisher_id, random_delta) #self._get_snapshot(publisher_id, random_delta)
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
        #TODO REMOVE ALL PUBLISHERS.
        self._buffer.remove_publisher(publisher_id)


  
    def buffer_message_directly(self,
                                 message: Dict,
                                 publisher_id: int,
                                 topic: TopicsEnum,):
        """
        This method is not implemented.
        It is a placeholder for future functionality.
        Há uma defasagem entre o IMU e o LIDAR em 1 step. para compensar isso,
        precisamos ajustar o step do lidar que é adicionado diretamente.
        """
        step = self._buffer.current_step - self.STABLE_DELTA_STEP
        #message["step"] = step
        message_context = MessageContext(
            publisher_id=publisher_id,
            step=step,
            entity_type=EntityType.LOYALWINGMAN  # self._publisher_entity_map.get(publisher_id)
        )

        #print(f"[DEBUG] FusedLIDAR BUFFER MESSAGE DIRECTLY - step: {step}, entity_type: {message_context.entity_type}, topic: {topic}")
        #print(message)

        self._buffer.record_message(
            publisher_id=message_context.publisher_id,
            delta_step=self._buffer.current_step - step,
            entity_type=EntityType.LOYALWINGMAN,  # Assuming the entity type is always LOYALWINGMAN for direct messages
            topic=topic,
            message=message,
            debug=True

        )

        #self.buffer_message(
        #    message=message,
        #    message_context=message_context,
        #    topic=topic
        #)


    def buffer_message(
        self,
        message: Dict,
        message_context: MessageContext,
        topic: TopicsEnum,
    ) -> None:
        """
        Buffers a message into the appropriate snapshot.
        Registers entity_type if it's a new publisher.
        """
        entity_type = message_context.entity_type

        # Try to fallback to already known entity type
        if entity_type is None:
            entity_type = self._publisher_entity_map.get(message_context.publisher_id)

        # If we still don't know the entity type, we can't proceed
        if entity_type is None:
            if self.verbose:
                print(f"[WARNING] Entity type missing for publisher {message_context.publisher_id}. Message discarded.")
            return
        
        step = message_context.step if message_context.step is not None else -1
        if step < 0:
            step = self._buffer.current_step

        delta_step = self._buffer.current_step - step
        if delta_step < 0 or delta_step >= self._buffer.max_size:
            return

        self._buffer.record_message(
            publisher_id=message_context.publisher_id,
            delta_step=delta_step,
            entity_type=entity_type,
            topic=topic,
            message=message
        )

    def get_random_neighborhood(self, n_neighbors: int, exclude_publisher_id: int = -1):
        return self._buffer.get_random_neighborhood(
            n_neighbors=n_neighbors,
            exclude_publisher_id=exclude_publisher_id,
            entity_type=EntityType.LOYALWINGMAN,
            min_delta_step=self.STABLE_DELTA_STEP
        )

    def get_latest_snapshot(self, publisher_id: int) -> Optional[PerceptionSnapshot]:
        """Get the latest snapshot for a specific publisher at the current step.
        The freshest snapshot available is delta_step = 1.
        """
        return self._buffer.get_snapshot(publisher_id, self.STABLE_DELTA_STEP)

    def get_latest_neighborhood(self, exclude_publisher_id: int) -> List[PerceptionSnapshot]:
        """
        Get the latest snapshot for all LOYALWINGMANs except the one excluded,
        using the current step (delta_step = 0).
        """
        return self.retrieve_snapshots(
            entity_type=EntityType.LOYALWINGMAN,
            delta_step=self.STABLE_DELTA_STEP,
            exclude_publisher_id=exclude_publisher_id
        )


    def get_latest_threats(self) -> List[PerceptionSnapshot]:
        """
        Get the latest snapshot (delta_step = 0) for all LOITERINGMUNITIONs.
        """
        return self.retrieve_snapshots(
            entity_type=EntityType.LOITERINGMUNITION,
            delta_step=self.STABLE_DELTA_STEP
        )

    def get_latests(self, exclude_publisher_id:Optional[int]=None):
        """
        The latest stable reading that reflects the world after the previous update cycle.
        maybe, There should be a (little) discrepancy between the real state and the extract state.
        """
        return self.retrieve_snapshots(
            entity_type=None,
            delta_step=self.STABLE_DELTA_STEP,
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
            delta_step=self.STABLE_DELTA_STEP,
            exclude_publisher_id=exclude_publisher_id
        )

        return {
            snapshot.publisher_id: snapshot.topics[topic]
            for snapshot in snapshots
            if topic in snapshot.topics
        }

    def print_buffer_status(self) -> None:
        self._buffer.print_status(self.STABLE_DELTA_STEP)
