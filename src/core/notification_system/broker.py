from typing import Any, Callable, Dict, List
from core.notification_system.topics_enum import TopicsEnum

class Broker:
    def __init__(self):
        self._subscribers: Dict[TopicsEnum,
                                List[Callable[[Any, int], None]]] = {}

    def subscribe(self, topic: TopicsEnum, subscriber: Callable[[Any, int], None]) -> None:
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(subscriber)

    def unsubscribe(self, topic: TopicsEnum, subscriber: Callable[[Any, int], None]) -> None:
        if topic in self._subscribers:
            self._subscribers[topic].remove(subscriber)

    def publish(self, topic: TopicsEnum, message: Any, publisher_id: int) -> None:
        for subscriber in self._subscribers.get(topic, []):
            subscriber(message, publisher_id)

