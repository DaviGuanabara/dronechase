from typing import Any, Callable, Dict, List
from core.dataclasses.message_context import MessageContext
from core.notification_system.topics_enum import TopicsEnum

class Broker:
    def __init__(self):
        self._subscribers: Dict[TopicsEnum,
                                List[Callable[[Dict, MessageContext], None]]] = {}

    def subscribe(self, topic: TopicsEnum, subscriber: Callable[[Dict, MessageContext], None]) -> None:
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(subscriber)

    def unsubscribe(self, topic: TopicsEnum, subscriber: Callable[[Dict, MessageContext], None]) -> None:
        if topic in self._subscribers:
            self._subscribers[topic].remove(subscriber)

    def publish(self, topic: TopicsEnum, message: Dict, message_context: MessageContext) -> None:
        for subscriber in self._subscribers.get(topic, []):
            subscriber(message, message_context)

