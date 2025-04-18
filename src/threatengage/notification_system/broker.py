from typing import Any, Callable, Dict, List

class Broker:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Any, int], None]]] = {} 

    def subscribe(self, topic: str, subscriber: Callable[[Any, int], None]) -> None: 
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(subscriber)

    def unsubscribe(self, topic: str, subscriber: Callable[[Any, int], None]) -> None: 
        if topic in self._subscribers:
            self._subscribers[topic].remove(subscriber)

    def publish(self, topic: str, message: Any, publisher_id: int) -> None:
        for subscriber in self._subscribers.get(topic, []):
            subscriber(message, publisher_id)

