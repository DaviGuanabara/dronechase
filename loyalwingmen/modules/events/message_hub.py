from .broker import Broker
from typing import Dict, Callable, Any, List, Optional
import threading

class MessageHub:
    _thread_local_data = threading.local()

    def __new__(cls):
        if not hasattr(cls._thread_local_data, '_instance'):
            cls._thread_local_data._instance = super(MessageHub, cls).__new__(cls)
            cls._thread_local_data._instance._initialize()
        return cls._thread_local_data._instance
            
    def _initialize(self) -> None:
        """Initialize attributes for the instance."""
        self._brokers: Dict[str, Broker] = {}
        self._publishers_dict: Dict[int, List[str]] = {}

    def get_broker(self, topic: str) -> Broker:
        """Get or create a broker for a specific topic."""
        return self._brokers.setdefault(topic, Broker())

    def subscribe(self, topic: str, subscriber: Callable[[Any, int], None]) -> None:
        """Subscribe to a specific topic."""
        self.get_broker(topic).subscribe(topic, subscriber)

    def unsubscribe(self, topic: str, subscriber: Callable[[Any, int], None]) -> None:
        """Unsubscribe from a specific topic."""
        self.get_broker(topic).unsubscribe(topic, subscriber)

    def publish(self, topic: str, message: Dict, publisher_id: int) -> None:
        """Publish a message to a topic."""
        self._register_publisher(topic, publisher_id)
        self.get_broker(topic).publish(topic, message, publisher_id)

    def terminate(self, publisher_id: int, topic: Optional[str] = None) -> None:
        if topic:
            # Terminate the specific topic
            self._unregister_publisher_topic(topic, publisher_id)
        else:
            # Terminate all topics for the publisher
            self._unregister_publisher(publisher_id)
    
    def _register_publisher(self, topic: str, publisher_id: int):
        """Register a publisher for a specific topic."""
        if publisher_id not in self._publishers_dict:
            self._publishers_dict[publisher_id] = []
            
        if topic not in self._publishers_dict[publisher_id]:
            self._publishers_dict[publisher_id].append(topic)

    def _unregister_publisher(self, publisher_id: int):
        """Unregister a publisher and send termination messages to all its topics."""
        
        topics = self._publishers_dict.get(publisher_id, []).copy()
        for topic in topics:
            self.publish(topic=topic, message={"termination": True}, publisher_id=publisher_id)
            
        if publisher_id in self._publishers_dict:
            del self._publishers_dict[publisher_id]
                 
            
    def _unregister_publisher_topic(self, topic: str, publisher_id: int):
        """Unregister a publisher and send termination messages to all its topics."""
        
        self.publish(topic=topic, message={"termination": True}, publisher_id=publisher_id)
            
        # Remove the topic from the list of topics for the publisher
        if publisher_id in self._publishers_dict:
            topics = self._publishers_dict[publisher_id]
            if topic in topics:
                topics.remove(topic)
            

    