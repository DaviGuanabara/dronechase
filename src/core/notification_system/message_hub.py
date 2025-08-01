from dataclasses import asdict
from core.dataclasses.message_context import MessageContext
from core.entities.entity_type import EntityType
from core.notification_system.topics_enum import TopicsEnum
from core.notification_system.broker import Broker
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
        self._brokers: Dict[TopicsEnum, Broker] = {}
        self._publishers_dict: Dict[int, List[TopicsEnum]] = {}

    def get_broker(self, topic: TopicsEnum) -> Broker:
        """Get or create a broker for a specific topic."""
        return self._brokers.setdefault(topic, Broker())

    def subscribe(self, topic: TopicsEnum, subscriber: Callable[[Dict, MessageContext], None]) -> None:
        """Subscribe to a specific topic."""
        self.get_broker(topic).subscribe(topic, subscriber)

    def unsubscribe(self, topic: TopicsEnum, subscriber: Callable[[Dict, MessageContext], None]) -> None:
        """Unsubscribe from a specific topic."""
        self.get_broker(topic).unsubscribe(topic, subscriber)

    def publish(self, topic: TopicsEnum, message: Dict, message_context: MessageContext) -> None:
        """Publish a message to a topic."""
        self._register_publisher(topic, message_context.publisher_id)
        self.get_broker(topic).publish(topic, message, message_context)

    def terminate(self, publisher_id: int, topic: Optional[TopicsEnum] = None) -> None:
        if topic:
            # Terminate the specific topic
            self._unregister_publisher_topic(topic, publisher_id)
        else:
            # Terminate all topics for the publisher
            self._unregister_publisher(publisher_id)
    
    def _register_publisher(self, topic: TopicsEnum, publisher_id: int):
        """Register a publisher for a specific topic."""
        if publisher_id not in self._publishers_dict:
            self._publishers_dict[publisher_id] = []
            
        if topic not in self._publishers_dict[publisher_id]:
            self._publishers_dict[publisher_id].append(topic)

    def _unregister_publisher(self, publisher_id: int):
        """Unregister a publisher and send termination messages to all its topics."""
        
        topics = self._publishers_dict.get(publisher_id, []).copy()
        message_context = self.create_message_context(publisher_id=publisher_id)
        for topic in topics:
            self.publish(topic=topic, message={"termination": True}, message_context=message_context)
            
        if publisher_id in self._publishers_dict:
            del self._publishers_dict[publisher_id]
                 
    def _unregister_publisher_topic(self, topic: TopicsEnum, publisher_id: int):
        """Unregister a publisher and send termination messages to all its topics."""
        message_context = self.create_message_context(publisher_id=publisher_id)
        self.publish(topic=topic, message={"termination": True}, message_context=message_context)
            
        # Remove the topic from the list of topics for the publisher
        if publisher_id in self._publishers_dict:
            topics = self._publishers_dict[publisher_id]
            if topic in topics:
                topics.remove(topic)
            

    def create_message_context(self, publisher_id:int, step: Optional[int] = None, entity_type: Optional[EntityType] = None) -> MessageContext:
        return MessageContext(publisher_id=publisher_id, step=step, entity_type=entity_type)

        #return asdict(context)