from dataclasses import dataclass
from typing import Optional
from core.entities.entity_type import EntityType

@dataclass
class MessageContext:
    publisher_id: int
    step: Optional[int] = None
    entity_type: Optional[EntityType] = None
    
    
