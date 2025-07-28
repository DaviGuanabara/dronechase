from dataclasses import dataclass
from typing import Tuple

@dataclass
class NormalizationConfig:
    max_temporal_offset: int = 10
    max_radius: int = 100
