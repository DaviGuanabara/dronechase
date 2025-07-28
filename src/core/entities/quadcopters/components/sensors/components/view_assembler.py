from typing import Tuple
from core.dataclasses.angle_grid import AngleGridConfig
from core.enums.channel_index import ChannelIndexConfig
from core.dataclasses.normalization import NormalizationConfig
from core.dataclasses.perception_keys import PerceptionKeys


class ViewAssembler:
    def __init__(
        self,
        perception_keys: PerceptionKeys,
        angle_grid: AngleGridConfig,
        channel_index: ChannelIndexConfig,
        normalization: NormalizationConfig
    ):
        self.perception_keys = perception_keys
        self.angle_grid = angle_grid
        self.channel_index = channel_index
        self.normalization = normalization


    def update_neighborhood(self, neighborhood: Tuple[float, float, float]) -> None:
        # Update the neighborhood based on the provided configuration
        self.neighborhood = neighborhood

    def update_agent_viewframe(self, agent: Tuple[float, float, float]) -> None:
        # Update the agent's position based on the provided configuration
        self.agent = agent
