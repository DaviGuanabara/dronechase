
from typing import Dict, List, Optional, Tuple
from core.dataclasses.perception_snapshot import PerceptionSnapshot
from core.entities.entity_type import EntityType
from core.entities.quadcopters.components.sensors.lidar import LIDAR
from core.notification_system.message_hub import MessageHub
from core.notification_system.topics_enum import TopicsEnum
from core.entities.quadcopters.components.sensors.interfaces.base_lidar import BaseLidar
from core.entities.quadcopters.components.sensors.components.lidar_buffer import LiDARBufferManager


import numpy as np
import random
import time

from core.dataclasses.normalization import NormalizationConfig
from core.dataclasses.angle_grid import LIDARSpec
from core.enums.channel_index import LidarChannels
from core.dataclasses.perception_keys import PerceptionKeys


class FusedLIDAR(BaseLidar):
    """
    FusedLiDAR estende a classe LiDAR para permitir a fusão de dados de sensores 
    de múltiplos drones aliados, com objetivo de fornecer observações mais ricas 
    e robustas ao agente principal.

    A lógica é baseada em um canal de comunicação no estilo publish-subscribe,
    onde cada drone transmite suas leituras de LiDAR, posição e timestamp. 
    Os demais drones podem consumir essas informações para construir um buffer 
    local e aplicar estratégias de fusão, como bootstrap aleatório.

    FUSÃO IMPLEMENTADA:
    - Fusão baseada em amostras aleatórias com restrições de distância e tempo.
    - Combinação simplificada por média aritmética dos dados válidos.

    PARÂMETROS DE FILTRAGEM:
    - `_max_time_diff`: restrição de tempo entre leituras (em segundos)
    - `_max_distance`: distância máxima entre drones para considerar leitura
    - `_max_buffer_size`: máximo de mensagens mantidas no buffer local

    AÇÕES REALIZADAS:
    - Publica automaticamente suas próprias leituras via `MessageHub`
    - Escuta mensagens de outros drones para manter buffer de leitura aliado
    - Executa fusão com base em amostras recentes e próximas

    USO:
        lidar = FusedLiDAR(...)
        lidar.enable_fusion()  # ativa a lógica de fusão ?

        
    O lidar pega a posição de todos os drones. 
    Ele recebe essas posições através da classe quadcopter, que por sua vez, subscreve no canal de mensagens
    e recebe as atualizações do IMU. No caso dos dados do LiDAR, o FusedLiDAR recebe, adicionalmente a saída anterior.

    Mas no meu caso, preciso também receber a leitura do LiDAR de todos os drones aliados.

    TODOs:
    """

    def __init__(
        self,
        parent_id: int,
        client_id: int,
        radius: float = 20,
        resolution: float = 8,
        n_neighbors_min: int = 1,
        n_neighbors_max: int = 5,
        debug: bool = False,
    ):
        super().__init__(parent_id, client_id, debug, radius, resolution, n_neighbors_min, n_neighbors_max)


    def bootstrap(self) -> List[PerceptionSnapshot]:

        n = random.choice(range(self.n_neighbors_min, self.n_neighbors_max))
        return self.buffer_manager.get_random_neighborhood(
            n_neighbors=n, exclude_publisher_id=self.parent_id
        )

    def _update_sphere_stack(self):
        agent = self.buffer_manager.get_latest_snapshot(self.parent_id)

        if agent is None or agent.sphere is None:
            return

        neighborhood: List[PerceptionSnapshot] = self.bootstrap()
        sphere_stack: List[np.ndarray] = [agent.sphere]

        for neighbor in neighborhood:
            neighbor_sphere_reframed = self.math.neighbor_sphere_from_new_frame(neighbor=neighbor, agent=agent)

            if neighbor_sphere_reframed is not None:
                sphere_stack.append(neighbor_sphere_reframed)

        return sphere_stack

    def get_snapshots_by_distance(self) -> List[PerceptionSnapshot]:
        """
        Return threats (LOITERINGMUNITION) within the max LiDAR radius.
        """
        latests = self.buffer_manager.get_latests(self.parent_id)
        max_radius: float = self.lidar_spec.max_radius

        agent_snapshot = self.get_agent()
        if not agent_snapshot or agent_snapshot.position is None:
            return []

        agent_pos = agent_snapshot.position

        return [
            latest for latest in latests
            if latest.position is not None and np.linalg.norm(latest.position - agent_pos) <= max_radius
        ]

    
    
    


    
    def update_data(self):
        # retrieve last positions and entity types from all publishers, excluding agent
        
        snapshots = self.get_snapshots_by_distance()
        agent = self.get_agent()

        if not agent or not agent.position:
            return

        features = []
        for snapshot in snapshots:
            if not snapshot.position:
                continue
            # reframe
            reframed_snapshot_position = self.math.reframe(snapshot.position, np.zeros(3), agent.position)
            spherical = self.math.cartesian_to_spherical(reframed_snapshot_position)
            spherical = self.math.normalize_spherical(spherical)
            # the lidars data are always
            # from the loyal wingman.
           
            delta_step = 0 #current delta step
            # mount features
            features.append((*spherical, snapshot.entity_type, delta_step))

        new_sphere = self.lidar_spec.empty_sphere()
        self.sphere = self.math.add_features(new_sphere, features)
        self.buffer_manager.buffer_message({"lidar": self.sphere, "step": self.buffer_manager.current_step}, self.parent_id, TopicsEnum.LIDAR_DATA_BROADCAST)



    def read_data(self) -> Dict:
        self.sphere_stack = self._update_sphere_stack()
        #TODO: I need to add mask on fused lidar
        return {"lidar": self.sphere, "fused_lidar": self.sphere_stack}
    
    def reset(self):
        pass


    def get_data_shape(self) -> Tuple:
        """
        
        return the shape together with the mask
        """
        return self.lidar_spec.stacked_output_shapes(self.n_neighbors_max)
    
    def _update_sphere_stack(self) -> Tuple[np.ndarray, np.ndarray]:
        agent = self.buffer_manager.get_latest_snapshot(self.parent_id)
        if agent is None or agent.sphere is None:
            # Return zeros and mask of all False
            empty_stack, empty_mask = self._empty_stack_and_mask()
            return empty_stack, empty_mask

        neighborhood: List[PerceptionSnapshot] = self.bootstrap()
        sphere_stack: List[np.ndarray] = [agent.sphere]
        mask: List[bool] = [True]

        for neighbor in neighborhood:
            reframed = self.math.neighbor_sphere_from_new_frame(neighbor=neighbor, agent=agent)
            if reframed is not None:
                sphere_stack.append(reframed)
                mask.append(True)

        # Pad with empty spheres if needed
        max_total = 1 + self.n_neighbors_max
        while len(sphere_stack) < max_total:
            sphere_stack.append(self.lidar_spec.empty_sphere())
            mask.append(False)

        # Ensure shape is consistent (N, C, θ, φ)
        stacked = np.stack(sphere_stack, axis=0)
        mask_np = np.array(mask, dtype=bool)

        return stacked, mask_np
    
    def _empty_stack_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        empty = self.lidar_spec.empty_sphere()
        stack = np.stack([empty] * (1 + self.n_neighbors_max), axis=0)
        mask = np.zeros((1 + self.n_neighbors_max,), dtype=bool)
        return stack, mask

