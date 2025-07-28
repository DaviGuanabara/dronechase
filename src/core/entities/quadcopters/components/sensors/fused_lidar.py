
from typing import Dict, List, Optional
from core.dataclasses.perception_snapshot import PerceptionSnapshot
from core.entities.entity_type import EntityType
from core.entities.quadcopters.components.sensors.lidar import LIDAR
from core.notification_system.message_hub import MessageHub
from core.notification_system.topics_enum import TopicsEnum
from core.entities.quadcopters.components.sensors.interfaces.lidar_interface import BaseLidar, CoordinateConverter
from core.entities.quadcopters.components.sensors.components.lidar_buffer import LiDARBufferManager
from core.entities.quadcopters.components.sensors.components.lidar_geometry import cartesian_to_spherical, neighbor_sphere_from_new_frame, normalize_spherical, reframe

import numpy as np
import random
import time

from core.dataclasses.normalization import NormalizationConfig
from core.dataclasses.angle_grid import LIDARSpec
from core.enums.channel_index import LidarChannels
from core.dataclasses.perception_keys import PerceptionKeys


class FusedLiDAR(BaseLidar):
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
            neighbor_sphere_reframed = neighbor_sphere_from_new_frame(
                neighbor=neighbor,
                agent=agent,
                lidar_spec=self.lidar_spec
            )
            if neighbor_sphere_reframed is not None:
                sphere_stack.append(neighbor_sphere_reframed)

        self.sphere_stack = sphere_stack

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

    def get_agent(self) -> Optional[PerceptionSnapshot]:
        return self.buffer_manager.get_latest_snapshot(self.parent_id)
    
    def add_features(self, sphere, features):
        """
        
        r, theta, phi, entity type, delta_step = features
        """

        for feature in features:
            r = feature[0]
            theta = feature[1]
            phi = feature[2]
            entity_type = feature[3]
            delta_step = feature[4]

            sphere[LidarChannels.distance][theta][phi] = r
            sphere[LidarChannels.flag][theta][phi] = entity_type
            sphere[LidarChannels.time][theta][phi] = delta_step

        return sphere


    
    def _update_own_sphere(self):
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
            reframed_snapshot_position = reframe(snapshot.position, np.zeros(3), agent.position)
            spherical = cartesian_to_spherical(reframed_snapshot_position)
            spherical = normalize_spherical(spherical, lidar_spec=self.lidar_spec)
            # the lidars data are always
            # from the loyal wingman.
           
            delta_step = 0 #current delta step
            features.append((*spherical, EntityType.LOYALWINGMAN, delta_step)) # mount features
        
        new_sphere = self.lidar_spec.empty_sphere()
        self.sphere = self.add_features(new_sphere, features)



    def read_data(self) -> Dict:
        return {"lidar": self.sphere, "fused_lidar": self.sphere_stack}
