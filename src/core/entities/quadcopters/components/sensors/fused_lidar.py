
from typing import List, Tuple, Union
from typing import Dict, List, Tuple
from core.dataclasses.perception_snapshot import PerceptionSnapshot
from core.notification_system.topics_enum import TopicsEnum
from core.entities.quadcopters.components.sensors.interfaces.base_lidar import BaseLidar



import numpy as np
import random


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
        activate_fusion: bool = False,
    ):
        super().__init__(parent_id, client_id, debug, radius, resolution, n_neighbors_min, n_neighbors_max)
        print("Fused Lidar created")
        self.activate_fusion: bool = False


    def bootstrap(self) -> List[PerceptionSnapshot]:

        n = random.choice(range(self.n_neighbors_min, self.n_neighbors_max))
        return self.buffer_manager.get_random_neighborhood(
            n_neighbors=n, exclude_publisher_id=self.parent_id
        )

    def _build_valid_spheres(self) -> List[np.ndarray]:
        """
        first is agent sphere, then neighbors spheres.
        Returns a list of spheres, where the first element is the agent's sphere
        shape is (N, C, θ, φ) where N = 1 + max_neighbors
        and C, θ, φ are defined by the LIDARSpec.
        """

        agent = self.buffer_manager.get_latest_snapshot(self.parent_id)

        if agent is None or agent.sphere is None:
            return []

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

        if agent is None or agent.position is None:
            return

        features = []
        for snapshot in snapshots:
            if snapshot.position is None:
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
        self.buffer_manager.buffer_message_directly({"lidar": self.sphere, "step": self.buffer_manager.current_step}, self.parent_id, TopicsEnum.LIDAR_DATA_BROADCAST)



    def read_data(self) -> Dict:
        if not self.activate_fusion:
            return {"lidar": self.sphere}
        
        self.sphere_stack = self._build_valid_spheres()
        padded_stack, mask = self._pad_sphere_stack(self.sphere_stack)
        #TODO: REMOVE THE PRINT
        print({"lidar": self.sphere, "fused_lidar": padded_stack, "mask": mask})
        return {"lidar": self.sphere, "fused_lidar": padded_stack, "mask": mask}

    def reset(self):
        pass


    def get_data_shape(self) -> Union[Dict[str, Tuple[int, ...]], Tuple[int, int, int]]:
        """
        
        return the shape together with the mask
        """

        #TODO Shape from fused lidar got 1 additional channel then LIDAR class.
        return self.lidar_spec.shape
    
    def get_stacked_lidar_shape(self):
        #TODO: How to handle the mask?
        # The shape is (N, C, θ, φ) where N = 1 + max_neighbors
        # and C, θ, φ are defined by the LIDARSpec.
        return self.lidar_spec.get_stacked_lidar_shape(self.n_neighbors_max)
    
    def get_validity_mask_shape(self):
        #TODO: How to handle the mask?
        # The shape is (N, C, θ, φ) where N = 1 + max_neighbors
        # and C, θ, φ are defined by the LIDARSpec.
        return self.lidar_spec.get_validity_mask_shape(self.n_neighbors_max)


    def _pad_sphere_stack(
        self,
        valid_spheres: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pads a list of valid spheres up to self.n_neighbors_max and returns the stack with a validity mask.

        Returns:
            sphere_stack: np.ndarray of shape (max_neighbors, C, θ, φ)
            validity_mask: np.ndarray of shape (max_neighbors,) with boolean values
        """
        
        max_spheres = self.n_neighbors_max + 1
        n_valid = len(valid_spheres)

        if n_valid > max_spheres:
            raise ValueError(
                f"Received {n_valid} spheres but max is {max_spheres}")

        # Create the full mask
        validity_mask = np.zeros(max_spheres, dtype=bool)
        validity_mask[:n_valid] = True

        # Pad with "invalid spheres"
        invalid_sphere = self.lidar_spec.empty_sphere()  # assume all ones = no detection
        padded_spheres = valid_spheres + \
            [invalid_sphere.copy() for _ in range(max_spheres - n_valid)]

        # Stack into single array (N, C, θ, φ)
        sphere_stack_padded = np.stack(padded_spheres, axis=0)

        return sphere_stack_padded, validity_mask

    def enable_fusion(self):
        self.activate_fusion = True
        print("Fused lidar activated")
