
from typing import Dict, List
from core.entities.quadcopters.components.sensors.lidar import LIDAR
from core.notification_system.message_hub import MessageHub
from core.notification_system.topics_enum import TopicsEnum
from core.entities.quadcopters.components.sensors.interfaces.lidar_interface import BaseLidar, Channels, CoordinateConverter
from core.entities.quadcopters.components.sensors.components.lidar_buffer import LiDARBufferManager

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
        lidar.enable_fusion()  # ativa a lógica de fusão

        
    O lidar pega a posição de todos os drones. 
    Ele recebe essas posições através da classe quadcopter, que por sua vez, subscreve no canal de mensagens
    e recebe as atualizações do IMU. No caso dos dados do LiDAR, o FusedLiDAR recebe, adicionalmente a saída anterior.

    Mas no meu caso, preciso também receber a leitura do LiDAR de todos os drones aliados.

    TODOs:
    [ ] FAZER COISA 1: Implementar compensação por orientação relativa (ajuste rotacional entre drones)
    [ ] FAZER COISA 2: Aplicar pesos baseados em distância e timestamp durante a fusão
    [ ] FAZER COISA 3: Tornar o método de fusão pluggable (ex: média, max, atenção, aprendizado)
    [ ] FAZER COISA 4: Adicionar controle de "decaimento" no buffer (ex: remova se passou muito tempo)
    [ ] FAZER COISA 5: (Opcional) Armazenar logs de quais leituras foram usadas em cada passo
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
    


    # ============================================================================================================================


    def bootstrap(self):

        n = random.choice(range(self.n_neighbors_min, self.n_neighbors_max))
        neighborhood = self.buffer_manager.get_neighborhood(n_neighbors=n, exclude_publisher_id=self.parent_id)





    










































    def update_data(self):

        """
        It is like it is reading the environment.
        it returns the resulting matrix of the lidar, to be broadcasted.
        As it will be broadcasted, it will be saved in the buffer, through 'buffer_lidar_data'.
        """
        buffer_data = self.buffer_manager.get_latest_from_topic(
            TopicsEnum.INERTIAL_DATA_BROADCAST)
        self.reset()

        for target_id, message in buffer_data.items():
            publisher_type = message.get("publisher_type")
            if publisher_type is None:
                continue  # ignore misformed or incomplete messages

            # extremity_points not working yet
            position = message.get("position")

            self._add_end_position_for_entity(
                position, publisher_type, target_id
            )
        # Return the current state of the sphere
        return self.sphere
    
    

    
    def bootstrap_deltas(self, n_min:int, n_max: int):
        """
        LIDAR:
        {"lidar": self.sphere}

        IMU:
        {
            "position": position,
            "attitude": attitude,
            "quaternion": quaternion,
            "velocity": velocity,
            "angular_rate": angular_rate,
        }

        self.position: np.ndarray = np.zeros(3)
        self.quaternions: np.ndarray = np.zeros(4)
        self.attitude: np.ndarray = np.zeros(3)

        self.velocity: np.ndarray = np.zeros(3)
        self.angular_rate: np.ndarray = np.zeros(3)

        def update_data(self):
        POSITION = 3
        EULER = 1
        VELOCITY = 2
        ANGULAR_RATE = 0
        
        self.quadx.update_state()
        state = self.quadx.state

        self.position = np.array(state[POSITION])
        self.attitude = np.array(state[EULER])
        self.quaternions = np.array(p.getQuaternionFromEuler(state[EULER]))
        
        self.velocity = np.array(state[VELOCITY])
        self.angular_rate = np.array(state[ANGULAR_RATE])


        I am going to use only velocity and position from IMU to generate the deltas.
        """

        neighbours = self.bootstrap(n_min, n_max)

        for neighbour in neighbours:
            IMU = TopicsEnum.INERTIAL_DATA_BROADCAST.name
            LIDAR = TopicsEnum.LIDAR_DATA_BROADCAST.name
            STEP = "step"

            neighbour[IMU]




    def read_data(self) -> Dict:
        return {"fused_lidar": self.get_sphere()}
