
from typing import Dict, List
from core.entities.quadcopters.components.sensors.lidar import LIDAR
from core.notification_system.message_hub import MessageHub
from core.notification_system.topics_enum import TopicsEnum
from core.entities.quadcopters.components.sensors.interfaces.lidar_interface import BaseLidar, Channels, LiDARBufferManager, CoordinateConverter

import numpy as np
import random
import time


class FusedLiDAR(LIDAR):
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
        debug: bool = False,
    ):
        self.parent_id = parent_id
        self.client_id = client_id
        self.debug = debug
        self.debug_line_id = -1
        self.debug_lines_id = []

        self.n_channels: int = len(Channels)

        self.flag_size = 3

        self.radius = radius
        self.resolution = resolution

        self.THETA_INITIAL_RADIAN = 0
        self.THETA_FINAL_RADIAN = np.pi
        self.THETA_SIZE = self.THETA_FINAL_RADIAN - self.THETA_INITIAL_RADIAN

        self.PHI_INITIAL_RADIAN = -np.pi
        self.PHI_FINAL_RADIAN = np.pi
        self.PHI_SIZE = self.PHI_FINAL_RADIAN - self.PHI_INITIAL_RADIAN

        self.n_theta_points, self.n_phi_points = self._count_points(
            radius, resolution)
        self.sphere: np.ndarray = self._gen_sphere(
            self.n_theta_points, self.n_phi_points, self.n_channels
        )

        self.buffer_manager = LiDARBufferManager(
            current_step=0, max_buffer_size=10)
        self.parent_inertia: Dict = {}

        self.DISTANCE_CHANNEL_IDX: int = Channels.DISTANCE_CHANNEL.value
        self.FLAG_CHANNEL_IDX: int = Channels.FLAG_CHANNEL.value

        self.threshold: float = 1




    # ============================================================================================================================
    # Buffer Management
    # the step associated with the message is considered the current step
    # TODO: Message delivered with a step.
    # ============================================================================================================================

    def buffer_lidar_data(self, message: Dict, publisher_id: int):
        topic: TopicsEnum = TopicsEnum.LIDAR_DATA_BROADCAST
        self._buffer_data(message, publisher_id, topic) 

    def buffer_inertial_data(self, message: Dict, publisher_id: int):
        topic: TopicsEnum = TopicsEnum.INERTIAL_DATA_BROADCAST
        self._buffer_data(message, publisher_id, topic)

    def buffer_step_broadcast(self, message: Dict, publisher_id: int):
        topic: TopicsEnum = TopicsEnum.AGENT_STEP_BROADCAST
        self._buffer_data(message, publisher_id, topic)

    def _buffer_data(self, message: Dict, publisher_id: int, topic: TopicsEnum):
        """
        Método genérico para bufferizar dados de diferentes tópicos.
        """
        if "termination" in message:
            self.buffer_manager.close_buffer(publisher_id, topic)
        else:
            self.buffer_manager.buffer_message(
                message, publisher_id, topic, self.buffer_manager.current_step
            )

    # ============================================================================================================================

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
            position = self.process_message(message)

            self._add_end_position_for_entity(
                position, publisher_type, target_id
            )
        # Return the current state of the sphere
        return self.sphere
    
    def bootstrap(self, n_min:int, n_max: int) -> List[Dict]:
        """
        Bootstrap the LiDAR data from the message.
        This method is called to initialize the LiDAR data with the given message.
        """
        
        n_neighbours = random.randint(n_min, n_max-1) # to let space for the parent
        parent = self.buffer_manager.get_latest_from_publisher(self.parent_id)
        neighbours = self.buffer_manager.get_neighbours(
            n_neighbours, self.parent_id
        )

        #TODO: Format the data to the following structure:
        # step, imu, lidar
        neighbours = [parent] + neighbours
        return neighbours

    def read_data(self) -> Dict:
        return {"fused_lidar": self.get_sphere()}
