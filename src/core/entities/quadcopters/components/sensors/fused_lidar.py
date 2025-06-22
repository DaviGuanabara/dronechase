from lidar import LiDAR
from core.notification_system.message_hub import MessageHub
from core.notification_system.topics_enum import Topics_Enum

import numpy as np
import random
import time


class FusedLiDAR(LiDAR):
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

    TODOs:
    [ ] FAZER COISA 1: Implementar compensação por orientação relativa (ajuste rotacional entre drones)
    [ ] FAZER COISA 2: Aplicar pesos baseados em distância e timestamp durante a fusão
    [ ] FAZER COISA 3: Tornar o método de fusão pluggable (ex: média, max, atenção, aprendizado)
    [ ] FAZER COISA 4: Adicionar controle de "decaimento" no buffer (ex: remova se passou muito tempo)
    [ ] FAZER COISA 5: (Opcional) Armazenar logs de quais leituras foram usadas em cada passo
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._should_fuse = False
        self._ally_lidar_buffer = []
        self._max_buffer_size = 30  # limite do buffer
        self._max_time_diff = 0.5   # segundos
        self._max_distance = 10.0   # metros

        self.message_hub = MessageHub()
        self.message_hub.subscribe(
            topic=Topics_Enum.LIDAR_UPDATE_BROADCAST.value,
            subscriber=self._on_ally_updated
        )

    def update_data(self) -> None:
        super().update_data()

        if not self._should_fuse:
            return

        # 🛰️ Publica sua leitura atual no canal
        self.message_hub.publish(
            topic=Topics_Enum.LIDAR_UPDATE_BROADCAST.value,
            message={
                "drone_id": self.owner.id,
                "lidar_data": self.data.copy(),
                "position": self.owner.inertial_data["position"].copy(),
                "timestamp": time.time(),
            },
            publisher_id=self.owner.id,
        )

        # 🔀 Executa bootstrap local com base no buffer
        self._run_bootstrap()

    def _on_ally_updated(self, message: dict, publisher_id: int):
        # ⚠️ Ignora mensagens do próprio drone
        if publisher_id == self.owner.id:
            return

        required_fields = {"lidar_data", "position", "timestamp", "drone_id"}
        if not required_fields.issubset(message.keys()):
            return

        self._ally_lidar_buffer.append(message)

        # Evita buffer crescer demais
        if len(self._ally_lidar_buffer) > self._max_buffer_size:
            self._ally_lidar_buffer.pop(0)

    def _run_bootstrap(self):
        now = time.time()
        pos = self.owner.inertial_data["position"]

        # 📌 Seleciona somente leituras recentes e próximas
        candidates = [
            m for m in self._ally_lidar_buffer
            if now - m["timestamp"] <= self._max_time_diff and
            np.linalg.norm(pos - m["position"]) <= self._max_distance
        ]

        if not candidates:
            return

        k = min(5, len(candidates))
        sampled = random.sample(candidates, k)

        # 🧠 Alinha os dados via delta de posição (simplificado aqui)
        fused_list = []
        for m in sampled:
            offset = pos - m["position"]
            # Aqui você pode compensar usando offsets ou registrar apenas se o deslocamento for pequeno
            fused_list.append(m["lidar_data"])

        # Combina os dados (ex: média simples)
        fused = np.mean(fused_list, axis=0)
        self.data = fused

    def enable_fusion(self):
        self._should_fuse = True
