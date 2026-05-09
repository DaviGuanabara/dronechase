
import itertools
from typing import Dict, List, Tuple, Union
from core.dataclasses.perception_snapshot import PerceptionSnapshot
from core.enums.channel_index import LidarChannels
from core.notification_system.topics_enum import TopicsEnum
from core.entities.quadcopters.components.sensors.interfaces.base_lidar import BaseLidar
import pybullet as p


import numpy as np


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
        # TODO: DEBUG AND VERBOSE -> SAME
        super().__init__(parent_id, client_id, debug, radius,
                         resolution, n_neighbors_min, n_neighbors_max)
        # print("Fused Lidar created")
        self.activate_fusion: bool = activate_fusion
        self.features = []
        self.sphere_stack: List[np.ndarray] = []
        self.padded_stack = np.empty(
            self.get_stacked_lidar_shape(), dtype=np.float32)
        self.mask = np.zeros(self.get_validity_mask_shape(), dtype=bool)
        self._zero_vector = np.zeros(3, dtype=np.float32)

    def bootstrap(self) -> List[PerceptionSnapshot]:
        """
        Gather randomly outdated snapshots from multiple sources (own drone included).
        """
        n = np.random.randint(self.n_neighbors_min, self.n_neighbors_max)
        neighborhood = self.buffer_manager.get_random_neighborhood(
            n_neighbors=n)
        # if len(neighborhood) > 0:
        #     print(f"[DEBUG] neighborhood: {neighborhood[0].publisher_id} - {neighborhood[0].step} - {neighborhood[0].position}")
        return neighborhood

    def _build_valid_spheres(self) -> List[np.ndarray]:
        """
        first is returned its own sphere, then neighbors spheres.
        Returns a list of spheres, where the first element is its own sphere
        shape is (N, C, θ, φ) where N = 1 + max_neighbors
        and C, θ, φ are defined by the LIDARSpec.
        """

        own_drone = self.buffer_manager.get_latest_snapshot(self.parent_id)

        if own_drone is None or own_drone.sphere is None:
            # print(f"[DEBUG] [{self.parent_id}] own_drone is None ? {own_drone is None};")
            # print(own_drone)
            return []

        neighborhood: List[PerceptionSnapshot] = self.bootstrap()
        # print(f"[DEBUG] neighborhood len: {len(neighborhood)}")
        # Buffer only keeps agents sphere
        sphere_stack: List[np.ndarray] = [own_drone.sphere]

        for neighbor in neighborhood:

            neighbor_sphere_reframed = self.math.neighbor_sphere_from_new_frame(
                neighbor=neighbor, own=own_drone)

            if neighbor_sphere_reframed is not None:
                sphere_stack.append(neighbor_sphere_reframed)

        return sphere_stack

    def get_snapshots_by_distance(self) -> List[PerceptionSnapshot]:
        """
        Return entities detected within the max LiDAR radius.
        """

        own_drone_snapshot = self.get_own_snapshot()

        if own_drone_snapshot is None or own_drone_snapshot.position is None:
            return []

        # TODO: Here gets only allies, not all entities.
        # This is a problem, because I want to get all entities.

        latests: List[PerceptionSnapshot] = self.buffer_manager.get_latests(
            self.parent_id)
        # latests: List[PerceptionSnapshot] = self.buffer_manager.get_latest_neighborhood(
        #   exclude_publisher_id=self.parent_id)

        # TODO: precisa mesmo adicionar o agente?
        # latests.append(agent_snapshot)

        # print(f"[DEBUG] FusedLIDAR: Agent snapshot: {agent_snapshot}")

        # agent_pos = agent_snapshot.position
        return latests
        # return [
        #    latest for latest in latests
        #    if latest.position is not None and np.linalg.norm(latest.position - agent_snapshot.position) <= self.lidar_spec.max_radius
        # ]

    def update_data(self):
        """
        Synthesizes the LiDAR sphere of its own_drone by converting known positions
        (from own_drone and allies) into spherical features and discretizing them.

        This is a simulated LiDAR. Instead of casting rays, we directly project
        known entity positions into the own_drone’s local frame and update the LiDAR grid.

        The result is saved to:
            - self.sphere: Discretized LiDAR sphere (C, θ, φ)
            - self.features: List of [r_norm, θ, φ, entity_type, delta_step]

        It is important to note that the delta_step is calculated based on the current
        time step. As the simulation progresses, the delta_step inserted in the feature
        will become outdated. So, should we be caution about it. 
        TODO: Should I remove the delta_step from the features?
        """
        own_drone = self.get_own_snapshot()
        # print(f"[FusedLIDAR {self.parent_id}] Starting synthetic LiDAR update.")

        if own_drone is None or own_drone.position is None or own_drone.quaternion is None:
            # print(f"[FusedLIDAR {self.parent_id}] own_drone snapshot incomplete. Skipping update.")
            self.features = []
            return

        snapshots: List[PerceptionSnapshot] = self.buffer_manager.get_latests(
            self.parent_id)

        raw_features = []

        for snap in snapshots:
            if snap.position is None or snap.quaternion is None:
                continue

            # Skip synthetic echo of self from neighbors
            # self visualization
            if snap.publisher_id == self.parent_id:
                # print("READING OF ITSELF REMOVED")
                continue

            # Convert entity's position into agent’s local frame
            relative_vector = self.math.reframe(
                # reference point is the entity origin
                local_vector=self._zero_vector,
                neighbor_position=snap.position,
                neighbor_quaternion=snap.quaternion,
                own_position=own_drone.position,
                own_quaternion=own_drone.quaternion,
            )

            if relative_vector is None:
                continue
            spherical = self.math.cartesian_to_spherical(relative_vector)
            normalized = self.math.normalize_spherical(spherical)

            # TODO: This delta step logic is broken, because once the time passes, the step difference increases
            # and the data becomes useless.
            delta_step = snap.normalized_delta
            feature = (*normalized, snap.entity_type,
                       delta_step, snap.publisher_id)
            raw_features.append(feature)

        # Fill the sphere with the synthetic features
        self.sphere, self.features = self.math.add_features(
            self.lidar_spec.empty_sphere(), raw_features)

        # It saves in 1 step behind, because is the step where it were collected the IMU data to build the sphere
        # TODO: refactor this call to BaseLidar
        self.buffer_manager.buffer_message_directly(
            {
                "sphere": self.sphere,  # aqui ele passa OK
                "features": self.features,
            },
            self.parent_id,
            TopicsEnum.LIDAR_DATA_BROADCAST,
        )

    def read_data(self) -> Dict:
        # print(self.buffer_manager.print_buffer_status())

        if self.debug:
            self.render_lidar_debug_rays()

        # TODO: FOR MULTIOBS, I HAVE TO RETURN THE COMPLETE OBSERVATION OF FUSED LIDAR
        # ABRIR AS PORTAS AQUI
        if not self.activate_fusion:
            return {"sphere": self.sphere, "features": self.features}

        self.sphere_stack = self._build_valid_spheres()
        self._pad_sphere_stack(self.sphere_stack)

        # if self.debug:
        # self.render_random_stacked_lidar_debug_rays()

        # print(f"time {self.padded_stack[1][LidarChannels.time.value]}")
        self.randomize_stack(self.padded_stack, self.mask)

        # print(f"[DEBUG] FusedLIDAR {self.parent_id} read_data After Pad and Random -> stack shape: {self.padded_stack.shape}, mask shape: {self.mask.shape}")
        return {"sphere": self.sphere, "features": self.features, "stacked_spheres": self.padded_stack, "validity_mask": self.mask}

    def randomize_stack(self, padded_stack: np.ndarray, validity_mask: np.ndarray) -> None:
        """
        Randomizes the order of the spheres in the stack while preserving the validity mask.
        Operates in-place to avoid Python list conversion and array reconstruction.
        """
        order = np.random.permutation(validity_mask.shape[0])
        padded_stack[:] = padded_stack[order]
        validity_mask[:] = validity_mask[order]

    def reset(self):
        pass

    def get_data_shape(self) -> Union[Dict[str, Tuple[int, ...]], Tuple[int, int, int]]:
        """

        return the shape together with the mask
        """

        # TODO Shape from fused lidar got 1 additional channel than LIDAR class.
        return self.lidar_spec.shape

    def get_stacked_lidar_shape(self):
        # TODO: How to handle the mask?
        # The shape is (N, C, θ, φ) where N = 1 + max_neighbors
        # and C, θ, φ are defined by the LIDARSpec.

        # print(f"[DEBUG] get_stacked_lidar_shape: {self.lidar_spec.get_stacked_lidar_shape(self.n_neighbors_max)}")
        return self.lidar_spec.get_stacked_lidar_shape(self.n_neighbors_max)

    def get_validity_mask_shape(self):
        # TODO: How to handle the mask?
        # The shape is (N, C, θ, φ) where N = 1 + max_neighbors
        # and C, θ, φ are defined by the LIDARSpec.
        # print(f"[DEBUG] get_validity_mask_shape: {self.lidar_spec.get_validity_mask_shape(self.n_neighbors_max)}")
        return self.lidar_spec.get_validity_mask_shape(self.n_neighbors_max)

    def _pad_sphere_stack(
        self,
        valid_spheres: List[np.ndarray],
    ) -> None:
        """
        Pads a list of valid spheres up to self.n_neighbors_max.
        Reuses self.padded_stack and self.mask to avoid per-step allocations.
        """
        max_spheres = self.n_neighbors_max + 1
        n_valid = len(valid_spheres)

        if n_valid > max_spheres:
            raise ValueError(
                f"Received {n_valid} spheres but max is {max_spheres}")

        self.mask.fill(False)
        self.mask[:n_valid] = True

        self.padded_stack.fill(1.0)
        for i, sphere in enumerate(valid_spheres):
            self.padded_stack[i] = sphere

    def enable_fusion(self):
        self.activate_fusion = True
        # print("Fused lidar activated")

    def render_random_stacked_lidar_debug_rays(self):
        """
        Render LiDAR rays from a random stacked sphere (already reframed),
        drawn from the own_drone's position using a unique color.
        """
        if not hasattr(self, 'padded_stack') or self.sphere_stack is None:

            # print("[DEBUG] No padded_stack available.")
            return

        if len(self.sphere_stack) < 2:
            # print("[DEBUG] Empty padded_stack.")
            return

        # Select a random index
        target_idx = 0  # random.randint(0, len(self.sphere_stack) - 1)

        own_drone = self.get_own_snapshot()
        if own_drone is None or own_drone.position is None:
            # print("[DEBUG] own_drone snapshot or position unavailable.")
            return

        position = own_drone.position
        theta_count = self.lidar_spec.n_theta_points
        phi_count = self.lidar_spec.n_phi_points
        max_distance = self.lidar_spec.max_radius
        Green = [0, 1, 0, 1]

        colors = [
            [0, 1, 0],  # green
            [1, 0, 0],  # red
            [0, 0, 1],  # blue
            [0, 1, 1],  # cyan
            [1, 1, 0],  # yellow
            [1, 0, 1],  # magenta

        ]

        if own_drone.sphere is not None:
            are_equal = np.array_equal(own_drone.sphere, self.sphere_stack[0])
            # print(f"[DEBUG] self.sphere == sphere_stack[0]? {are_equal}")

        color = colors[target_idx % len(colors)]
        # print(f"[DEBUG] Random sphere idx={target_idx}, color={color}")

        for theta_idx, phi_idx in itertools.product(range(theta_count), range(phi_count)):
            norm_dist = self.sphere_stack[1][LidarChannels.distance.value][theta_idx][phi_idx]
            if norm_dist >= 1.0:
                # REMOVE INVALID POSITIONS
                continue

            dist = norm_dist * max_distance
            direction = self.math.spherical_to_cartesian(
                np.array([
                    dist,
                    self.math.theta_radian_from_index(theta_idx),
                    self.math.phi_radian_from_index(phi_idx)
                ], dtype=np.float32)
            )

            end_position = own_drone.position + direction

            p.addUserDebugLine(
                lineFromXYZ=position,
                lineToXYZ=end_position,
                lineColorRGB=color,
                lineWidth=2,
                lifeTime=1,
                physicsClientId=self.client_id
            )

    def render_lidar_debug_rays(self):
        """
        Render LiDAR debug rays based on a randomly selected sphere from the stacked buffer
        (already transformed into the agent’s reference frame). Rays are drawn from the
        agent’s current position using a distinct color.

        Note:
            The stacked buffer may include historical data from both neighbors and the
            agent itself. As a result, the rendered rays can depict temporal variability,
            i.e., the same spatial direction may correspond to different entities across
            time steps.
        """
        # print(f"[VISUALIZATION] Showing fused rays for loyalwingman {self.parent_id}")
        # if self.parent_id == 12:
        #    return
        # Select a random index
        target_idx = self.parent_id % 10

        own_drone = self.get_own_snapshot()
        if own_drone is None or own_drone.position is None:
            # print("[DEBUG] Own drone snapshot or position unavailable.")
            return

        position = own_drone.position
        theta_count = self.lidar_spec.n_theta_points
        phi_count = self.lidar_spec.n_phi_points
        max_distance = self.lidar_spec.max_radius
        Green = [0, 1, 0, 1]

        colors = [
            [1, 1, 0],  # yellow
            [0, 1, 0],  # green
            [0, 0, 1],  # blue


            [0, 1, 1],  # cyan

            [1, 0, 1],  # magenta
            [1, 0, 0],  # red
        ]

        sphere = self.sphere
        if sphere is None:
            return

        color = colors[target_idx % len(colors)]
        # print(f"Color choice: {color}")
        # print(f"[DEBUG] Random sphere idx={target_idx}, color={color}")

        for theta_idx, phi_idx in itertools.product(range(theta_count), range(phi_count)):
            norm_dist = sphere[LidarChannels.distance.value][theta_idx][phi_idx]
            if norm_dist >= 1.0:
                continue

            dist = norm_dist * max_distance
            direction = self.math.spherical_to_cartesian(
                np.array([
                    dist,
                    self.math.theta_radian_from_index(theta_idx),
                    self.math.phi_radian_from_index(phi_idx)
                ], dtype=np.float32)
            )

            end_position = own_drone.position + direction

            p.addUserDebugLine(
                lineFromXYZ=position,
                lineToXYZ=end_position,
                lineColorRGB=color,
                lineWidth=2,
                lifeTime=1,
                physicsClientId=self.client_id
            )
