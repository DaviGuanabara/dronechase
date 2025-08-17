import numpy as np
from core.entities.entity_type import EntityType
from core.dataclasses.perception_snapshot import PerceptionSnapshot
from core.dataclasses.angle_grid import LIDARSpec
from core.entities.quadcopters.components.sensors.components.lidar_math import LidarMath
from core.enums.channel_index import LidarChannels
from core.notification_system.topics_enum import TopicsEnum



def test_transform_features_single_feature():
    # Setup: Lidar spec básico
    lidar_resolution = 16
    spec = LIDARSpec(theta_initial_radian=0, theta_final_radian=np.pi,
                                phi_initial_radian=-np.pi, phi_final_radian=np.pi,
                     resolution=lidar_resolution, n_channels=len(LidarChannels), max_radius=20)
    lidar_math = LidarMath(spec)

    # Feature: alvo em (5, 0, 0) no frame do vizinho
    r = 0.5              # normalizado (5/10)
    theta = np.pi / 2    # ângulo 90° (no plano XY)
    phi = 0.0            # olhando pro eixo +X
    entity_type = EntityType.LOITERINGMUNITION
    relative_step = 0
    publisher_id = 1

    lidar_features = [
        (r, theta, phi, entity_type, relative_step, publisher_id)]
    # Criar snapshot do vizinho
    neighbor = PerceptionSnapshot(
        topics={TopicsEnum.INERTIAL_DATA_BROADCAST: {
            "position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]}, TopicsEnum.LIDAR_DATA_BROADCAST: {
            "features": lidar_features
        }},
        publisher_id=1,
        step=0,
        entity_type=EntityType.LOYALWINGMAN,
        max_delta_step=10
    )

    own = PerceptionSnapshot(
        topics={TopicsEnum.INERTIAL_DATA_BROADCAST: {
            "position": [1.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]}},
        publisher_id=2,
        step=0,
        entity_type=EntityType.LOYALWINGMAN,
        max_delta_step=10
    )



    

    # Executar transformação
    transformed = lidar_math.transform_features(neighbor, own)

    print("Transformed feature:", transformed)

    # --- EXPECTATIVA ---
    # O alvo que estava em (5,0,0) no vizinho deve aparecer no frame do "own"
    # como (4,0,0), já que o own está deslocado +1 no eixo X.
    # Ou seja: a distância deve ser ~0.4 (4/10).
    assert len(transformed) == 1
    new_r, new_theta, new_phi, _, _ = transformed[0]

    assert np.isclose(new_r, 0.45, atol=1e-2)

    assert np.isclose(new_theta, np.pi/2, atol=1e-2)  # ainda no plano XY
    assert np.isclose(new_phi, 0.0, atol=1e-2)        # mesma direção


test_transform_features_single_feature()
