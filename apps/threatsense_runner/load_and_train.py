# load_and_train.py
import torch
from threatsense.level5.level5_dumb_multiobs import Level5DumbMultiObs
from threatsense.level5.level5_fusion_environment import Level5FusionEnvironment
from core.rl_framework.utils.sl_pipeline import SupervisedImitationTrainer
from stable_baselines3 import PPO
from core.rl_framework.agents.policies.fused_lidar_extractor import FLIAExtractor

if __name__ == "__main__":
    # === Caminho da pasta de dados (HDF5) ===
    #folder_path = "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\output\\collect_and_save\\08.09.2025"
    folder_path = "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\output\\collect_and_save\\09.09.2025_office"

    # === Cria ambiente apenas para obter observation_space ===
    level5_fusion = Level5FusionEnvironment(GUI=False)
    observation_space = level5_fusion.observation_space
    #level5_fusion.close()

    # === Inicializa pipeline ===
    trainer = SupervisedImitationTrainer(folder_path)

    # modelo

    

    policy_kwargs = dict(
        features_extractor_class=FLIAExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(
            pi=[128, 256, 512],  # cabeça da policy (actor)
            vf=[128, 256, 512]   # cabeça do value (critic)
        )
    )

    model = PPO(
        "MultiInputPolicy",
        level5_fusion,  # seu Level5Fusion ou outro env compatível
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
    )


    # === Treinamento completo (com cross-validation + dataset inteiro) ===

    print("Iniciando treinamento completo...")
    trainer.train_student_ppo_wrapped(
        folder_path=folder_path,
        model=model,
        action_dim=4,
        epochs=1,
        batch_size=128,
        lr=1e-3,
        model_output_dir="09.09.2025_trained_models_all_3",
        limit_size=900_000,
    )


    print("✅ Treinamento concluído e modelos salvos.")
