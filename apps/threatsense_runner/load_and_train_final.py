# load_and_train.py
import torch
from threatsense.level5.level5_dumb_multiobs import Level5DumbMultiObs
from threatsense.level5.level5_fusion_environment import Level5FusionEnvironment
from core.rl_framework.utils.sl_pipeline import SupervisedImitationTrainer
from stable_baselines3 import PPO
from core.rl_framework.agents.policies.fused_lidar_extractor import FLIAExtractor, SingleWorldViewExtractor

if __name__ == "__main__":
    
    
    
    folder_path = "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\output\\collect_and_save\\08.09.2025"
    output_dir = "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\output\\sl\\12.09.2025"
    trainer = SupervisedImitationTrainer(folder_path)
    level5_fusion = Level5FusionEnvironment(GUI=False)

    policy_kwargs = dict(
        features_extractor_class=SingleWorldViewExtractor,
        features_extractor_kwargs=dict(features_dim=128),
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
        learning_rate=1e-3,
    )


    # === Treinamento completo (com cross-validation + dataset inteiro) ===

    print("Iniciando treinamento completo...")
    trainer.train_loop(
        env=level5_fusion,
        output_dir=output_dir,
        policy_kwargs=policy_kwargs,
        sample_size=1_000_000,
        n_runs=10,
        epochs=1,
        batch_size=128,
        lr=1e-3,
    )

    print("✅ Treinamento concluído e modelos salvos.")
