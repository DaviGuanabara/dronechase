# load_and_train.py
from threatsense.level5.level5_c2_fusion_environment import Level5C2FusionEnvironment as Level5C2Fusion
from core.rl_framework.utils.sl_pipeline import SupervisedImitationTrainer
from core.rl_framework.agents.policies.fused_lidar_extractor import SingleWorldViewExtractor
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if __name__ == "__main__":

    "O código está péssimo. Preciso fazer uma refatoração depois"

    folder_path = "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/collect_and_save/29.04.2026"
    output_dir = "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/sl/30.04.2026/train_sl_1"

    trainer = SupervisedImitationTrainer(folder_path)
    level5_fusion = Level5C2Fusion(GUI=False)

    policy_kwargs = dict(
        features_extractor_class=SingleWorldViewExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(
            pi=[128, 256, 512],  # cabeça da policy (actor)
            vf=[128, 256, 512]   # cabeça do value (critic)
        )
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
