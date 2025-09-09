# load_and_train.py
import torch
from threatsense.level5.level5_dumb_multiobs import Level5DumbMultiObs
from threatsense.level5.level5_fusion_environment import Level5FusionEnvironment
from core.rl_framework.utils.sl_pipeline import SupervisedImitationTrainer

if __name__ == "__main__":
    # === Caminho da pasta de dados (HDF5) ===
    folder_path = "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\output\\collect_and_save\\08.09.2025"

    # === Cria ambiente apenas para obter observation_space ===
    level5_fusion = Level5FusionEnvironment(GUI=False)
    observation_space = level5_fusion.observation_space
    level5_fusion.close()

    # === Inicializa pipeline ===
    trainer = SupervisedImitationTrainer(folder_path)

    # === Treinamento completo (com cross-validation + dataset inteiro) ===

    print("Iniciando treinamento completo...")
    trainer.train_student(
        folder_path=folder_path,
        observation_space=observation_space,
        epochs=1,
        batch_size=128,
        lr=1e-3,
        model_output_dir="09.09.2025_trained_models_all_3",
        limit_size=3_000_000,
    )


    print("✅ Treinamento concluído e modelos salvos.")
