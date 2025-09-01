import torch
import torch.nn as nn
from pathlib import Path


# Gerenciadores wildfire
from wildfire_pyro.wrappers.distillation_learning_manager import DistillationLearningManager  # type: ignore
from wildfire_pyro.wrappers.supervised_learning_manager import SupervisedLearningManager # type: ignore
from wildfire_pyro.common.seed_manager import configure_seed_manager, get_seed # type: ignore

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# Ambiente corrigido (sem canal temporal)
from threatsense.level5.level5_fusion_environment import Level5FusionEnvironment




# Seu extractor novo
from core.rl_framework.agents.policies.fused_lidar_extractor import FLIAExtractor
# --- antes do PPO.load(...), coloque este bloco de compat ---
import sys, importlib, logging

def alias(old: str, new: str):
    try:
        sys.modules[old] = importlib.import_module(new)
        logging.info(f"[compat] {old} -> {new}")
    except Exception as e:
        logging.warning(f"[compat] Falhou alias {old}->{new}: {e}")


# just need to set it once, for reproducibility
global_seed = 42
configure_seed_manager(global_seed)

# === Caminho do modelo professor (Exp03) ===
BASE_MODEL_PATH = Path(
    r"C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\experiment_results\\threat_engage\\Etapa 03 - Level 4\\EXP03\\Treinamento - 1RL 1BT - EXP03\\31_01_2024_level4_2.00M_exp03_vFinal\\Trial_8\\models_dir\\t8_PPO_r6995.40.zip"
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Carrega o teacher (PPO antigo)
    alias('loyalwingmen', 'core')  # mantenha se o zip referenciar esse prefixo
    teacher_model: PPO = PPO.load(str(BASE_MODEL_PATH), device=device)
    teacher_model.policy.eval()  # garante modo inferência

    # 2) Instancia ambiente corrigido
    env = Level5FusionEnvironment(GUI=False, rl_frequency=15)

    # 3) Cria student network (ex: FLIAExtractor + MLP)
    obs_space = env.observation_space
    student_model = FLIAExtractor(observation_space=obs_space, features_dim=512)

    # 4) Parâmetros
    runtime_parameters = {"device": device, "verbose": 1, "seed": 42}
    logging_parameters = {
        "log_path": "./output/distillation_logs",
        "format_strings": ["stdout", "csv", "tensorboard"],
    }
    model_parameters = {
        "batch_size": 128,
        "rollout_size": 128,
        "lr": 1e-4,
    }

    # 5) Manager de destilação
    manager = DistillationLearningManager(
        student=student_model,
        environment=env,
        logging_parameters=logging_parameters,
        runtime_parameters=runtime_parameters,
        model_parameters=model_parameters,
        teacher_nn=teacher_model.policy,   # usa a policy do PPO como professor
    )

    # 6) Treino supervisionado (destilação)
    manager.learn(total_timesteps=200_000, progress_bar=True)

    # 7) Salva student
    manager.save("./output/distilled_student.zip")


if __name__ == "__main__":
    main()
