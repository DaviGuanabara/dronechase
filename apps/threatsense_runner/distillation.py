from typing import cast
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# Gerenciadores wildfire
from wildfire_pyro.wrappers.distillation_learning_manager import DistillationLearningManager  # type: ignore
from wildfire_pyro.wrappers.supervised_learning_manager import SupervisedLearningManager  # type: ignore
from wildfire_pyro.common.seed_manager import configure_seed_manager, get_seed  # type: ignore

from core.rl_framework.utils.pipeline import (
    ReinforcementLearningPipeline,
    CallbackType,
)
from stable_baselines3.common.vec_env import VecMonitor

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# Ambiente corrigido (sem canal temporal)
from threatsense.level5.level5_fusion_environment import Level5FusionEnvironment
import gymnasium as gym  



# Seu extractor novo
from core.rl_framework.agents.policies.fused_lidar_extractor import FLIAExtractor
# --- antes do PPO.load(...), coloque este bloco de compat ---
import sys, importlib, logging

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv


class CustomDummyVecEnv(DummyVecEnv):

    def __init__(self, env_fns):
        super().__init__(env_fns)
        # Assume que todos os envs têm os mesmos atributos extras
        base_env = env_fns[0]()
        if hasattr(base_env, "teacher_observation_space"):
            self.teacher_observation_space = base_env.teacher_observation_space  # type: ignore
        if hasattr(base_env, "student_observation_space"):
            self.student_observation_space = base_env.student_observation_space  # type: ignore
        base_env.close()  # evita vazamento

    def reset(self):
        obs, info = super().reset()
        # garante que cada chave tem batch na frente
        if isinstance(obs, dict):
            obs = {k: np.array(v) for k, v in obs.items()}
        else:
            obs = np.array(obs)
        return obs, info

    def step(self, actions):
        obs, rewards, dones, infos = super().step(actions)
        if isinstance(obs, dict):
            obs = {k: np.array(v) for k, v in obs.items()}
        else:
            obs = np.array(obs)
        return obs, rewards, dones, [False]*len(dones), infos



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
    #"C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\experiment_results\\threat_engage\\Etapa 03 - Level 4\\EXP03\\Treinamento - 1RL 1BT - EXP03\\31_01_2024_level4_2.00M_exp03_vFinal\\Trial_8\\models_dir\\t8_PPO_r6995.40.zip"
    "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\experiment_results\\threat_engage\\Etapa 03 - Level 4\\EXP03\\Treinamento - 1RL 1BT - EXP03\\31_01_2024_level4_2.00M_exp03_vFinal\\Trial_8\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t8_PPO_r6995.40.zip"
)

def load_teacher_model(device: torch.device) -> torch.nn.Module:
    return PPO.load(str(BASE_MODEL_PATH), device=device).policy


def load_student_model(device: torch.device, environment) -> torch.nn.Module:
    policy_kwargs = dict(
        features_extractor_class=FLIAExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[128, 256, 512], vf=[128, 256, 512]),
    )
    #eu acredito que esse seja o net_arch correto, que deveria bater com o teacher
    
    model = PPO(
        "MultiInputPolicy",
        environment,
        verbose=0,
        device="cuda",
        policy_kwargs=policy_kwargs,
    )


    return model.policy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Uso
    #env_fns = [lambda: cast(gym.Env, Level5FusionEnvironment(GUI=False, rl_frequency=15))]
    #level5 = CustomDummyVecEnv(env_fns)
    level5 = Level5FusionEnvironment(GUI=False, rl_frequency=15)


    
    alias('loyalwingmen', 'core')  # mantenha se o zip referenciar esse prefixo
    teacher_policy: torch.nn.Module = load_teacher_model(device)
    student_policy: torch.nn.Module = load_student_model(device, level5)



    runtime_parameters = {"device": device, "verbose": 1, "seed": 42}
    logging_parameters = {
        "log_path": "./output/distillation_logs",
        "format_strings": ["stdout", "csv", "tensorboard"],
    }
    model_parameters = {
        "batch_size": 1024,
        "rollout_size": 2048,
        "lr": 1e-3,
    }

    # 5) Manager de destilação
    manager = DistillationLearningManager(
        student_nn=student_policy,
        teacher_nn=teacher_policy, 
        environment=level5,
        logging_parameters=logging_parameters,
        runtime_parameters=runtime_parameters,
        model_parameters=model_parameters,
        
    )

    # 6) Treino supervisionado (destilação)
    manager.learn(total_timesteps=2_000_000, progress_bar=True)

    # 7) Salva student
    manager.save("./output/distilled_student.zip")


if __name__ == "__main__":
    main()
