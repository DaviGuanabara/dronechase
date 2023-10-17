import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)

from datetime import datetime
import os

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from stable_baselines3 import PPO

from loyalwingmen.modules.environments.level2_lidar.level2_environment import (
    Level2_lidar,
)


from multiprocessing import cpu_count
from stable_baselines3.common.vec_env import SubprocVecEnv

from loyalwingmen.rl_tools.callback_factory import callbacklist
from loyalwingmen.rl_tools.policies.ppo_policies import LidarInertialActionExtractor
import torch.nn as nn
import torch as th
import math

from stable_baselines3.common.vec_env import VecMonitor

# ===============================================================================
# Setup
# ===============================================================================


def create_output_folder(experiment_name: str):
    # Obter a data atual
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Criar o nome da pasta com base na data atual
    # folder_name = f"output/{experiment_name}"
    folder_name = os.path.join("output", experiment_name)

    # Criar a pasta se ainda não existir
    os.makedirs(folder_name, exist_ok=True)

    # Retornar o caminho completo para a pasta
    return folder_name


def main():
    output_folder = create_output_folder("demo_training2_app")
    model_path = os.path.join(output_folder, "models")
    log_path = os.path.join(output_folder, "logs")

    number_of_logical_cores = cpu_count()
    n_envs = int(number_of_logical_cores / 2)

    env_fns = [lambda: Level2_lidar(GUI=False, rl_frequency=30) for _ in range(n_envs)]

    vectorized_environment = VecMonitor(SubprocVecEnv(env_fns))  # type: ignore

    callback_list = callbacklist(
        vectorized_environment,
        log_path=log_path,
        model_path=model_path,
        save_freq=100_000,
    )

    nn_t = [512, 512, 512]

    policy_kwargs = dict(
        features_extractor_class=LidarInertialActionExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=nn_t, vf=nn_t),
    )

    model = PPO(
        "MultiInputPolicy",
        vectorized_environment,
        verbose=0,
        device="auto",
        policy_kwargs=policy_kwargs,
        learning_rate=1e-5,
    )

    print(model.policy)
    model.learn(total_timesteps=1_000_000, callback=callback_list)
    model.save("trained_level1_ppo")


if __name__ == "__main__":
    # https://stackoverflow.com/questions/29690091/python2-7-exception-the-freeze-support-line-can-be-omitted-if-the-program
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
