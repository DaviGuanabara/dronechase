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
from stable_baselines3 import PPO

from threatengage.modules.environments_pyflyt.level3.pyflyt_level3_environment import (
    PyflytL3Enviroment as Level3,
)
from threatengage.rl_tools.policies.ppo_policies import (
    LidarInertialActionExtractor,
    LidarInertialActionExtractor2,
)

from multiprocessing import cpu_count
from stable_baselines3.common.vec_env import SubprocVecEnv

from threatengage.rl_tools.callback_factory import callbacklist
import torch.nn as nn
import torch as th
import math

from stable_baselines3.common.vec_env import VecMonitor
from threatengage.rl_tools.directory_manager import DirectoryManager
import torch

# ===============================================================================
# Setup
# ===============================================================================


def directories(study_name: str):
    app_name, _ = os.path.splitext(os.path.basename(__file__))
    output_folder = os.path.join("output", app_name, study_name)
    DirectoryManager.create_directory(output_folder)

    models_dir = os.path.join(output_folder, "models_dir")
    logs_dir = os.path.join(output_folder, "logs_dir")

    return models_dir, logs_dir, output_folder


def main():
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study_name = "01.10.2023"  # datetime.date.today()
    models_dir, logs_dir, output_folder = directories(study_name)

    number_of_logical_cores = cpu_count()
    n_envs = int(number_of_logical_cores)

    env_fns = [lambda: Level3(GUI=False, rl_frequency=15) for _ in range(n_envs)]

    vectorized_environment = VecMonitor(SubprocVecEnv(env_fns))  # type: ignore

    callback_list = callbacklist(
        vectorized_environment,
        log_path=logs_dir,
        model_path=models_dir,
        save_freq=100_000,
    )

    nn_t = [512, 128, 256]

    policy_kwargs = dict(
        features_extractor_class=LidarInertialActionExtractor2,
        features_extractor_kwargs=dict(features_dim=512, device=device),
        net_arch=dict(pi=nn_t, vf=nn_t),
    )

    model = PPO(
        "MultiInputPolicy",
        vectorized_environment,
        verbose=0,
        device=device,
        policy_kwargs=policy_kwargs,
        learning_rate=0.00001,
        batch_size=512,
    )

    print(model.policy)
    model.learn(total_timesteps=500_000, callback=callback_list)
    model.save("ppo_level3_lidar")


if __name__ == "__main__":
    # https://stackoverflow.com/questions/29690091/python2-7-exception-the-freeze-support-line-can-be-omitted-if-the-program
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
