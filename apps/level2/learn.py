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

from loyalwingmen.modules.environments_pyflyt.level2.pyflyt_level2_environment import (
    PyflytL2Enviroment as Level2,
)
from loyalwingmen.rl_tools.policies.ppo_policies import LidarInertialActionExtractor

from multiprocessing import cpu_count
from stable_baselines3.common.vec_env import SubprocVecEnv

from loyalwingmen.rl_tools.callback_factory import callbacklist
import torch.nn as nn
import torch as th
import math

from stable_baselines3.common.vec_env import VecMonitor
from loyalwingmen.rl_tools.directory_manager import DirectoryManager

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
    study_name = "01.10.2023"#datetime.date.today()
    models_dir, logs_dir, output_folder = directories(study_name)

    number_of_logical_cores = cpu_count()
    n_envs = int(number_of_logical_cores)

    env_fns = [lambda: Level2(GUI=False, rl_frequency=15) for _ in range(n_envs)]

    vectorized_environment = VecMonitor(SubprocVecEnv(env_fns))  # type: ignore

    callback_list = callbacklist(
        vectorized_environment,
        log_path=logs_dir,
        model_path=models_dir,
        save_freq=100_000,
    )

    nn_t = [512, 512, 128]

    policy_kwargs = dict(
        features_extractor_class=LidarInertialActionExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=nn_t, vf=nn_t),
    )

    model = PPO(
        "MultiInputPolicy",
        vectorized_environment,
        verbose=0,
        device="cuda",
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
        batch_size=128,
    )

    print(model.policy)
    model.learn(total_timesteps=2_000_000, callback=callback_list)
    model.save("ppo_level2_lidar")


if __name__ == "__main__":
    # https://stackoverflow.com/questions/29690091/python2-7-exception-the-freeze-support-line-can-be-omitted-if-the-program
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
