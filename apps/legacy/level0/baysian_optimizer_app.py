"""
This is the optimizer aplication for the level 1 environment. It uses the optuna library to optimize the hyperparameters of the PPO algorithm.
I made it before a hude refactoring and creation the level 1 environment.

So, the adaptations to make it work with the level 1 environment were simple, but i didn't have time to make it work.
I didnt test it.
I think it will work with some minor changes.
Because of that, i would say that this is a work in progress.

Status: Not working

Please, read description.markdown for more information.
"""

import os
import sys

sys.path.append("..")
import logging
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from loyalwingmen.modules.environments.level0.level0_environment import Level0
from loyalwingmen.rl_tools.pipeline import (
    ReinforcementLearningPipeline,
    callbacklist,
    CallbackType,
)
from loyalwingmen.rl_tools.directory_manager import DirectoryManager
from loyalwingmen.rl_tools.policies.ppo_policies import (
    CustomActorCriticPolicy,
    CustomCNN,
)

from typing import List, Tuple
from datetime import datetime
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from openpyxl import load_workbook, Workbook
from openpyxl.worksheet.worksheet import Worksheet
import warnings
from gymnasium import Env
from gymnasium import spaces, Env
from typing import Optional, Union

import torch as th
from torch import backends
from sys import platform

# warnings.filterwarnings("ignore", message="WARN: Box bound precision lowered by casting to float32")
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_random_parameters() -> Tuple[list, int, float]:
    hidden_dist = randint(10, 1000)
    num_hiddens = randint(3, 8).rvs()
    hiddens = [hidden_dist.rvs() for _ in range(num_hiddens)]
    frequency = randint(1, 8).rvs() * 15
    learning_rate = uniform(1e-9, 1e-1).rvs()
    return hiddens, frequency, learning_rate


def log_suggested_parameters(suggestions: dict):
    info_message = "Suggested Parameters:\n"
    for key in suggestions.keys():
        info_message += f"  - {key}: {suggestions[key]}\n"
    logging.info(info_message)


def suggest_parameters(trial: Trial) -> dict:
    suggestions = {}
    n_hiddens = trial.suggest_int(f"n_hiddens", 3, 3)
    for i in range(1, n_hiddens + 1):
        suggestions[f"hidden_{i}"] = trial.suggest_categorical(
            f"hiddens_{i}", [128, 256, 512]
        )

    # suggestions[f"hidden_{1}"] = 512
    # suggestions[f"hidden_{2}"] = 256
    # suggestions[f"hidden_{3}"] = 128

    suggestions["rl_frequency"] = trial.suggest_categorical(
        "frequency", [1, 2, 5, 10, 15]
    )
    # suggestions["learning_rate"] = 1e-7 #
    suggestions["learning_rate"] = 10 ** trial.suggest_int("exponent", -9, -2)
    # suggestions["speed_amplification"] = 1 #
    suggestions["speed_amplification"] = trial.suggest_categorical(
        "speed_amplification", [1]
    )  # [.5, 1, 2, 3, 4, 5])

    suggestions["model"] = trial.suggest_categorical("model", ["ppo"])  #'sac'

    return suggestions


def rl_pipeline(
    suggestion: dict,
    n_timesteps: int,
    models_dir: str,
    logs_dir: str,
    n_eval_episodes: int = 100,
) -> Tuple[float, float, float]:
    frequency = suggestion["rl_frequency"]
    learning_rate = suggestion["learning_rate"]
    speed_amplification = suggestion["speed_amplification"]

    hiddens = []
    for i in range(1, len(suggestion) + 1):
        key = f"hidden_{i}"
        if key in suggestion:
            hiddens.append(suggestion[key])
        else:
            break

    vectorized_environment: VecMonitor = (
        ReinforcementLearningPipeline.create_vectorized_environment(
            environment=Level0, env_kwargs=suggestion
        )
    )
    specific_model_folder = ReinforcementLearningPipeline.gen_specific_folder_path(
        hiddens, frequency, learning_rate, dir=models_dir
    )
    specific_log_folder = ReinforcementLearningPipeline.gen_specific_folder_path(
        hiddens, frequency, learning_rate, dir=logs_dir
    )

    callback_list = ReinforcementLearningPipeline.create_callback_list(
        vectorized_environment,
        model_dir=specific_model_folder,
        log_dir=specific_log_folder,
        callbacks_to_include=[CallbackType.EVAL, CallbackType.PROGRESSBAR],
        n_eval_episodes=n_eval_episodes,
        debug=True,
    )
    policy_kwargs = ReinforcementLearningPipeline.create_policy_kwargs(hiddens)
    model = ReinforcementLearningPipeline.create_model(
        model_type="PPO",
        vectorized_enviroment=vectorized_environment,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        logs_dir=specific_log_folder,
        debug=True,
    )

    logging.info(model.policy)
    model = ReinforcementLearningPipeline.train_model(model, callback_list, n_timesteps)

    avg_reward, std_dev, num_episodes = ReinforcementLearningPipeline.evaluate(
        model, vectorized_environment, n_eval_episodes=n_eval_episodes
    )
    ReinforcementLearningPipeline.save_model(
        model, hiddens, frequency, learning_rate, avg_reward, std_dev, models_dir
    )

    return avg_reward, std_dev, num_episodes


def objective(
    trial: Trial,
    output_folder: str,
    n_timesteps: int,
    study_name: str,
    models_dir: str,
    logs_dir: str,
) -> float:
    suggestions: dict = suggest_parameters(trial)
    log_suggested_parameters(suggestions)

    avg_score, std_deviation, n_episodes = rl_pipeline(
        suggestions, n_timesteps=n_timesteps, models_dir=models_dir, logs_dir=logs_dir
    )
    logging.info(f"Avg score: {avg_score}")

    print("saving results...")

    suggestions["avg_score"] = avg_score
    suggestions["std_deviation"] = std_deviation

    ReinforcementLearningPipeline.save_results_to_excel(
        output_folder, f"results_{study_name}.xlsx", suggestions
    )

    print("results saved")

    return avg_score


def get_os_name() -> str:
    if platform == "linux" or platform == "linux2":
        return "linux"
    elif platform == "darwin":
        return "macos"
    elif platform == "win32":
        return "windows"

    return "unknown"


def check_gpu():
    os_name = get_os_name()

    if os_name == "windows" and th.cuda.is_available():
        device_name = th.cuda.get_device_name(0)
        logging.info(
            f"Operating System: {os_name}\nGPU Available: Yes\nGPU Device: {device_name}"
        )
    elif os_name == "windows" and not th.cuda.is_available():
        logging.info(
            f"Operating System: {os_name}\nGPU Available: No\n Note: CUDA is not supported on this system or the necessary drivers are not installed."
        )
        if os_name == "windows":
            raise Exception("Fix CUDA support on Windows")

    elif os_name == "macos":
        logging.info(f"Operating System: {os_name}\nGPU may not be Available\n ")


def main():
    # TODO: PRECISO CORRIGIR OS LOGS DE FORMA QUE ELES SEJAM SALVOS EM ARQUIVOS DIFERENTES PARA CADA EXECUÇÃO, COM OS NOMES COMPATÍVEIS COM O DO MODELO.
    check_gpu()
    n_timesteps = 1_000_000
    n_timesteps_in_millions = n_timesteps / 1e6
    study_name = f"ULTIMO_RANDOM_no_physics_{n_timesteps_in_millions:.2f}M_steps_lidar_range_20m_32_20s"
    app_name = os.path.basename(__file__)
    app_name = os.path.join(app_name, study_name)

    output_folder = DirectoryManager.get_outputs_dir(app_name=app_name)
    models_dir = DirectoryManager.get_models_dir(app_name=app_name)
    logs_dir = DirectoryManager.get_logs_dir(app_name=app_name)

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(), study_name=study_name
    )
    study.optimize(
        lambda trial: objective(
            trial,
            output_folder,
            n_timesteps,
            study_name,
            models_dir=models_dir,
            logs_dir=logs_dir,
        ),
        n_trials=100,
    )


if __name__ == "__main__":
    main()
