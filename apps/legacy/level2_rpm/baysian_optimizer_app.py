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
# Standard libraries
import os
import sys
import logging
import warnings
from datetime import datetime
from typing import List, Tuple, Optional, Union

# Third-party libraries
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from openpyxl import load_workbook, Workbook
from openpyxl.worksheet.worksheet import Worksheet
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from gymnasium import spaces, Env

# Your modules
from loyalwingmen.modules.environments.level2_rpm.level2_environment import Level2
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


# warnings.filterwarnings("ignore", message="WARN: Box bound precision lowered by casting to float32")
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def objective(
    trial: Trial,
    n_timesteps: int,
    study_name: str,
    output_folder: str,
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


def suggest_parameters(trial: Trial) -> dict:
    suggestions = {}

    n_hiddens = trial.suggest_int("n_hiddens", 3, 8)
    suggestions = {
        f"hidden_{i}": trial.suggest_categorical(f"hiddens_{i}", [128, 256, 512, 1024])
        for i in range(1, n_hiddens + 1)
    }

    suggestions["rl_frequency"] = trial.suggest_categorical(
        "frequency", [15, 30, 60, 120, 240]
    )
    suggestions["learning_rate"] = 10 ** trial.suggest_int("exponent", -9, -3)
    suggestions["batch_size"] = trial.suggest_categorical(
        "batch_size", [128, 256, 512, 1024, 2048, 4096]
    )

    return suggestions


def log_suggested_parameters(suggestions: dict):
    info_message = "Suggested Parameters:\n"
    for key in suggestions:
        info_message += f"  - {key}: {suggestions[key]}\n"
    logging.info(info_message)


def rl_pipeline(
    suggestion: dict,
    n_timesteps: int,
    models_dir: str,
    logs_dir: str,
    n_eval_episodes: int = 10,
) -> Tuple[float, float, float]:
    frequency = suggestion["rl_frequency"]
    learning_rate = suggestion["learning_rate"]

    hiddens = get_hiddens(suggestion)

    vectorized_environment: VecMonitor = (
        ReinforcementLearningPipeline.create_vectorized_environment(
            environment=Level2, env_kwargs=suggestion
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

    policy_kwargs = dict(net_arch=dict(pi=hiddens, vf=hiddens))
    model = ReinforcementLearningPipeline.create_ppo_model(
        vectorized_environment,
        policy_kwargs=policy_kwargs,
        tensorboard_log=specific_log_folder,
        learning_rate=learning_rate,
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


def get_hiddens(suggestion):
    hiddens = []
    for i in range(1, len(suggestion) + 1):
        key = f"hidden_{i}"
        if key in suggestion:
            hiddens.append(suggestion[key])
        else:
            break

    return hiddens


def directories(study_name: str):
    app_name, _ = os.path.splitext(os.path.basename(__file__))
    output_folder = os.path.join("output", app_name, study_name)
    DirectoryManager.create_directory(output_folder)

    models_dir = os.path.join(output_folder, "models_dir")
    logs_dir = os.path.join(output_folder, "logs_dir")

    return models_dir, logs_dir, output_folder


def main():
    n_timesteps = 2_000_000
    n_timesteps_in_millions = n_timesteps / 1e6
    study_name = f"level2_{n_timesteps_in_millions:.2f}M_end_to_end_NN_19.09.2023"

    models_dir, logs_dir, output_folder = directories(study_name)

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(), study_name=study_name
    )
    study.optimize(
        lambda trial: objective(
            trial,
            n_timesteps,
            study_name,
            output_folder,
            models_dir=models_dir,
            logs_dir=logs_dir,
        ),
        n_trials=100,
    )


if __name__ == "__main__":
    main()
