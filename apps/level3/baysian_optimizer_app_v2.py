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

import logging
import warnings

from typing import List, Tuple

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from loyalwingmen.environments.level3.pyflyt_level3_environment import (
    PyflytL3Enviroment as Level3,
)
from loyalwingmen.rl_framework.agents.policies.ppo_policies import (
    LidarInertialActionExtractor2,
)
from loyalwingmen.rl_framework.utils.pipeline import (
    ReinforcementLearningPipeline,
    callbacklist,
    CallbackType,
)
from loyalwingmen.rl_framework.utils.directory_manager import DirectoryManager

import torch
from pathlib import Path


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
        suggestions,
        n_timesteps=n_timesteps,
        models_dir=models_dir,
        logs_dir=logs_dir,
        n_eval_episodes=10,
        trial_number=trial.number,
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
    # This version is for load and train the model.
    suggestions = {}

    suggestions["hidden_1"] = 512
    suggestions["hidden_2"] = 256
    suggestions["hidden_3"] = 128
    suggestions["rl_frequency"] = 15

    suggestions["learning_rate"] = 10 ** trial.suggest_int("exponent", -6, -2)
    suggestions["batch_size"] = trial.suggest_categorical(
        "batch_size", [64, 128, 256, 512, 1024]
    )

    return suggestions


def log_suggested_parameters(suggestions: dict):
    info_message = "Suggested Parameters:\n"
    for key in suggestions:
        info_message += f"  - {key}: {suggestions[key]}\n"
    logging.info(info_message)


def rl_pipeline(
    suggestions: dict,
    n_timesteps: int,
    models_dir: str,
    logs_dir: str,
    n_eval_episodes: int = 3,
    trial_number: int = 0,
) -> Tuple[float, float, float]:
    frequency = 15  # suggestions["rl_frequency"]
    learning_rate = suggestions["learning_rate"]

    hiddens = get_hiddens(suggestions)

    vectorized_environment: VecMonitor = (
        ReinforcementLearningPipeline.create_vectorized_environment(
            environment=Level3,
            env_kwargs=suggestions,
            n_envs=int((os.cpu_count() or 1)),
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

    path_lib = Path("from_level_2/mPPO-r4985.2099609375-sd1149.3851318359375.zip")
    model = PPO.load(str(path_lib), env=vectorized_environment)
    lr_schedule = lambda _: learning_rate
    model.learning_rate = lr_schedule
    model.batch_size = suggestions["batch_size"]
    model.tensorboard_log = specific_log_folder
    # model.set_env(vectorized_environment)

    logging.info(model.policy)
    model.learn(total_timesteps=n_timesteps, callback=callback_list)

    avg_reward, std_dev, num_episodes = ReinforcementLearningPipeline.evaluate(
        model, vectorized_environment, n_eval_episodes=n_eval_episodes
    )
    ReinforcementLearningPipeline.save_model(
        model,
        hiddens,
        frequency,
        learning_rate,
        avg_reward,
        std_dev,
        models_dir,
        trial_number=trial_number,
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
    output_folder = os.path.join("output_level3", app_name, study_name)
    DirectoryManager.create_directory(output_folder)

    models_dir = os.path.join(output_folder, "models_dir")
    logs_dir = os.path.join(output_folder, "logs_dir")

    return models_dir, logs_dir, output_folder


def main():
    n_timesteps = 2_000_000
    n_timesteps_in_millions = n_timesteps / 1e6
    study_name = f"19_12.2023_level3_{n_timesteps_in_millions:.2f}M"

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
