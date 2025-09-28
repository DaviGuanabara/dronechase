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


from threatengage.environments.level2.pyflyt_level2_environment_modified_v2 import (
    PyflytL2EnviromentModifiedV2 as Level2,
)

from threatengage.rl_framework.utils.pipeline import (
    ReinforcementLearningPipeline,
    callbacklist,
    CallbackType,
)
from stable_baselines3.common.logger import configure
from threatengage.rl_framework.utils.directory_manager import DirectoryManager

from threatengage.rl_framework.agents.policies.ppo_policies import (
    # LidarInertialActionExtractor,
    # LidarInertialActionExtractor2,
    # LidarInertialActionExtractor2MODIFIED,
    LidarInertialActionExtractor,
)


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

    specific_output_folder = os.path.join(output_folder, f"Trial_{trial.number}")
    specific_model_folder = os.path.join(specific_output_folder, "models_dir")
    specific_log_folder = os.path.join(specific_output_folder, "logs_dir")

    DirectoryManager.create_directory(specific_output_folder)
    DirectoryManager.create_directory(specific_model_folder)
    DirectoryManager.create_directory(specific_log_folder)

    print(f"Directory {specific_output_folder} created")
    print(f"Directory {specific_model_folder} created")
    print(f"Directory {specific_log_folder} created")

    avg_reward, std_dev, n_episodes, episode_rewards = rl_pipeline(
        suggestions,
        n_timesteps=n_timesteps,
        models_dir=specific_model_folder,
        logs_dir=specific_log_folder,
    )
    logging.info(f"Avg score: {avg_reward}")

    print("saving results...")

    suggestions["avg_score"] = avg_reward
    suggestions["std_deviation"] = std_dev

    ReinforcementLearningPipeline.save_results_to_excel(
        output_folder, f"results_{study_name}.xlsx", suggestions
    )

    print("results saved")

    return avg_reward


def suggest_parameters(trial: Trial) -> dict:
    suggestions = {}

    n_hiddens = trial.suggest_int("n_hiddens", 3, 3)
    suggestions = {
        f"hidden_{i}": trial.suggest_categorical(f"hiddens_{i}", [128, 256, 512])
        for i in range(1, n_hiddens + 1)
    }

    # suggestions["hidden_1"] = trial.suggest_categorical("hidden_1", [512])
    # suggestions["hidden_2"] = trial.suggest_categorical("hidden_2", [256])
    # suggestions["hidden_3"] = trial.suggest_categorical("hidden_3", [128])

    suggestions["rl_frequency"] = trial.suggest_categorical(
        "frequency", [15]  # [1, 5, 10, 15, 30]
    )
    suggestions["learning_rate"] = 10 ** trial.suggest_int("exponent", -3, -3)
    suggestions["batch_size"] = trial.suggest_categorical(
        "batch_size", [128, 256, 512, 1024]
    )
    suggestions["features_dim"] = trial.suggest_categorical("feature_dim", [512])
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
    n_eval_episodes: int = 100,
) -> Tuple[float, float, float, List[float]]:
    frequency = suggestions["rl_frequency"]
    learning_rate = suggestions["learning_rate"]

    hiddens = get_hiddens(suggestions)

    vectorized_environment: VecMonitor = (
        ReinforcementLearningPipeline.create_vectorized_environment(
            environment=Level2, env_kwargs=suggestions, GUI=False
        )
    )

    # specific_model_folder = ReinforcementLearningPipeline.gen_specific_folder_path(
    #    hiddens, frequency, learning_rate, dir=models_dir
    # )
    # specific_log_folder = ReinforcementLearningPipeline.gen_specific_folder_path(
    #    hiddens, frequency, learning_rate, dir=logs_dir
    # )

    callback_list = ReinforcementLearningPipeline.create_callback_list(
        vectorized_environment,
        model_dir=models_dir,
        log_dir=logs_dir,
        callbacks_to_include=[
            CallbackType.EVAL,
            CallbackType.CHECKPOINT,
            CallbackType.PROGRESSBAR,
        ],
        n_eval_episodes=n_eval_episodes,
        debug=True,
    )

    policy_kwargs = dict(
        features_extractor_class=LidarInertialActionExtractor,
        features_extractor_kwargs=dict(features_dim=suggestions["features_dim"]),
        net_arch=dict(pi=hiddens, vf=hiddens),
    )

    model = PPO(
        "MultiInputPolicy",
        vectorized_environment,
        verbose=0,
        device="cuda",
        policy_kwargs=policy_kwargs,
        learning_rate=suggestions["learning_rate"],
        batch_size=suggestions["batch_size"],
    )

    new_logger = configure(logs_dir, ["csv", "tensorboard"])
    model.set_logger(new_logger)

    logging.info(model.policy)
    model.learn(total_timesteps=n_timesteps, callback=callback_list)

    (
        avg_reward,
        std_dev,
        n_episodes,
        episode_rewards,
    ) = ReinforcementLearningPipeline.evaluate(
        model, vectorized_environment, n_eval_episodes=n_eval_episodes
    )

    ReinforcementLearningPipeline.save_model(
        model, hiddens, frequency, learning_rate, avg_reward, std_dev, models_dir
    )

    episode_rewards = [
        float(x)
        for x in (
            episode_rewards if isinstance(episode_rewards, list) else [episode_rewards]
        )
    ]

    return avg_reward, std_dev, n_episodes, episode_rewards


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
    study_name = f"level2_{n_timesteps_in_millions:.2f}M_14.01.2023_baysian_v_old_v2_multi_topology_p2_homeoffice"

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
        n_trials=20,
    )


if __name__ == "__main__":
    main()
