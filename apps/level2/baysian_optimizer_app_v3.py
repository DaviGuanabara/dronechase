"""
This V3 differs for v2 only for the use of the extractor, where it is LidarInertialActionExtractor3.
The LidarInertialActionExtractor3 only have extractor for the lidar.
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


from loyalwingmen.environments.level2.pyflyt_level2_environment import (
    PyflytL2Enviroment as Level2,
)

from loyalwingmen.rl_framework.utils.pipeline import (
    ReinforcementLearningPipeline,
    callbacklist,
    CallbackType,
)

from loyalwingmen.rl_framework.utils.directory_manager import DirectoryManager

from loyalwingmen.rl_framework.agents.policies.ppo_policies import (
    # LidarInertialActionExtractor,
    LidarInertialActionExtractor3,
)

from stable_baselines3.common.logger import configure
import torch

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def objective(
    trial: Trial,
    n_timesteps: int,
    study_name: str,
    output_folder: str,
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

    avg_score, std_deviation, n_episodes = rl_pipeline(
        suggestions,
        n_timesteps=n_timesteps,
        models_dir=specific_model_folder,
        logs_dir=specific_log_folder,
        n_eval_episodes=100,
        trial_number=trial.number,
    )
    logging.info(f"Avg score: {avg_score}")

    print("saving results...")

    suggestions["avg_score"] = avg_score
    suggestions["std_deviation"] = std_deviation
    suggestions["trial_number"] = trial.number

    ReinforcementLearningPipeline.save_results_to_excel(
        output_folder, f"results_{study_name}.xlsx", suggestions
    )

    print("results saved")

    return avg_score


def suggest_parameters(trial: Trial) -> dict:
    suggestions = {}

    n_hiddens = trial.suggest_int("n_hiddens", 3, 3)
    suggestions = {
        f"hidden_{i}": trial.suggest_categorical(f"hiddens_{i}", [128, 256, 512, 1024])
        for i in range(1, n_hiddens + 1)
    }

    suggestions["rl_frequency"] = trial.suggest_categorical(
        "frequency", [15]  # [1, 5, 10, 15, 30]
    )
    suggestions["learning_rate"] = 10 ** trial.suggest_int("exponent", -6, -2)
    suggestions["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512, 1024]
    )
    suggestions["features_dim"] = trial.suggest_categorical("feature_dim", [256, 512])
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
    trial_number: int = 0,
) -> Tuple[float, float, float]:
    # app_name, _ = os.path.splitext(os.path.basename(__file__))

    frequency = suggestions["rl_frequency"]
    learning_rate = suggestions["learning_rate"]

    hiddens = get_hiddens(suggestions)

    vectorized_environment: VecMonitor = (
        ReinforcementLearningPipeline.create_vectorized_environment(
            environment=Level2, env_kwargs=suggestions
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
        callbacks_to_include=[CallbackType.EVAL, CallbackType.PROGRESSBAR],
        n_eval_episodes=n_eval_episodes,
        debug=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_kwargs = dict(
        features_extractor_class=LidarInertialActionExtractor3,
        features_extractor_kwargs=dict(
            features_dim=suggestions["features_dim"], device=device
        ),
        net_arch=dict(pi=hiddens, vf=hiddens),
    )

    model = PPO(
        "MultiInputPolicy",
        vectorized_environment,
        verbose=0,
        device=device,
        policy_kwargs=policy_kwargs,
        learning_rate=suggestions["learning_rate"],
        batch_size=suggestions["batch_size"],
        tensorboard_log=logs_dir,
    )

    new_logger = configure(logs_dir, ["csv", "tensorboard"])
    model.set_logger(new_logger)

    logging.info(model.policy)
    model.learn(
        total_timesteps=n_timesteps,
        callback=callback_list,
        tb_log_name="stage1_first_run",
    )

    # TODO: SAVE ALL THE DATA ABOUT THE TRAINING. USE 100 EPISODES TO EVALUATE THE MODEL
    (
        avg_reward,
        std_dev,
        num_episodes,
        episode_rewards,
    ) = ReinforcementLearningPipeline.evaluate(
        model, vectorized_environment, n_eval_episodes=n_eval_episodes
    )

    ReinforcementLearningPipeline.save_evaluation(episode_rewards, logs_dir)

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
    output_folder = os.path.join("output", app_name, study_name)
    print(output_folder)
    DirectoryManager.create_directory(output_folder)

    return output_folder


def main():
    n_trials = 100
    n_timesteps = 1_000_000
    n_timesteps_in_millions = n_timesteps / 1e6
    study_name = f"21_12.2023_level3_{n_timesteps_in_millions:.2f}M_v2_for_extractor_v3"

    print("Baysian Optimizer App - V2")
    print(f"number of trials: {n_trials}")
    print(f"number of timesteps: {n_timesteps_in_millions}M")
    print(f"study name: {study_name}")

    output_folder = directories(study_name)

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(), study_name=study_name
    )

    study.optimize(
        lambda trial: objective(
            trial,
            n_timesteps,
            study_name,
            output_folder,
        ),
        n_trials=n_trials,
    )

    trial_data_save_path_file = os.path.join(output_folder, "optuna_study_results.csv")

    trial_data = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trial_data.to_csv(trial_data_save_path_file, index=False)


if __name__ == "__main__":
    main()
