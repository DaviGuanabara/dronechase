import os

import logging
import warnings

from typing import List, Tuple

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor


from loyalwingmen.environments.level4.exp05_cooperative_only_2_rl_v2_environment import (
    Exp05CooperativeOnly2RLV2Environment as level4,
)

from loyalwingmen.rl_framework.utils.pipeline import (
    ReinforcementLearningPipeline,
    callbacklist,
    CallbackType,
)

from loyalwingmen.rl_framework.utils.directory_manager import DirectoryManager

from stable_baselines3.common.logger import configure
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

    suggestions[f"hidden_{1}"] = trial.suggest_categorical(f"hiddens_{1}", [128])
    suggestions[f"hidden_{2}"] = trial.suggest_categorical(f"hiddens_{2}", [256])
    suggestions[f"hidden_{3}"] = trial.suggest_categorical(f"hiddens_{3}", [512])

    suggestions["feature_dim"] = trial.suggest_categorical("feature_dim", [512])

    suggestions["learning_rate"] = 10 ** trial.suggest_int("exponent", -4, -3)
    suggestions["batch_size"] = trial.suggest_categorical(
        "batch_size", [128, 256, 512, 1024, 2048]  # [32, 64, 128, 256, 512, 1024]
    )

    suggestions["rl_frequency"] = trial.suggest_categorical("rl_frequency", [15])

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
    frequency = suggestions["rl_frequency"]
    learning_rate = suggestions["learning_rate"]

    vectorized_environment: VecMonitor = (
        ReinforcementLearningPipeline.create_vectorized_multi_agent_v2_environment(
            environment=level4, env_kwargs=suggestions
        )
    )

    print("Melhor do Level 3: ")
    print("t2_PPO_r4427.63.zip - Trial 02")
    path_lib = Path(
        "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level3\\output\\baysian_optimizer_app_v2_ciclo_02\\19_01_2023_level3_cicle_02_2.00M_v2_6_m_p2_600_calls\\Trial_2\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t2_PPO_r4427.63.zip"
    )

    model = PPO.load(str(path_lib), vectorized_environment, device="cuda")
    suggestions["model"] = model

    lr_schedule = lambda _: learning_rate
    model.learning_rate = lr_schedule
    model.batch_size = suggestions["batch_size"]
    model.tensorboard_log = logs_dir
    new_logger = configure(logs_dir, ["csv", "tensorboard"])
    model.set_logger(new_logger)

    logging.info(model.policy)

    vectorized_environment.env_method("update_model", model)

    callback_list = ReinforcementLearningPipeline.create_callback_list(
        vectorized_environment,
        model_dir=models_dir,
        log_dir=logs_dir,
        callbacks_to_include=[CallbackType.EVAL, CallbackType.PROGRESSBAR],
        n_eval_episodes=n_eval_episodes,
        debug=True,
    )

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model.learn(
        total_timesteps=n_timesteps,
        callback=callback_list,
        tb_log_name=f"stage{1}",
    )

    (
        avg_reward,
        std_dev,
        num_episodes,
        episode_rewards,
    ) = ReinforcementLearningPipeline.evaluate(
        model, vectorized_environment, n_eval_episodes=n_eval_episodes
    )

    ReinforcementLearningPipeline.save_evaluation(episode_rewards, logs_dir)

    hiddens = get_hiddens(suggestions)

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
    print("REMINDER: The exp05 is air combat only WITH 2 RL.")
    n_trials = 10
    n_timesteps = 500_000
    n_timesteps_in_millions = n_timesteps / 1e6
    study_name = f"28_01_2024_level4_{n_timesteps_in_millions:.2f}M_exp05_v2_p04"

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
