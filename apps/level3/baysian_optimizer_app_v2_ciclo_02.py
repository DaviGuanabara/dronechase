import os

import logging
import warnings

from typing import List, Tuple

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor


from loyalwingmen.environments.level3.pyflyt_level3_environment_v2 import (
    PyflytL3EnviromentV2 as Level3,
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

    suggestions["learning_rate"] = 10 ** trial.suggest_int("exponent", -6, -2)
    suggestions["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512, 1024]
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
    # app_name, _ = os.path.splitext(os.path.basename(__file__))

    frequency = suggestions["rl_frequency"]
    learning_rate = suggestions["learning_rate"]

    hiddens = get_hiddens(suggestions)

    vectorized_environment: VecMonitor = (
        ReinforcementLearningPipeline.create_vectorized_environment(
            environment=Level3, env_kwargs=suggestions
        )
    )

    callback_list = ReinforcementLearningPipeline.create_callback_list(
        vectorized_environment,
        model_dir=models_dir,
        log_dir=logs_dir,
        callbacks_to_include=[CallbackType.EVAL, CallbackType.PROGRESSBAR],
        n_eval_episodes=n_eval_episodes,
        debug=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # path_lib = Path("from_level_2/mPPO-r4985.2099609375-sd1149.3851318359375.zip")
    print("Melhor do Level 3 Ciclo 1: ")

    path_lib = Path(
        # "C:\\Users\\davi_\\OneDrive\\1. Projects\\1. Dissertação\\01 - Code\\Experiment Results\\Etapa 01\\level2_2.00M_14.01.2023_baysian_v_old_v2_multi_topology_p3_mac\\Trial_4\\models_dir\\h[128, 256, 512]_f15_lr0.001\\t0_PPO_r8861.86.zip"
        # "C:\\Users\\davi_\\OneDrive\\1. Projects\\1. Dissertação\\01 - Code\\Experiment Results\\Etapa 02\\Ciclo 01\\15_01_2023_level3_2.00M_v2_best_of_level2__6_m_limited_p8_mac\\Trial_3\\models_dir\\h[128, 256, 512]_f15_lr1e-05\\t3_PPO_r1360.74.zip"
        "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level3\\output\\baysian_optimizer_app_v2_ciclo_01\\18_01_2023_level3_2.00M_v2_best_of_level2_p2_6m_reward_fixed\\Trial_1\\models_dir\\h[128, 256, 512]_f15_lr0.001\\t1_PPO_r3597.46.zip"
    )

    print(path_lib)
    model = PPO.load(str(path_lib), env=vectorized_environment)
    lr_schedule = lambda _: learning_rate
    model.learning_rate = lr_schedule
    model.batch_size = suggestions["batch_size"]
    model.tensorboard_log = logs_dir

    new_logger = configure(logs_dir, ["csv", "tensorboard"])
    model.set_logger(new_logger)

    logging.info(model.policy)
    model.learn(
        total_timesteps=n_timesteps,
        callback=callback_list,
        tb_log_name="level3_second_run",
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
    n_timesteps = 2_000_000
    n_timesteps_in_millions = n_timesteps / 1e6

    print("A distância máxima dos kamikazes nascem até 6 metros")
    print("Esse é o ciclo 2 do level 3")
    study_name = (
        f"19_01_2023_level3_cicle_02_{n_timesteps_in_millions:.2f}M_v2_6_m_p2_600_calls"
    )

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
