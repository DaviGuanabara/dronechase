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
from loyalwingmen.modules.environments.level0.pid_auto_tune import PIDAutoTuner

from loyalwingmen.modules.environments.level0.nova_controladora import (
    PID,
    QuadcopterController,
    QuadcopterDynamics,
)

from typing import Dict, List, Tuple, Union, Optional

# warnings.filterwarnings("ignore", message="WARN: Box bound precision lowered by casting to float32")
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def suggest_parameters(trial: Trial) -> np.ndarray:
    pid_gains = [trial.suggest_float(f"pid_gain_{i}", -1, 1) for i in range(18)]

    return np.array(pid_gains, dtype=np.float32)


def execute_env(suggestions):
    desires = [
        np.ones(3) * 0.5,
        np.ones(3),
        -np.ones(3) * 0.5,
        -np.ones(3),
        np.zeros(3),
        np.array([0, 0, 1]),
        np.array([0, 1, 0]),
        np.array([1, 0, 0]),
        np.array([1, 0, 1]),
        np.array([1, 1, 0]),
        np.random.uniform(-1, 1, 3),
        np.random.uniform(-1, 1, 3),
        np.random.uniform(-1, 1, 3),
        np.random.uniform(-1, 1, 3),
        np.random.uniform(-1, 1, 3),
    ]

    scores = []

    for desired_velocity in desires:
        pid_autotuner = PIDAutoTuner(10, 30, 240, False)
        pid_autotuner._reset_simulation(suggestions)
        for _ in range(5):
            pid_autotuner.apply_step_input(desired_velocity)

        flight_state = pid_autotuner.quadcopter.flight_state.copy()
        score = float(
            np.linalg.norm(desired_velocity - flight_state.get("velocity", np.zeros(3)))
        )

        if np.isnan(score):
            score = 1000

        scores.append(score)
        pid_autotuner.close()
    return sum(scores) / len(scores)


def objective(
    trial: Trial,
    results: Dict,
) -> float:
    suggestions: np.ndarray = suggest_parameters(trial)
    # logging.info(trial.number)

    score = execute_env(suggestions)
    results = update_results(results, trial.number, score, suggestions)

    data = np.concatenate((suggestions, np.array([score])))
    df = pd.DataFrame(data)
    df.to_excel("pid_bo_output.xlsx", index=False)
    return score


def update_results(
    results: Dict, trial_number: int, score: float, parameters: np.ndarray
) -> Dict:
    """
    Update the results dictionary to store the 10 best scores and their parameters.

    :param results: Dictionary containing the results.
    :param trial_number: The number of the trial.
    :param score: The score obtained in the trial.
    :param parameters: The parameters used in the trial.
    :return: The updated results dictionary.
    """
    # Add the current trial's results to the dictionary
    results[trial_number] = {"score": score, "parameters": parameters}

    # Sort the results dictionary by score
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]["score"]))

    # Keep only the top 10 results
    top_10_results = dict(list(sorted_results.items())[:10])

    return top_10_results


def main():
    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(), study_name="Novo Studo"
    )

    results = {}
    study.optimize(
        lambda trial: objective(trial, results),
        n_trials=100_000,
    )

    logging.info(f" results: {results}")


if __name__ == "__main__":
    main()
