"""

This evaluates the model trained in exp 04 (1RL + 1BT STOPPED) inside the exp 03 (1RL + 1BT), 
to show thata the training in exp03 leds to a better model than the training in exp04.
"""


import sys

sys.path.append("..")
from stable_baselines3 import PPO

from loyalwingmen.environments.level4.evaluation_exp03_cooperative_only_environment import (
    EvaluationExp03CooperativeOnlyEnvironment as level4,
)

import cProfile
import pstats

from stable_baselines3.common.vec_env import VecMonitor
from loyalwingmen.rl_framework.utils.pipeline import (
    ReinforcementLearningPipeline,
    callbacklist,
    CallbackType,
)

import csv
from glob import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

trials_path = glob(
    "*C:\\Users\\davi_\\OneDrive\\1. Projects\\1. Dissertação\\01 - Code\\Experiment Results\\Etapa 03\\Treinamento 1 RL + 1 BT PARADO\\25_01_2024_level4_2.00M_exp04_best_of_level3_p01\\",
    recursive=True,
)
print(trials_path)


def setup_environment():
    suggestions = {}
    suggestions["rl_frequency"] = 15
    # suggestions["GUI"] = True
    vectorized_environment: VecMonitor = (
        ReinforcementLearningPipeline.create_vectorized_environment(
            environment=level4, env_kwargs=suggestions, GUI=True
        )
    )
    # observation, _ = env.reset(0)
    return vectorized_environment


def main():
    # env = level4(GUI=True, rl_frequency=15)
    exp04_model = PPO.load(
        # "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp03_cycle_01_home_office\\25_01_2024_level4_2.00M_exp03_best_of_level3_p09\\Trial_0\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t0_PPO_r12698.07.zip"
        # "C:\\Users\\davi_\\OneDrive\\1. Projects\\1. Dissertação\\01 - Code\\Experiment Results\\Etapa 03\\Treinamento 1 RL + 1 BT PARADO\\25_01_2024_level4_2.00M_exp04_best_of_level3_p01\\Trial_4\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t4_PPO_r7678.80.zip"
        "C:\\Users\\davi_\\OneDrive\\1. Projects\\1. Dissertação\\01 - Code\\Experiment Results\\Etapa 03\\Treinamento 1 RL + 1 BT PARADO\\28_01_2024_level4_2.00M_exp04_best_of_level3_p01\\Trial_3\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t3_PPO_r3887.54.zip"
    )

    vectorized_environment = setup_environment()
    print("Evaluating the performance of EXP04 into an EXP03 environment")
    print("EXP04: 1RL + 1BT PARADO")
    print("EXP03: 1RL + 1BT")
    (
        avg_reward,
        std_dev,
        num_episodes,
        episode_rewards,
    ) = ReinforcementLearningPipeline.evaluate(
        exp04_model, vectorized_environment, n_eval_episodes=100
    )

    print(
        avg_reward,
        std_dev,
        num_episodes,
        episode_rewards,
    )


if __name__ == "__main__":
    main()
