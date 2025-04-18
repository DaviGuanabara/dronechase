import sys

sys.path.append("..")
from stable_baselines3 import PPO

from threatengage.environments.level4.evaluation_environment import (
    EvaluationEnvironment as level4,
)

import cProfile
import pstats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from threatengage.rl_framework.utils.pipeline import (
    ReinforcementLearningPipeline,
    callbacklist,
    CallbackType,
)

from stable_baselines3.common.vec_env import VecMonitor
from typing import Dict, List


def preprocess_data(data: List[Dict]):

    pre_data = []

    for i, episode in enumerate(data):
        lw_names = episode.keys()

        for name in lw_names:
            values = list(episode[name].values())
            print("Values:")
            print(values)
            values.append(name)
            values.append(i + 1)
            pre_data.append(values)

    pandas_data = pd.DataFrame(
        pre_data,
        columns=["kills", "alive", "munitions", "wave", "step", "name", "episode"],
    )

    print(pandas_data)
    return pandas_data


def update_data(info: Dict, data: Dict):
    # Preprocessing the data
    # data = data
    lw_names = info.keys()
    # print(info)
    for name in lw_names:
        if name not in data.keys():
            data[name] = info[name]

        elif info[name]["step"] > data[name]["step"]:
            data[name] = info[name]

    return data


def main():
    GUI = True
    episodes = 100

    configuration = {}
    configuration["SHOW_NAME"] = GUI
    configuration["drivers"] = [
        {
            "type": "nn",
            "name": "nn_1",
            "path": "C:\\Users\\davi_\\OneDrive\\1. Projects\\1. Dissertação\\01 - Code\\Experiment Results\\Etapa 03\\Treinamento - 1RL - EXP02\\exp02 vfinal 31_01_2024_level4_2.00M_exp02_vFinal\\Trial_6\\models_dir\\h[128, 256, 512]_f15_lr0.001\\t6_PPO_r5618.51.zip",
        },
    ]

    env = level4(configuration, GUI=GUI, rl_frequency=15)
    observation, _ = env.reset(0)

    all_data = []

    for episode in range(episodes):
        episode_data = {}  # List to store step data for the current episode

        while True:
            # action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(np.zeros(1))

            # Store step info in episode_data
            update_data(info, episode_data)
            print(info)

            if terminated:
                print("terminated")
                all_data.append(episode_data)  # Store the collected episode data
                observation, info = env.reset(0)

                break

    return all_data

    # At this point, episode_infos contains the info data for each episode
    # You can now analyze this data or save it to a file for later analysis


if __name__ == "__main__":
    all_data = main()
    preprossed_data = preprocess_data(all_data)
    # print(all_data)

    # Save the DataFrame to an Excel file
    excel_filename = "evaluation_03.02.2024_exp02_1rl_episode_info.xlsx"
    preprossed_data.to_excel(excel_filename, index=False)

    print(f"Data saved to {excel_filename}")
