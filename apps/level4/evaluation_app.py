import sys

sys.path.append("..")
from stable_baselines3 import PPO

from loyalwingmen.environments.level4.evaluation_environment import (
    EvaluationEnvironment as level4,
)

import cProfile
import pstats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loyalwingmen.rl_framework.utils.pipeline import (
    ReinforcementLearningPipeline,
    callbacklist,
    CallbackType,
)

from stable_baselines3.common.vec_env import VecMonitor


# Initialize a list to store the processed data for each episode
def preprocess_data(episode_infos):
    processed_data = []

    # Iterate through each episode in episode_infos
    for episode_idx, episode_data in enumerate(episode_infos):
        # Initialize a dictionary to store data for the current episode
        episode_dict = {
            "Episode": episode_idx + 1,  # Assuming episode numbering starts from 1
        }

        # Trackers for each 'loyalwingman' in the episode
        lw_kills = {}
        lw_max_wave = {}

        # Process data for each step in the episode
        for step_info in episode_data:
            for bt_name, bt_info in step_info.items():
                # Initialize kills and max_wave for each 'loyalwingman' if not already done
                lw_kills.setdefault(bt_name, 0)
                lw_max_wave.setdefault(bt_name, 0)

                # Update kills and max wave reached
                lw_kills[bt_name] += bt_info.get("lw_kills", 0)
                lw_max_wave[bt_name] = max(
                    lw_max_wave[bt_name], bt_info.get("current_wave", 0)
                )

        # Populate the episode_dict with the collected data
        for i, (bt_name, kills) in enumerate(lw_kills.items(), start=1):
            episode_dict[f"Max Wave LW {i}"] = lw_max_wave[bt_name]
            episode_dict[f"Kills LW{i}"] = kills

        processed_data.append(episode_dict)
    return pd.DataFrame(processed_data)


def main():
    GUI = False
    episodes = 10

    configuration = {}
    configuration["SHOW_NAME"] = GUI
    configuration["drivers"] = [
        {"type": "bt", "name": "bt_1"},
        {"type": "bt", "name": "bt_2"},
    ]

    env = level4(configuration, GUI=GUI, rl_frequency=15)
    observation, _ = env.reset(0)

    episode_infos = []  # List to store info data for each episode

    for episode in range(episodes):
        episode_data = []  # List to store step data for the current episode

        while True:
            # action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(np.zeros(1))

            # Store step info in episode_data
            episode_data.append(info)

            if terminated:
                print("terminated")
                episode_infos.append(episode_data)  # Store the collected episode data
                observation, info = env.reset(0)

                break

    return episode_infos

    # At this point, episode_infos contains the info data for each episode
    # You can now analyze this data or save it to a file for later analysis


if __name__ == "__main__":
    episode_infos = main()
    print(episode_infos)
    preprossed_data = preprocess_data(episode_infos)
    print(preprossed_data)

    # Save the DataFrame to an Excel file
    excel_filename = "episode_info.xlsx"
    preprossed_data.to_excel(excel_filename, index=False)

    print(f"Data saved to {excel_filename}")
