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


# Assuming episode_infos is populated as shown in the previous code

# Initialize metrics
total_kills = []
total_munitions = []
survival_rate = []

# Process data
for episode_data in episode_infos:
    episode_kills = [
        step["lw_kills"] for bt, step in episode_data[-1].items() if "lw_kills" in step
    ]
    episode_munitions = [
        step["lw_munitions"]
        for bt, step in episode_data[-1].items()
        if "lw_munitions" in step
    ]
    episode_survival = [
        step["lw_alive"] for bt, step in episode_data[-1].items() if "lw_alive" in step
    ]

    total_kills.append(np.mean(episode_kills))
    total_munitions.append(np.mean(episode_munitions))
    survival_rate.append(np.mean(episode_survival))

# Calculate average metrics across all episodes
avg_kills = np.mean(total_kills)
avg_munitions = np.mean(total_munitions)
avg_survival_rate = np.mean(survival_rate)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Average Kills per Episode
axs[0].plot(total_kills, label="Average Kills")
axs[0].set_title("Average Kills per Episode")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Kills")
axs[0].legend()

# Average Munitions Left per Episode
axs[1].plot(total_munitions, label="Average Munitions Left")
axs[1].set_title("Average Munitions Left per Episode")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Munitions")
axs[1].legend()

# Survival Rate per Episode
axs[2].plot(survival_rate, label="Survival Rate")
axs[2].set_title("Survival Rate per Episode")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Survival Rate")
axs[2].legend()

plt.tight_layout()
plt.show()

# Print average metrics
print(f"Average Kills: {avg_kills}")
print(f"Average Munitions Left: {avg_munitions}")
print(f"Average Survival Rate: {avg_survival_rate}")
