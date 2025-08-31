import csv
from glob import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

trials_path = glob("*/", recursive=True)


# Assuming trials_path is a list of paths
all_data = []

# Collect all the data
for trial_path in trials_path:
    csv_file_path = Path(trial_path, "logs_dir", "progress.csv")
    training_data = pd.read_csv(csv_file_path)

    # Check if 'rollout/ep_rew_mean' column exists
    if "rollout/ep_rew_mean" in training_data.columns:
        # Filter out rows with timestep index less than 32.768
        # Assuming 'time/total_timesteps' is your timestep index
        if "time/total_timesteps" in training_data.columns:
            training_data = training_data[
                training_data["time/total_timesteps"] >= 32768
            ]
        else:
            print(f"No 'time/total_timesteps' in {trial_path}, skipping this file.")
            continue

        # Remove duplicate timestamps
        # Assuming 'time/total_timesteps' is your index for checking duplicates
        training_data = training_data.drop_duplicates(subset=["time/total_timesteps"])
        training_data_filled = training_data.fillna(
            training_data.rolling(5, min_periods=1).mean()
        )

        # Append the specific column of interest to the all_data list
        all_data.append(training_data_filled["rollout/ep_rew_mean"])
    else:
        print(
            f"No 'rollout/ep_rew_mean' column found in {trial_path}, skipping this file."
        )


# Convert the list of DataFrames into a single DataFrame where each column is a trial
# print(all_data)
combined_data = pd.concat(all_data, axis=1)


# Calculate mean and standard deviation across trials at each time point
mean_performance = combined_data.mean(axis=1)
std_performance = combined_data.std(axis=1)

# Plot mean performance and standard deviation as a background
plt.plot(
    mean_performance.index, mean_performance, label="Mean Performance", color="grey"
)
plt.fill_between(
    mean_performance.index,
    mean_performance - std_performance,
    mean_performance + std_performance,
    color="grey",
    alpha=0.2,
)

# Identify specific trials of interest
# For example, we can highlight the best and worst final performances
final_performances = [data.iloc[-1] for data in all_data]
best_trial_index = np.argmax(final_performances)
worst_trial_index = np.argmin(final_performances)

print(f"Best trial: {best_trial_index}")
print(f"Worst trial: {worst_trial_index}")
# Highlight the best and worst trials
# plt.plot(
#    all_data[best_trial_index].index,
#    all_data[best_trial_index],
#    label=f"Best: Trial_{best_trial_index}",
#    color="green",
# )

# for data in all_data:
#    plt.plot(data.index, data, alpha=0.7)

plt.plot(
    all_data[4].index,
    all_data[4],
    label=f"Best Eval: Trial_{4}",
    color="blue",
)


# plt.plot(
#    all_data[worst_trial_index].index,
#    all_data[worst_trial_index],
#    label=f"Worst: Trial_{worst_trial_index}",
#    color="red",
# )

# plt.plot(
#    all_data[4].index,
#    all_data[4],
#    label=f"Best in Evaluation: Trial_{4}",
#    color="yellow",
# )

# Optionally, annotate other trials of interest
# for i, data in enumerate(all_data):
#    if i == 4:
#        plt.plot(data.index, data, label=f"Trial {i}", alpha=0.7)

plt.xlabel("TimeSteps")
plt.ylabel("Moving Mean Reward (100)")
# plt.title("Performance Across Trials")
plt.legend()
plt.show()
