import csv
from glob import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

trials_path = glob("*/", recursive=True)

all_data = []

# Collect all the data
for trial_path in trials_path:
    csv_file_path = Path(trial_path, "logs_dir", "progress.csv")
    # Read the CSV file, assuming the first row is the header
    training_data = pd.read_csv(csv_file_path, header=0)

    # Check if 'time/total_timesteps' column exists and set it as the index
    if "time/total_timesteps" in training_data.columns:
        training_data.set_index("time/total_timesteps", inplace=True)
    else:
        print(
            f"Column 'time/total_timesteps' does not exist in the file: {csv_file_path}"
        )
        continue  # Skip this file or handle the error as needed

    # Fill missing data with the rolling mean, window of 5
    training_data_filled = training_data.fillna(
        training_data.rolling(5, min_periods=1).mean()
    )

    # Append the 'rollout/ep_rew_mean' series to the list, check if the column exists
    if "rollout/ep_rew_mean" in training_data_filled.columns:
        all_data.append(training_data_filled["rollout/ep_rew_mean"])
    else:
        print(
            f"Column 'rollout/ep_rew_mean' does not exist in the file: {csv_file_path}"
        )

    print(trial_path)

# Convert the list of Series into a single DataFrame where each column is a trial
combined_data = pd.concat(all_data, axis=1, join="outer")

# Calculate mean and standard deviation across trials at each time point
mean_performance = combined_data.mean(axis=1)
std_performance = combined_data.std(axis=1)

# Plotting code below...


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

plt.plot(
    all_data[8].index,
    all_data[8],
    label=f"Bayesian Optimizer Trial {8}",
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
plt.ylabel("Moving Mean Reward")
# plt.title("Performance Across Trials")
plt.legend()

# Save the figure with a higher DPI
# plt.savefig(
#    "performance_plot.pdf", dpi=300
# )  # Saves the figure in PDF format with 300 DPI

# If you want to save in SVG or EPS format, simply change the file extension
plt.savefig(
    "exp 03 - performance_plot - single 10 times.svg", dpi=300
)  # Saves the figure in SVG format
# plt.savefig('performance_plot.eps', dpi=300)  # Saves the figure in EPS format

plt.show()
