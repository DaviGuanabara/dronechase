import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths for the Excel files
exp_01_path = Path("evaluation_03.02.2024_exp01_1bt_episode_info.xlsx")
exp_02_path = Path("evaluation_03.02.2024_exp02_1rl_episode_info.xlsx")
exp_03_path = Path("evaluation_03.02.2024_exp03_1rl_1bt_episode_info.xlsx")
exp_04_path = Path(
    "evaluation_03.02.2024_exp04_1rl_1bt_trained_stopped_episode_info.xlsx"
)
exp_05_path = Path("evaluation_03.02.2024_exp05_2RL_episode_info.xlsx")
exp_06_path = Path("evaluation_03.02.2024_exp06_2bt_episode_info.xlsx")
exp_07_path = Path("evaluation_04.02.2024_exp07_1RL_1NOT_MOVING_episode_info.xlsx")

paths = [
    exp_01_path,
    exp_02_path,
    exp_03_path,
    exp_04_path,
    exp_05_path,
    exp_06_path,
    exp_07_path,
]

dataframes = []
experiment_labels = []

for i, csv_file_path in enumerate(paths):
    if not csv_file_path.exists():
        print(f"File not found: {csv_file_path}")
    else:
        evaluation_data = pd.read_excel(csv_file_path, header=0)
        if "episode" in evaluation_data.columns:
            evaluation_data.set_index("episode", inplace=True)
        else:
            print(f"Column 'episode' does not exist in the file: {csv_file_path}")

        # Assuming 'kills' column exists and we're renaming it to 'hits' for a neutral term
        if "kills" in evaluation_data.columns:
            evaluation_data.rename(columns={"kills": "hits"}, inplace=True)

        evaluation_data["exp"] = f"Exp {i + 1}"
        dataframes.append(evaluation_data)
        experiment_labels.append(f"Exp {i + 1}")

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes)

# Determine the common x-axis range for all subplots based on the combined data
x_min = 0  # We set the minimum to 0, as negative 'hits' don't make sense.
x_max = combined_df["hits"].max()

# Create subplots with a shared y-axis
fig, axes = plt.subplots(
    len(paths), 1, figsize=(10, 2 * len(paths))
)  # Adjusted height for each subplot

# Flatten the axes array for easy indexing if it's multidimensional
if len(paths) > 1:
    axes = axes.flatten()

# Plotting each experiment's KDE in a separate subplot
for i, label in enumerate(experiment_labels):
    subset = combined_df[combined_df["exp"] == label]
    sns.kdeplot(subset["hits"], ax=axes[i], cut=0)
    # Shortened title
    axes[i].set_title(f"Exp {i+1}", fontsize=14)
    axes[i].set_xlabel("Hits", fontsize=12)
    axes[i].set_ylabel("Density", fontsize=12)
    axes[i].set_xlim(x_min, x_max)  # Use the common x-axis range

# Adjust layout for better spacing and display titles without breaking
plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the rect parameters as needed
plt.show()
