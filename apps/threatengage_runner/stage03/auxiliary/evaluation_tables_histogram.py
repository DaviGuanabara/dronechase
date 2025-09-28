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

# Plotting
plt.figure(figsize=(20, 10))
for label in experiment_labels:
    subset = combined_df[combined_df["exp"] == label]
    # Plot the KDE for each subset
    sns.kdeplot(subset["hits"], label=label, cut=0)

# plt.title("Hits Distribution per Experiment")
plt.xlabel("Loitering Munition Neutralizations")
plt.ylabel("Density")
plt.legend()
plt.xlim(left=0)  # Set the left x-limit to 0
plt.show()
