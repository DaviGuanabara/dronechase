import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Paths for the Excel files
paths = [
    "evaluation_03.02.2024_exp01_1bt_episode_info.xlsx",
    "evaluation_03.02.2024_exp02_1rl_episode_info.xlsx",
    "evaluation_03.02.2024_exp03_1rl_1bt_episode_info.xlsx",
    "evaluation_03.02.2024_exp04_1rl_1bt_trained_stopped_episode_info.xlsx",
    "evaluation_03.02.2024_exp05_2RL_episode_info.xlsx",
    "evaluation_03.02.2024_exp06_2bt_episode_info.xlsx",
    "evaluation_04.02.2024_exp07_1RL_1NOT_MOVING_episode_info.xlsx",
]

dataframes = []
for i, file_path in enumerate(paths):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    evaluation_data = pd.read_excel(file_path, header=0)
    evaluation_data["Experiment"] = f"{i+1}"
    dataframes.append(evaluation_data)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Create the boxplot
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))
sns.boxplot(data=combined_df, x="Experiment", y="kills", palette="Pastel1")

# Customize the plot
plt.xlabel("Experiment", fontsize=14)
plt.ylabel("Loitering Munitions Neutralized", fontsize=14)
plt.xticks(rotation=0, fontsize=12)  # Rotate x labels for better readability
plt.yticks(fontsize=12)
plt.ylim(bottom=0)  # Set the y-axis to start from 0
plt.tight_layout()  # Adjust layout to fit the plot and annotations

# Save the figure in SVG format
plt.savefig("boxplot_experiment_kills.svg", format="svg", dpi=300)

plt.show()
