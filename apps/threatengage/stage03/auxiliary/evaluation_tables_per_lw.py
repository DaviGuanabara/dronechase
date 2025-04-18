import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Paths para os arquivos Excel
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
for i, csv_file_path in enumerate(paths):
    if not csv_file_path.exists():
        print(f"Arquivo não encontrado: {csv_file_path}")
    else:
        evaluation_data = pd.read_excel(csv_file_path, header=0)
        if "episode" in evaluation_data.columns and "name" in evaluation_data.columns:
            evaluation_data.set_index("episode", inplace=True)
            evaluation_data["exp"] = "exp" + str(i + 1)  # Add experiment identifier
            dataframes.append(evaluation_data)
        else:
            print(f"Coluna 'episode' ou 'name' não existe no arquivo: {csv_file_path}")

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes)

# Create a new column that combines the experiment and the name for unique identifiers
combined_df["exp_name"] = combined_df["exp"] + "_" + combined_df["name"]

# Group the data by the new unique identifier
grouped_data = combined_df.groupby("exp_name")

# Calculate the median, mean, and standard deviation of kills for each unique identifier
stats_kills = grouped_data["kills"].agg(["median", "mean", "std"]).reset_index()

# Rename columns for clarity
stats_kills.columns = [
    "Unique Identifier",
    "Median Kills",
    "Mean Kills",
    "Standard Deviation Kills",
]

print(stats_kills)
