import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Path to the Excel file
exp_01_path = Path("evaluation_03.02.2024_exp01_1bt_episode_info.xlsx")
exp_02_path = Path("evaluation_03.02.2024_exp02_1rl_episode_info.xlsx")
exp_03_path = Path("evaluation_03.02.2024_exp03_1rl_1bt_episode_info.xlsx")
exp_04_path = Path(
    "evaluation_03.02.2024_exp04_1rl_1bt_trained_stopped_episode_info.xlsx"
)
exp_05_path = Path("evaluation_03.02.2024_exp05_2RL_episode_info.xlsx")
exp_06_path = Path("evaluation_03.02.2024_exp06_2bt_episode_info.xlsx")
exp_07_path = Path("evaluation_04.02.2024_exp07_1RL_1NOT_MOVING_episode_info.xlsx")

# paths = [exp_01_path, exp_02_path, exp_03_path, exp_04_path, exp_05_path, exp_06_path]
# paths = [exp_01_path, exp_02_path]
# paths = [exp_03_path, exp_04_path]

# paths = [exp_05_path, exp_06_path]
# paths = [exp_07_path]

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
    # Check if the file exists
    if not csv_file_path.exists():
        print(f"File not found: {csv_file_path}")
    else:
        # Read the Excel file, assuming the first sheet and header in the first row
        evaluation_data = pd.read_excel(csv_file_path, header=0)

        # Check if the 'episode' column exists
        if "episode" in evaluation_data.columns:
            evaluation_data.set_index("episode", inplace=True)
        else:
            print(f"Column 'episode' does not exist in the file: {csv_file_path}")

        evaluation_data.insert(0, "exp", "exp" + str(i + 1))
        dataframes.append(evaluation_data)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes)


# Now create a new column that combines the evaluation and the name for unique identifiers
combined_df["exp_name"] = combined_df["exp"] + "_" + combined_df["name"]


unique_names = combined_df["exp_name"].unique()


# Custom sort function that sorts 'bt' before 'nn'
def custom_sort(name):
    if "bt" in name:
        return "0_" + name  # '0_' ensures 'bt' comes before 'nn'
    else:
        return "1_" + name


# Now sort the unique names with the custom sort function
sorted_names = sorted(unique_names, key=custom_sort)


# Assuming 'combined_df' is a pre-existing DataFrame with the correct data.
sns.set_style("whitegrid")

# Create a larger figure size to accommodate the plots
plt.figure(figsize=(12, 8))

# Create the boxplot (instead of a violin plot)
sns.boxplot(
    data=combined_df,
    x="exp_name",
    y="kills",
    palette="Pastel1",  # order=sorted_names
)

# Customize the plot
plt.xlabel("Experiment and Loyal Wingman", fontsize=14)
plt.ylabel("Loitering Munitions Neutralized", fontsize=14)
plt.xticks(rotation=45, fontsize=12)  # Rotate x labels for better readability
plt.yticks(fontsize=12)

# Set the y-axis to start from 0
plt.ylim(bottom=0)

plt.tight_layout()  # Adjust layout to fit the plot and annotations
plt.show()
