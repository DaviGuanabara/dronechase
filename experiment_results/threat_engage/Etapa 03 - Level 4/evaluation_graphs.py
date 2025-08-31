import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to the Excel file
csv_file_path = Path("/evaluation_03.02.2024_exp06_2bt_episode_info.xlsx")

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

    # Check if the 'kills' and 'name' columns exist before plotting
    if "kills" in evaluation_data.columns and "name" in evaluation_data.columns:
        # Create the boxplot for 'kills' grouped by 'name'
        evaluation_data.boxplot(column="kills", by="name", notch=True, grid=False)
        plt.title("Boxplot of Kills by Name")
        plt.suptitle("")  # Remove the automatic 'Boxplot grouped by name' title
        plt.xlabel("Name")
        plt.ylabel("Kills")
        plt.show()
    else:
        print("Required columns 'kills' and/or 'name' are missing from the data.")
