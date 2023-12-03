import sys
import pandas as pd
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)

from stable_baselines3.common.env_checker import check_env

from loyalwingmen.environments.level4.exp01_environment import Exp01Environment


env = Exp01Environment(GUI=False, rl_frequency=15)

# Dictionary to store run data
run_data = {
    "Run_Number": [],
    "Kills": [],
    "Last_Wave": [],  # Field to store the last wave number
}

# Define the Excel file path
excel_path = os.path.join(current_directory, "exp01_run_data.xlsx")

# Running the environment 1000 times
for run in range(1000):
    print(f"Run {run+1}")

    observation, info = env.reset()
    kills = 0
    deads = 0
    is_building_destroyed = False
    last_wave = 0  # Variable to track the last wave

    for _ in range(50_000):
        observation, reward, terminated, truncated, info = env.step()

        # Update kills, deads, is_building_destroyed, and last_wave
        if (
            info["kills"] > kills
            or info["deads"] > deads
            or info["is_building_destroyed"] != is_building_destroyed
        ):
            print(
                f"Wave: {info['current_wave']} Kills: {info['kills']}, Deads: {info['deads']}, is_building_destroyed: {info['is_building_destroyed']}"
            )
            kills = info["kills"]
            deads = info["deads"]
            is_building_destroyed = info["is_building_destroyed"]
            last_wave = info["current_wave"]

        if terminated:
            print(f"Run {run+1}, TERMINATED with kills: {kills}")
            break

    # Storing data for each run
    run_data["Run_Number"].append(run + 1)
    run_data["Kills"].append(kills)
    run_data["Last_Wave"].append(last_wave)

    # Creating a DataFrame and saving after each run
    df = pd.DataFrame(run_data)

    # Save to Excel
    df.to_excel(excel_path, index=False)

    print(f"Run {run+1} data saved to {excel_path}")

print("All runs completed and data saved.")
