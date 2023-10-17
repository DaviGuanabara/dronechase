import sys
import os

sys.path.append("..")


from ml.directory_manager import DirectoryManager
from stable_baselines3 import PPO

from prompt_toolkit.shortcuts import checkboxlist_dialog
from typing import List, Dict, Tuple
import logging
from typing import List, Dict, Tuple
import os
import glob
import logging
from prompt_toolkit.shortcuts import checkboxlist_dialog
from stable_baselines3 import PPO
from modules.environments.drone_chase_env import DroneChaseEnv
#from ml.pipeline

def describe_folder(folder_name: str) -> str:
    #parts = folder_name.split('_')
    
    #model_type, hidden, frequency, learning_rate, other_info = parts[1:6]
    #frequency = frequency.strip('[]').split(',')
    #description = f"Type: {model_type}, Hidden: {hidden}, Frequency: {frequency}, learning_rate: {float(learning_rate)}, other_info: {other_info}"
    return folder_name #description

def get_output_dir() -> str:
    base_dir = DirectoryManager.get_base_dir()
    logging.debug(f"(get_output_dir) base_dir: {base_dir}")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    logging.debug(f"(get_output_dir) outputs_dir: {outputs_dir}")
    return outputs_dir

def names_of_available_models() -> Tuple[List[str], Dict[str, str]]:
    base_path = get_output_dir()
    logging.debug(f"(names_of_available_models) base_path: {base_path}")
    print("base_path: ", base_path)
    zip_files = glob.glob(f"{base_path}/**/*.zip", recursive=True)
    names = []
    
    # Associate names with their corresponding ZIP files
    name_zip_map: Dict[str, str] = {}

    for zip_file in zip_files:
        parent_folder_name = os.path.basename(os.path.dirname(zip_file))
        file_name = os.path.basename(zip_file)
        parent_description = describe_folder(parent_folder_name)

        name_with_description = f"{file_name} -> {parent_description}"
        names.append(name_with_description)

        # Store the association in the map
        name_zip_map[name_with_description] = zip_file
        
    return names, name_zip_map

def select_model(names: List[str]) -> str:
    selected_name = checkboxlist_dialog(
        title="Select a model:",
        text="Use the arrow keys to navigate and the Space key to select. Press Enter to confirm.",
        values=[(name, name) for name in names],
    ).run()

    return selected_name[0] if selected_name else ""

def load_and_run_model(selected_zip: str, frequency: int = 15):
    if not selected_zip:
        print("No model selected.")
        return
    
    print("Selected model:", selected_zip)
    model = PPO.load(selected_zip)
    env = DroneChaseEnv(GUI=True, rl_frequency=frequency, debug=True)
    
    observation, info = env.reset()

    for steps in range(50_000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        #print(reward, action)

        if terminated:
            print("Episode terminated")
            env.reset()

def main():
    names, name_zip_map = names_of_available_models()
    selected_name = select_model(names)
    selected_zip = name_zip_map.get(selected_name, "")
    load_and_run_model(selected_zip)

if __name__ == "__main__":
    main()
