from pathlib import Path
from typing import Any, cast
import os
import sys
import time
import cProfile
import pstats

import numpy as np
import torch
from stable_baselines3 import PPO

from threatsense.level5.level5_c2_fusion_environment import (
    Level5C2FusionEnvironment as Level5C2Fusion,
)


current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)


N_GUI_EPISODES = 1

INPUT_MODEL = Path(
    # "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/rl_2/c2_27.04.2026/train_rl_1/models/final.zip"
    # "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/rl_1/26.04.2026/train_rl_1/models/final.zip"
    # "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/sl/25.04.2026/train_sl_9/train_sl_9_model.zip"
    # "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/sl/25.04.2026/train_sl_2/train_sl_2_model.zip"
    # "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/rl_1/26.04.2026/train_rl_1/models/final.zip"

    # "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/sl/29.04.2026/train_sl_2/model.zip"
    # "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/sl/29.04.2026/train_sl_4/model.zip"
    # "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/rlc2/29.04.2026_2/train_rl_1/models/final.zip"
    # "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/sl/30.04.2026/train_sl_1/model.zip"
    "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/rlc1/30.04.2026_c1_2/train_rl_1/models/final.zip"
)

ENABLE_PROFILING = True
PROFILE_OUTPUT_DIR = Path(
    "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/profiling/gui"
)
PROFILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_BINARY_FILE = PROFILE_OUTPUT_DIR / "gui_inspection.prof"
PROFILE_TEXT_FILE = PROFILE_OUTPUT_DIR / "gui_inspection_profile.txt"


def get_torch_device() -> torch.device:
    return torch.device("cpu")


def run_gui_inspection() -> None:
    environment = Level5C2Fusion(GUI=True)

    device = get_torch_device()

    model = PPO.load(
        INPUT_MODEL,
        env=environment,
        device=device,
        custom_objects={
            "learning_rate": 1e-3,
            "batch_size": 128,
        },
    )

    observation, _ = environment.reset()
    episode_count = 0
    current_episode_reward = 0.0
    current_episode_length = 0

    print("Running GUI qualitative inspection.")
    print("No files will be saved.")
    print("Watch for: pursuit, evasive motion, boundary exits, kills, and ally dependence.")

    try:
        while episode_count < N_GUI_EPISODES:

            noisy_last_action = np.random.uniform(
                low=-1.0,
                high=1.0,
                size=observation["last_action"].shape,
            ).astype(np.float32)
            noisy_last_action[..., 3:] = np.random.uniform(
                low=0.0,
                high=1.0,
                size=noisy_last_action[..., 3:].shape,
            ).astype(np.float32)
            observation["last_action"] = noisy_last_action

            action, _ = model.predict(
                cast(Any, observation), deterministic=True)

            action_delta = action - noisy_last_action
            print(
                f"input noisy_last_action: {noisy_last_action}, "
                f"model action: {action}, "
                f"norm(action - input_last_action): {np.linalg.norm(action_delta):.6f}"
            )

            observation, reward, terminated, truncated, info = environment.step(
                action)

            done = terminated or truncated

            current_episode_reward += float(reward)
            current_episode_length += 1

            if done:
                episode_info = dict(info)
                episode_count += 1

                agent_kills = episode_info.get("agent_kills", 0)
                allies_kills = episode_info.get("allies_kills", 0)
                total_kills = episode_info.get("total_kills", 0)
                agent_suicides = episode_info.get("agent_suicides", 0)

                print(
                    f"Episode {episode_count}/{N_GUI_EPISODES} | "
                    f"reward={current_episode_reward:.2f} | "
                    f"duration={current_episode_length} | "
                    f"total_kills={total_kills} | "
                    f"agent_kills={agent_kills} | "
                    f"allies_kills={allies_kills} | "
                    f"agent_suicides={agent_suicides}"
                )

                current_episode_reward = 0.0
                current_episode_length = 0
                observation, _ = environment.reset()

    except KeyboardInterrupt:
        print("\nGUI inspection interrupted by user.")

    finally:
        environment.close()
        print("GUI inspection finished.")


def main() -> None:
    if not ENABLE_PROFILING:
        run_gui_inspection()
        return

    profiler = cProfile.Profile()
    try:
        profiler.enable()
        run_gui_inspection()
    finally:
        profiler.disable()
        profiler.dump_stats(PROFILE_BINARY_FILE)

        with PROFILE_TEXT_FILE.open("w") as profile_file:
            stats = pstats.Stats(profiler, stream=profile_file)
            stats.strip_dirs()
            stats.sort_stats("cumulative")
            stats.print_stats(60)

        print(f"Profiling binary saved to: {PROFILE_BINARY_FILE}")
        print(f"Profiling report saved to: {PROFILE_TEXT_FILE}")


if __name__ == "__main__":
    main()
