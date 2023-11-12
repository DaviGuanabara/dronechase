import cProfile
import sys
import os
import time
import tqdm as tqdm_renamed
from tqdm import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)

from stable_baselines3.common.env_checker import check_env
from loyalwingmen.modules.utils.keyboard_listener import KeyboardListener
from loyalwingmen.modules.environments.level0.pid_auto_tune import PIDAutoTuner
from loyalwingmen.modules.utils.displaytext import log
import numpy as np
import pstats
from multiprocessing import cpu_count
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
import torch

from loyalwingmen.modules.environments_pyflyt.level0.pyflyt_level0_environment import (
    Level0 as Level0_env,
)

env = Level0_env(rl_frequency=30, simulation_frequency=240, GUI=False, dome_radius=10)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
# model = PPO("MlpPolicy", env, verbose=1)
check_env(env)


def on_avaluation_step():
    n = 1_000
    number_of_logical_cores = cpu_count()
    n_envs = int(number_of_logical_cores / 2)
    env_fns = [
        lambda: PIDAutoTuner(
            rl_frequency=30, simulation_frequency=240, GUI=False, dome_radius=10
        )
        for _ in range(n_envs)
    ]
    vectorized_environment = VecMonitor(SubprocVecEnv(env_fns))  # type: ignore
    max_allowed_time = (
        1 / env.environment_parameters.rl_frequency
    )  # Convert frequency to time. For 30Hz, this is about 0.0333 seconds.

    successful_evaluations = 0
    worst_case = -1
    best_case = 1_000_000  # initialize best case to a large value
    total_time = 0  # accumulator for total time spent

    for _ in tqdm(range(n), desc="Processing", ncols=100):
        start_time = time.time()
        # vectorized_environment.step(np.zeros(4))
        actions = np.array([np.random.rand(4) for _ in range(n_envs)])
        vectorized_environment.step(actions)

        end_time = time.time()

        elapsed_time = end_time - start_time
        total_time += elapsed_time  # accumulate the elapsed time

        if elapsed_time <= max_allowed_time:
            successful_evaluations += 1

        worst_case = max(worst_case, elapsed_time)
        best_case = min(best_case, elapsed_time)
    average_frequency = n / total_time

    print(
        f"In {successful_evaluations} of {n} evaluations of env.step, the elapsed time was within the limit of {max_allowed_time:.4f} seconds."
    )
    print(f"Worst-case frequency: {1/worst_case:.4f} hz")
    if best_case == 0:
        print("Best-case frequency: Infinite (elapsed time was zero)")
    else:
        print(f"Best-case frequency: {1/best_case:.4f} hz")

    print(f"Average frequency: {average_frequency:.4f} hz")
    if successful_evaluations == n:
        print(
            f"Success: All {n} evaluations of env.step were within the limit of {max_allowed_time:.4f} seconds."
        )


def main():
    cProfile.run("on_avaluation_step()", "result_with_lidar.prof")

    stats = pstats.Stats("result.prof")
    stats.sort_stats("cumulative").print_stats(
        20
    )  # Show the top 10 functions by cumulative time


if __name__ == "__main__":
    # https://stackoverflow.com/questions/29690091/python2-7-exception-the-freeze-support-line-can-be-omitted-if-the-program
    # freeze_support() here if program needs to be frozen
    # pass
    main()  # execute this only when run directly, not when imported!