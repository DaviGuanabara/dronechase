import sys

sys.path.append("..")
from stable_baselines3 import PPO
from loyalwingmen.modules.environments.level2_lidar.level2_environment import (
    Level2_lidar,
)

import cProfile
import pstats


def on_avaluation_step():
    # IT NEEDS TO BE FIXED BEFORE USE
    env = Level2_lidar(GUI=True, rl_frequency=30)

    # preciso corrigir o caminho do modelo
    # model = PPO.load("./trained_level2_ppo_lidar_v2")
    model = PPO.load("trained_level2_ppo_lidar_v2")

    observation, info = env.reset(0)
    for steps in range(50_000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"reward:{reward:.2f} - action:{action}")
        if terminated:
            print("terminated")
            observation, info = env.reset(0)
            # break


def main():
    cProfile.run("on_avaluation_step()", "result_with_lidar.prof")

    stats = pstats.Stats("result.prof")
    stats.sort_stats("cumulative").print_stats(
        20
    )  # Show the top 10 functions by cumulative time


if __name__ == "__main__":
    main()
