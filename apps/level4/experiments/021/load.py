import sys

sys.path.append("..")
from stable_baselines3 import PPO

from loyalwingmen.environments.level4.exp021_environment import Exp021Environment

import cProfile
import pstats


def setup_environment():
    env = Exp021Environment(GUI=True, rl_frequency=15)
    model = PPO.load("exp021")

    observation, _ = env.reset(0)
    return env, model, observation


def on_avaluation_step(env: Exp021Environment, model, observation):
    reward_acc = 0
    for _ in range(5000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        reward_acc += reward
        # print("Reward:", reward, "Accumulated reward:", reward_acc)
        # logging.debug(f"(main) reward: {reward}")
        # print(f"reward:{reward:.2f} - action:{action} - observation:{observation}")
        # print("observation:", observation)
        print(action)
        if terminated:
            print("terminated")
            reward_acc = 0
            observation, info = env.reset(0)


def main():
    env, model, observation = setup_environment()
    cProfile.runctx(
        "on_avaluation_step(env, model, observation)",
        {
            "env": env,
            "model": model,
            "observation": observation,
            "on_avaluation_step": on_avaluation_step,
        },
        {},
        "result_with_lidar.prof",  # filename argument for output
    )

    stats = pstats.Stats("result_with_lidar.prof")
    stats.sort_stats("cumulative").print_stats(
        50
    )  # Show the top 40 functions by cumulative time


if __name__ == "__main__":
    main()
