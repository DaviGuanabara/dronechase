import sys

sys.path.append("..")
from stable_baselines3 import PPO

from threatengage.environments.level4.exp02_environment import Exp02Environment

import cProfile
import pstats


def setup_environment():
    env = Exp02Environment(GUI=True, rl_frequency=15)
    model = PPO.load(
        "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\experiments\\02\\exp02_output\\exp02_baysian_optimizer_app\\exp02_0.50M_02.12.2023\\models_dir\\h[512, 256, 512]-f15-lr0.001\\mPPO-r-940.5191040039062-sd732.5383911132812"
    )

    observation, _ = env.reset(0)
    return env, model, observation


def on_avaluation_step(env: Exp02Environment, model, observation):
    reward_acc = 0
    for _ in range(5000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        reward_acc += reward
        print("Reward:", reward, "Accumulated reward:", reward_acc)
        # logging.debug(f"(main) reward: {reward}")
        # print(f"reward:{reward:.2f} - action:{action} - observation:{observation}")
        # print("observation:", observation)
        # print(action)
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
