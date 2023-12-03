import sys

sys.path.append("..")
from stable_baselines3 import PPO

from loyalwingmen.environments.level4.exp021_environment import Exp021Environment

import cProfile
import pstats


def setup_environment():
    env = Exp021Environment(GUI=True, rl_frequency=15)
    model = PPO.load(
        "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\experiments\\021\\exp02_output\\exp021_baysian_optimizer_app\\exp021_0.50M_dome_radius_20_02.12.2023\\models_dir\\h[512, 128, 128]-f15-lr0.0001\\mPPO-r156.59059143066406-sd163.1992950439453.zip"
    )

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
