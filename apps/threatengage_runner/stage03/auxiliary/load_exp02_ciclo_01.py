import sys

sys.path.append("..")
from stable_baselines3 import PPO

from threatengage.environments.level4.exp02_environment import (
    Exp02Environment as level4,
)

import cProfile
import pstats


def setup_environment():
    env = level4(GUI=True, rl_frequency=15)
    model = PPO.load(
        "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_v2_exp02_ciclo_01\\20_01_2023_level4_2.00M_v2_best_of_level4_p1\\Trial_0\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t0_PPO_r3694.00.zip"
    )
    observation, _ = env.reset(0)
    return env, model, observation


def on_avaluation_step(env: level4, model, observation):
    for _ in range(5000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # logging.debug(f"(main) reward: {reward}")
        # print(f"reward:{reward:.2f} - action:{action} - observation:{observation}")
        print(action)
        if terminated:
            print("terminated")
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
