import sys

sys.path.append("..")
from stable_baselines3 import PPO

from threatengage.modules.environments_pyflyt.level3.pyflyt_level3_environment import (
    PyflytL3Enviroment as Level3,
)
import cProfile
import pstats


def setup_environment():
    env = Level3(GUI=True, rl_frequency=15)
    model = PPO.load("mPPO-r-668-intel-15.11.2023")
    observation, _ = env.reset(0)
    return env, model, observation


def on_avaluation_step(env: Level3, model, observation):
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
