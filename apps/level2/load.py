import sys

sys.path.append("..")
from stable_baselines3 import PPO

from loyalwingmen.environments.level2.pyflyt_level2_environment import (
    PyflytL2Enviroment as Level2,
)
import cProfile
import pstats


def setup_environment():
    env = Level2(GUI=True, rl_frequency=15)
    model = PPO.load("./ppo_level2_lidar_03_12_2023")
    observation, _ = env.reset(0)
    return env, model, observation


def on_avaluation_step(env: Level2, model, observation):
    for steps in range(50_000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # logging.debug(f"(main) reward: {reward}")
        # print(f"reward:{reward:.2f} - action:{action} - observation:{observation}")

        if terminated:
            print("terminated")
            observation, info = env.reset(0)


def main():
    env, model, observation = setup_environment()
    on_avaluation_step(env, model, observation)
    # cProfile.runctx(
    #    "on_avaluation_step(env, model, observation)",
    #    {
    #        "env": env,
    #        "model": model,
    #        "observation": observation,
    #        "on_avaluation_step": on_avaluation_step,
    #    },
    #    {},
    # )
    # cProfile.run(
    #    "on_avaluation_step(env, model, observation)", "result_with_lidar.prof"
    # )

    # stats = pstats.Stats("result_with_lidar.prof")
    # stats.sort_stats("cumulative").print_stats(
    #    2
    # )  # Show the top 40 functions by cumulative time


if __name__ == "__main__":
    main()
