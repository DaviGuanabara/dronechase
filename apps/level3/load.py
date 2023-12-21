import sys

sys.path.append("..")
from stable_baselines3 import PPO

from loyalwingmen.environments.level3.pyflyt_level3_environment import (
    PyflytL3Enviroment as Level3,
)

import cProfile
import pstats
from pathlib import Path

"""
[ ] - Aumentar o número de testes para 10 ou mais. Pois com somente 3, a sorte dá uma grande influência no resultado.
[ ] - Ele não está aprendendo. É como se ele estivesse sempre fazendo a mesma coisa.
[ ] - Aumentar o número de camadas ocultas para 4.
"""


def setup_environment():
    # path = "apps\\level3\\from level 2\\mPPO-r4985.2099609375-sd1149.3851318359375.zip"
    path = str(
        Path(
            "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level3\\output_level3\\baysian_optimizer_app_v2\\19_12.2023_level3_2.00M\\models_dir\\h[512, 256, 128]_f15_lr0.01\\PPO_t4_r759.74.zip"
        )
    )
    env = Level3(GUI=True, rl_frequency=15)
    model = PPO.load(path)
    observation, _ = env.reset(0)
    return env, model, observation


def on_avaluation_step(env: Level3, model, observation):
    reward_acc = 0
    for _ in range(5000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # logging.debug(f"(main) reward: {reward}")
        reward_acc += reward
        # print(
        #    f"reward:{reward:.2f} - reward_acc:{reward_acc} observation:{observation}"
        # )

        print(f"reward:{reward:.2f} - reward_acc:{reward_acc}")
        # print(action)

        if terminated:
            print("terminated")
            observation, info = env.reset(0)
            reward_acc = 0


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
