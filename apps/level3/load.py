import sys

sys.path.append("..")
from stable_baselines3 import PPO

from loyalwingmen.environments.level3.pyflyt_level3_environment import (
    PyflytL3Enviroment as Level3,
)

import cProfile
import pstats

"""
[ ] - Aumentar o número de testes para 10 ou mais. Pois com somente 3, a sorte dá uma grande influência no resultado.
[ ] - Ele não está aprendendo. É como se ele estivesse sempre fazendo a mesma coisa.
[ ] - Aumentar o número de camadas ocultas para 4.
"""


def setup_environment():
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level3\\output_level3\\baysian_optimizer_app\\level3_2.00M_06_12.2023\\models_dir\\h[512, 128, 512]-f15-lr0.0001\\mPPO-r-230.39683532714844-sd573.9155883789062.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level3\\output_level3\\baysian_optimizer_app\\level3_2.00M_06_12.2023\\models_dir\\h[512, 128, 512]-f15-lr0.0001\\mPPO-r-262.65863037109375-sd636.4769287109375.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level3\\output_level3\\baysian_optimizer_app\\level3_2.00M_06_12.2023\\models_dir\\h[128, 128, 128]-f15-lr0.001\\mPPO-r-701.0654907226562-sd83.1894760131836.zip"
    path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level3\\output_level3\\baysian_optimizer_app\\level3_1.00M_08_12.2023_reward_fixed\\models_dir\\h[128, 256, 512]-f15-lr1e-05\\mPPO-r-768.593505859375-sd684.119140625.zip"
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
        print(
            f"reward:{reward:.2f} - reward_acc:{reward_acc}"
        )  # - observation:{observation}")
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
