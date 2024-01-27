import sys

sys.path.append("..")
from stable_baselines3 import PPO

from loyalwingmen.environments.level4.exp02_v3_environment import (
    Exp02V3Environment as level4,
)

import cProfile
import pstats


def setup_environment():
    env = level4(GUI=True, rl_frequency=15)
    model = PPO.load(
        # "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_v2_exp02_ciclo_01\\20_01_2023_level4_2.00M_v2_best_of_level4_p1\\Trial_0\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t0_PPO_r3694.00.zip"
        # "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp02_v2_ciclo_01\\21_01_2024_level4_2.00M_exp02_v2_best_of_level4_p2\\Trial_0\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t0_PPO_r-530.00.zip"
        # "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp02_v2_ciclo_01\\21_01_2024_level4_2.00M_exp02_v2_best_of_level4_p8\\Trial_0\\models_dir\\h[128, 256, 512]_f15_lr0.001\\t0_PPO_r3308.32.zip"
        # "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp02_v2_ciclo_01\\21_01_2024_level4_2.00M_exp02_v2_best_of_level4_p9\\Trial_1\\models_dir\\h[128, 256, 512]_f15_lr0.001\\t1_PPO_r4691.31.zip"
        # "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp02_v2_ciclo_01\\21_01_2024_level4_2.00M_exp02_v2_best_of_level4_p9\\Trial_6\\models_dir\\h[128, 256, 512]_f15_lr0.001\\t6_PPO_r6065.00.zip"
        # "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp02_v2_ciclo_01\\22_01_2024_level4_1.00M_exp02_v2_best_of_level3_p10\\Trial_0\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t0_PPO_r2473.98.zip"
        # "C:\\Users\\davi_\\Documents\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp02_v2_ciclo_01\\22_01_2024_level4_2.00M_exp02_v2_best_of_level3_p14\\Trial_1\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t1_PPO_r-19336.62.zip"
        # "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp02_v2_ciclo_01\\22_01_2024_level4_2.00M_exp02_v2_best_of_level3_p14\\Trial_2\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t2_PPO_r-147.46.zip"
        # "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp02_v2_ciclo_01\\23_01_2024_level4_2.00M_exp02_v2_best_of_level3_p16\\Trial_1\\models_dir\\h[128, 256, 512]_f15_lr0.001\\t1_PPO_r207.96.zip"
        # "C:\\Users\\davi_\\OneDrive\\1. Projects\\1. Dissertação\\01 - Code\\Experiment Results\\Etapa 03\\Ciclo01\\23_01_2024_level4_2.00M_exp02_v2_best_of_level3_p16\\Trial_3\\models_dir\\h[128, 256, 512]_f15_lr0.001\\t3_PPO_r1019.33.zip"
        "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp02_v3_ciclo_01\\23_01_2024_level4_v3_2.00M_exp02_v2_best_of_level3_p01\\Trial_0\\models_dir\\h[128, 256, 512]_f15_lr0.001\\t0_PPO_r2503.89.zip"
    )
    observation, _ = env.reset(0)
    return env, model, observation


def main():
    env, model, observation = setup_environment()

    acc_reward = 0
    for _ in range(5000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # logging.debug(f"(main) reward: {reward}")
        print(
            f"reward:{reward:.2f}, acc_reward:{acc_reward:.2f}, gun:{observation['inertial_data'][-3:]} position:{observation['inertial_data'][:3]}"
        )

        acc_reward += reward
        if terminated:
            print("terminated")
            observation, info = env.reset(0)

            acc_reward = 0


if __name__ == "__main__":
    main()
