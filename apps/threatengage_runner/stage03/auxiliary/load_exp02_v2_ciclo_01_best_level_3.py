import sys

sys.path.append("..")
from stable_baselines3 import PPO

from threatengage.environments.level4.exp02_v2_environment import (
    Exp02V2Environment as level4,
)

import cProfile
import pstats


def setup_environment():
    env = level4(GUI=True, rl_frequency=15)
    model = PPO.load(
        "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level3\\output\\baysian_optimizer_app_v2_ciclo_02\\19_01_2023_level3_cicle_02_2.00M_v2_6_m_p2_600_calls\\Trial_2\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t2_PPO_r4427.63.zip"
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
            f"reward:{reward:.2f}, acc_reward:{acc_reward:.2f} gun:{observation['inertial_data'][-3:]} position:{observation['inertial_data'][:3]}"
        )

        acc_reward += reward
        if terminated:
            print("terminated")
            observation, info = env.reset(0)

            acc_reward = 0


if __name__ == "__main__":
    main()
