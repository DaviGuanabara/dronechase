import sys

sys.path.append("..")
from stable_baselines3 import PPO

from threatengage.environments.level4.exp04_cooperative_only_bt_stopped_environment import (
    Exp04CooperativeOnlyBTStoppedEnvironment as level4,
)

import cProfile
import pstats


def setup_environment():
    env = level4(GUI=True, rl_frequency=15)
    model = PPO.load(
        "C:\\Users\\davi_\\OneDrive\\1. Projects\\1. Dissertação\\01 - Code\\Experiment Results\\Etapa 03\\Treinamento 1 RL + 1 BT PARADO\\25_01_2024_level4_2.00M_exp04_best_of_level3_p01\\Trial_4\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t4_PPO_r7678.80.zip"
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
