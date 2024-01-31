import sys

sys.path.append("..")
from stable_baselines3 import PPO

from loyalwingmen.environments.level4.exp05_cooperative_only_2_rl_environment import (
    Exp05CooperativeOnly2RLEnvironment as level4,
)

import cProfile
import pstats


def setup_environment():
    model_path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp05_cycle_01_home_office\\27_01_2024_level4_2.00M_exp05_p06\\Trial_0\\models_dir\\t0_s9_PPO.zip"
    # model_path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp05_cycle_01_home_office\\27_01_2024_level4_2.00M_exp05_p06\\Trial_0\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t0_PPO_r8550.07.zip"
    # model_path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level4\\output\\baysian_optimizer_app_exp05_cycle_01_home_office\\27_01_2024_level4_2.00M_exp05_p06\\Trial_0\\models_dir\\t0_s4_PPO.zip"
    model = PPO.load(model_path)

    env = level4(GUI=True, rl_frequency=15, model_path=model_path)

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
