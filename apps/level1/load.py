import sys

sys.path.append("..")
from stable_baselines3 import PPO

from loyalwingmen.modules.environments_pyflyt.level1.pyflyt_level1_environment import (
    PyflytL1Enviroment as Level1,
)
import cProfile
import pstats


def setup_environment():
    env = Level1(GUI=True, rl_frequency=15)
    model = PPO.load("./trained_level1_ppo_new_reward_v3_fixed")

    observation, _ = env.reset()
    return env, model, observation


def on_avaluation_step(env: Level1, model, observation):
    for _ in range(50_000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # logging.debug(f"(main) reward: {reward}")
        print(f"reward:{reward:.2f} - action:{action}")
        #print(f"observation:{observation}")
        # print(f"info:{info}")

        if terminated:
            print("terminated")
            observation, info = env.reset(0)


def main():
    env, model, observation = setup_environment()
    on_avaluation_step(env, model, observation)


if __name__ == "__main__":
    main()
