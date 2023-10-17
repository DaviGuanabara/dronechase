import sys

sys.path.append("..")
from stable_baselines3 import PPO

from loyalwingmen.modules.environments.level1.level1_environment import Level1
import cProfile
import pstats


def setup_environment():
    env = Level1(GUI=True, rl_frequency=30)
    model = PPO.load("./trained_level1_ppo")
    observation, _ = env.reset(0)
    return env, model, observation


def on_avaluation_step(env, model, observation):
    for steps in range(50_000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # logging.debug(f"(main) reward: {reward}")
        # print(f"reward:{reward:.2f} - action:{action}")

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
    )
    cProfile.run(
        "on_avaluation_step(env, model, observation)", "result_with_lidar.prof"
    )

    stats = pstats.Stats("result_with_lidar.prof")
    stats.sort_stats("cumulative").print_stats(
        20
    )  # Show the top 40 functions by cumulative time


if __name__ == "__main__":
    main()
