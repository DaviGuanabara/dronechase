import sys

sys.path.append("..")
from stable_baselines3 import PPO

from loyalwingmen.modules.environments_pyflyt.level3.pyflyt_level3_environment import (
    PyflytL3Enviroment as Level3,
)
import cProfile
import pstats


def setup_environment():
    env = Level3(GUI=True, rl_frequency=30)
    model = PPO.load("C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level3\\output_level3\\baysian_optimizer_app\\level3_2.00M_30.09.2023\\models_dir\\h[128, 256, 128]-f15-lr1e-05\\mPPO-r-6121.62353515625-sd73.42634582519531.zip")
    observation, _ = env.reset(0)
    return env, model, observation


def on_avaluation_step(env: Level3, model, observation):
    for _ in range(500):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        #logging.debug(f"(main) reward: {reward}")
        #print(f"reward:{reward:.2f} - action:{action} - observation:{observation}")
        print(reward)
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
        "result_with_lidar.prof"  # filename argument for output
    )

    stats = pstats.Stats("result_with_lidar.prof")
    stats.sort_stats("cumulative").print_stats(50)  # Show the top 40 functions by cumulative time



if __name__ == "__main__":
    main()
