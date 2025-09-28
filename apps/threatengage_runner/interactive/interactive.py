import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)

from stable_baselines3.common.env_checker import check_env

from threatengage.modules.utils.keyboard_listener import KeyboardListener
from threatengage.modules.environments_pyflyt.level1.pyflyt_level1_environment import (
    PyflytL1Enviroment as Level1,
)

# from loyalwingmen.modules.utils.displaytext import log
import numpy as np

env = Level1(GUI=True, rl_frequency=30)
observation, info = env.reset()


# keyboard_listener = KeyboardListener(env.get_keymap())
print("keyboard listener created")

data = {}
for _ in range(50_000):
    # print("getting action")
    action = -np.array(observation[0:3])
    action = np.concatenate((action, np.array([0.6])))
    # print("action took:", action)

    # action = np.array(np.random.rand(4))
    observation, reward, terminated, truncated, info = env.step(action)
    # print(action)
    # print(observation[12:15])
    print(reward)
    # print(info)
    # log(f"reward:{reward:.2f}")

    if terminated:
        observation, info = env.reset()
