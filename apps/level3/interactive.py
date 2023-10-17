import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)

from stable_baselines3.common.env_checker import check_env

from loyalwingmen.modules.utils.keyboard_listener import KeyboardListener
from loyalwingmen.modules.environments_pyflyt.level3.pyflyt_level3_environment import (
    PyflytL3Enviroment as Level3,
)

# from loyalwingmen.modules.utils.displaytext import log
import numpy as np

env = Level3(GUI=True, rl_frequency=15)

keyboard_listener = KeyboardListener(env.get_keymap())

observation, info = env.reset()
data = {}
for _ in range(50_000):
    action = keyboard_listener.get_action()
    #action = np.array(np.random.rand(4))
    #action = np.array([-1, -1, -1, 0.5])
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    # log(f"reward:{reward:.2f}")

    if terminated:
        observation, info = env.reset()
