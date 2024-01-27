import sys
import os
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)

from stable_baselines3.common.env_checker import check_env

from loyalwingmen.environments.utils.keyboard_listener import KeyboardListener
from loyalwingmen.environments.level3.pyflyt_level3_environment import (
    PyflytL3Enviroment as Level3,
)

# from loyalwingmen.modules.utils.displaytext import log
import numpy as np

env = Level3(GUI=True, rl_frequency=15)

keyboard_listener = KeyboardListener(env.get_keymap())

observation, info = env.reset()
data = {}
reward_acc = 0
for _ in range(50_000):
    action = keyboard_listener.get_action()
    # action = np.array(np.random.rand(4))
    # action = np.array([0, 0, 0, 0.5])
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation["gun"])
    print(f"reward:{reward:.2f}, gun observation:{observation['gun']}")
    # reward_acc += reward
    # print(f"reward:{reward:.2f}, reward_acc:{reward_acc}")
    # log(f"reward:{reward:.2f}")
    # print(f"reward:{reward} - reward_acc:{reward_acc}")
    # print(f"observation:{observation}")

    # print(f"action:{action}")
    time.sleep(0.01)
    reward_acc += reward

    if terminated:
        observation, info = env.reset()
        reward_acc = 0
