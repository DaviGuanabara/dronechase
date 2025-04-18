import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)

from stable_baselines3.common.env_checker import check_env


from threatengage.environments.level4.exp02_v2_environment import Exp02V2Environment

import time
from threatengage.environments.utils.keyboard_listener import KeyboardListener

# from loyalwingmen.modules.utils.displaytext import log
import numpy as np

env = Exp02V2Environment(GUI=True, rl_frequency=15)

keyboard_listener = KeyboardListener(env.get_keymap())

observation, info = env.reset()
data = {}
reward_acc = 0
step_count = 0
for _ in range(50_000):
    action = keyboard_listener.get_action()
    # action = np.array(np.random.rand(4))
    # action = np.array([0, 0, 0, 0.5])
    observation, reward, terminated, truncated, info = env.step(action)
    step_count += 1
    # print(observation["gun"])
    # log(f"reward:{reward:.2f}")
    reward_acc += reward
    print("Reward:", reward, "Accumulated reward:", reward_acc, "Step:", step_count)
    time.sleep(0.01)

    if terminated:
        observation, info = env.reset()
        reward_acc = 0
        step_count = 0
