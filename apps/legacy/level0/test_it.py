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
from loyalwingmen.modules.environments.level0.level0_environment import Level0
from loyalwingmen.modules.utils.displaytext import log
import numpy as np

rl_frequency = 48
env = Level0(GUI=True, rl_frequency=rl_frequency, debug=True)

keyboard_listener = KeyboardListener(env.get_keymap())

observation, info = env.reset()
data = {}
for _ in range(50_000):
    # action = keyboard_listener.get_action()

    action = np.array([-0.5, 1, 0, 0.99])
    print(f"action: {action}")
    for _ in range(int(240)):
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"observation: {observation}")
        # log(f"reward:{reward:.2f}_action:{action}")

    # observation, info = env.reset()

    # if terminated:
    #    observation, info = env.reset()
