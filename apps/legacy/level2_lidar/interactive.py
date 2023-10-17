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
from loyalwingmen.modules.environments.level1.level1_environment import Level1
from loyalwingmen.modules.utils.displaytext import log


env = Level1(GUI=True, rl_frequency=30, debug=True)

keyboard_listener = KeyboardListener(env.get_keymap())

observation, info = env.reset()
data = {}
for _ in range(50_000):
    action = keyboard_listener.get_action()
    observation, reward, terminated, truncated, info = env.step(action)

    # log(f"reward:{reward:.2f}")

    if terminated:
        observation, info = env.reset()
