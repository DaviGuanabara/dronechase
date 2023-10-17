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

from loyalwingmen.modules.utils.keyboard_listener import KeyboardListener
from loyalwingmen.modules.environments.level2_rpm.level2_environment import Level2
from loyalwingmen.modules.utils.displaytext import log


env = Level2(GUI=True, rl_frequency=15, debug=True, interactive_mode=False)
# check_env(env)


keyboard_listener = KeyboardListener(env.get_keymap())

observation, info = env.reset()
data = {}
for _ in range(50_000):
    action = keyboard_listener.get_action()
    observation, reward, terminated, truncated, info = env.step(action)

    # log(f"reward:{reward:.2f}")
    # log(str(observation[:18]))

    log(string=f"action:{str(action)}")
    time.sleep(0.1)

    if terminated:
        observation, info = env.reset()
