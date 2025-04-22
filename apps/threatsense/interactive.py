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


from core.utils.keyboard_listener import KeyboardListener
from threatsense.level5.level5_envrionment import Level5Environment as Level5



env = Level5(GUI=True, rl_frequency=15)
# check_env(env)


keyboard_listener = KeyboardListener(env.get_keymap())

observation, info = env.reset()
data = {}
for _ in range(50_000):
    action = keyboard_listener.get_action()
    observation, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.1)

    if terminated:
        observation, info = env.reset()