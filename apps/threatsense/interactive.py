import sys
import os
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)



print("O problema não é aqui")
from core.utils.keyboard_listener import KeyboardListener

print("Importing Level 5 environment...")
from threatsense.level5.level5_envrionment import Level5Environment as Level5
print("Level 5 environment imported successfully.")

print("Initializing Level 5 environment...")
env = Level5(GUI=True, rl_frequency=15)
print("Level 5 environment initialized.")
print("Environment class:", env.__class__.__name__)

# check_env(env)

print("Interactive mode started. Press 'q' to quit.")
print("Available actions:", env.get_keymap())
print("Level 5 environment initialized.")

keyboard_listener = KeyboardListener(env.get_keymap())


observation, info = env.reset()
data = {}
for _ in range(50_000):
    action = keyboard_listener.get_action()
    observation, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.1)

    if terminated:
        observation, info = env.reset()