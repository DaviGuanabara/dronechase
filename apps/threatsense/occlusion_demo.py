from core.utils.keyboard_listener import KeyboardListener
import sys
import os
import time


#current_directory = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(current_directory)

#parent_directory = os.path.dirname(current_directory)
#sys.path.append(parent_directory)

#grand_parent_directory = os.path.dirname(parent_directory)
#sys.path.append(grand_parent_directory)



print("Importing Level 5 Occlusion Demo...")
from threatsense.level5.level5_occlusion_demo import Level5OcclusionDemo as OcclusionDemo
print("Level 5 Occlusion Demo imported successfully.")

env = OcclusionDemo(GUI=True, rl_frequency=15)


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