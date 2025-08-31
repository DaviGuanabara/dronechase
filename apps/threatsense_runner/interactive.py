import sys
import os
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)



from core.utils.keyboard_listener import KeyboardListener
from threatsense.level5.level5_fusion_environment import Level5FusionEnvironment as Level5Fusion
print("Level 5 Fusion imported successfully.")

print("Initializing Level 5 environment...")
env = Level5Fusion(GUI=True, rl_frequency=15)
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