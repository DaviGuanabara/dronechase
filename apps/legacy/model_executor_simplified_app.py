
import sys
sys.path.append("..")

from datetime import datetime
import os

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from stable_baselines3 import PPO
from modules.environments.drone_chase_env import DroneChaseEnv
from modules.environments.randomized_drone_chase_env import RandomizedDroneChaseEnv
from modules.environments.randomized_drone_chase_env_action_fixed import RandomizedDroneChaseEnvFixed
from modules.environments.drone_chase_level_1 import DroneChaseEnvLevel1


from multiprocessing import cpu_count
from stable_baselines3.common.vec_env import SubprocVecEnv
from modules.factories.callback_factory import callbacklist
from modules.models.policy import CustomActorCriticPolicy, CustomCNN
import torch.nn as nn
import torch as th
import math


def main():
    
    

 
    #selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\baysian_optimizer_app.py\\ULTIMO_no_physics_2.00M_steps_lidar_range_100m_16_20s\\models\\h[512, 512, 128]-f15-lr0.0001\\mPPO-r7849.205078125-sd763.091552734375"
    selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\baysian_optimizer_app.py\\ULTIMO_RANDOM_no_physics_2.00M_steps_lidar_range_100m_16_20s\\models\\h[256, 256, 512]-f10-lr1e-09\\mPPO-r74375928.0-sd740232064.0.zip"
    selected_zip = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\baysian_optimizer_app.py\\ULTIMO_RANDOM_no_physics_1.00M_steps_lidar_range_20m_32_20s\models\h[256, 256, 256]-f2-lr1e-07\mPPO-r-750122496.0-sd284461344.0.zip"
    model = PPO.load(selected_zip)
    env = RandomizedDroneChaseEnvFixed(GUI=True, rl_frequency=10, speed_amplification=1, debug=True)
    
    observation, info = env.reset()

    for steps in range(50_000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"{reward:.2f}, {action}")

        if terminated:
            print("Episode terminated")
            env.reset()


if __name__ == '__main__':
    # https://stackoverflow.com/questions/29690091/python2-7-exception-the-freeze-support-line-can-be-omitted-if-the-program
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
