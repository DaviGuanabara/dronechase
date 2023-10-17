import time
import os
import sys
sys.path.append("..")
from stable_baselines3 import PPO, SAC
from modules.factories.callback_factory import callbacklist
from modules.utils.Logger import Logger
from modules.utils.utils import sync, str2bool
import torch
from stable_baselines3.common.env_util import make_vec_env
import torch as th
#from modules.environments.simplified_env import DroneChaseEnvLevel1
#from modules.environments.randomized_drone_chase_env import RandomizedDroneChaseEnv
from modules.environments.drone_chase_env import DroneChaseEnv



env = DroneChaseEnv(GUI=True, rl_frequency=30)

model = SAC.load("./sac_randomized_drone_chase_env")

observation, info = env.reset(0)
for steps in range(50_000):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    
    
    #logging.debug(f"(main) reward: {reward}")
    print(f"reward:{reward:.2f} - action:{action}, observation:{observation}")
    if terminated:
        print("terminated")
        observation, info = env.reset(0)