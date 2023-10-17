import time
import os
import sys

sys.path.append("..")
from stable_baselines3 import PPO, SAC

from loyalwingmen.rl_tools.callback_factory import callbacklist

from loyalwingmen.modules.utils.Logger import Logger
from loyalwingmen.modules.utils.utils import sync, str2bool
import torch
from stable_baselines3.common.env_util import make_vec_env
import torch as th
from loyalwingmen.modules.environments.level1.level1_environment import Level1


# IT NEEDS TO BE FIXED BEFORE USE
env = Level1(GUI=True, rl_frequency=30)

# preciso corrigir o caminho do modelo
model = PPO.load("./trained_level1_ppo")

observation, info = env.reset(0)
for steps in range(50_000):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.1)

    # logging.debug(f"(main) reward: {reward}")
    print(f"reward:{reward:.2f} - action:{action}")
    if terminated:
        print("terminated")
        observation, info = env.reset(0)
