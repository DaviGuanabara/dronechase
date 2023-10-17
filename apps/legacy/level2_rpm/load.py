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
from loyalwingmen.modules.environments.level2_rpm.level2_environment import Level2


# IT NEEDS TO BE FIXED BEFORE USE
env = Level2(GUI=True, rl_frequency=15)

# preciso corrigir o caminho do modelo
# model = PPO.load("./trained_level2_ppo")
model = PPO.load(
    "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\apps\\level2_rpm\\output\\baysian_optimizer_app\\level2_4.00M_end_to_end_NN_3\\models_dir\\h[512, 256, 1024, 1024, 128, 1024, 256, 512]-f30-lr0.001\\best_model.zip"
)

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
