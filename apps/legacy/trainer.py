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
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    ProgressBarCallback,
    BaseCallback
)
import math
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np


def main():
    
    
    #env = DroneChaseEnvLevel1(GUI=False, rl_frequency=30)
    #check_env(env, warn=True, skip_render_check=True)
    
    
    n_envs = 2 * math.ceil((os.cpu_count() or 1))
    env_fns = [lambda: DroneChaseEnv(GUI=False, rl_frequency=30) for i, _ in enumerate(range(n_envs))]
        
    vectorized_environment = SubprocVecEnv(env_fns)# type: ignore
    
    vectorized_environment = VecMonitor(vectorized_environment)
    
    #observation, info = vectorized_environment.reset()

    #action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(3), sigma=float(0.8) * np.ones(3))

    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU,
                         
                     net_arch=[1024, 512, 256]
                     )
    
    model = SAC(
        "MlpPolicy", 
        vectorized_environment, 
        policy_kwargs=policy_kwargs, 
        verbose=0, 
        learning_rate=1e-5,
     #   action_noise=action_noise
        )
    
    progressbar_callback = ProgressBarCallback()
    model.learn(total_timesteps=2_000_000, callback=progressbar_callback)
    
    model.save("./sac_randomized_drone_chase_env")


if __name__ == '__main__':
    # https://stackoverflow.com/questions/29690091/python2-7-exception-the-freeze-support-line-can-be-omitted-if-the-program
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!