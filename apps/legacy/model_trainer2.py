import os
import sys
sys.path.append("..")
import logging
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


from modules.environments.drone_chase_env import DroneChaseEnv
from modules.environments.randomized_drone_chase_env import RandomizedDroneChaseEnv
from modules.environments.drone_chase_level_1 import DroneChaseEnvLevel1

from modules.models.policy import CustomActorCriticPolicy, CustomCNN
from modules.factories.callback_factory import callbacklist, CallbackType
from typing import List, Tuple
from datetime import datetime
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from openpyxl import load_workbook, Workbook
from openpyxl.worksheet.worksheet import Worksheet
import warnings
from gymnasium import Env
from gymnasium import spaces, Env
from typing import Optional, Union
from apps.ml.pipeline import ReinforcementLearningPipeline
from apps.ml.directory_manager import DirectoryManager
import torch as th
from torch import backends
from sys import platform

def main():
    
    environment = DroneChaseEnvLevel1
    models_dir = "C:\\Users\\davi_\\Documents\\GitHub\\loyalWingmen\\loyalwingmen\\outputs\\Model_Trainer2\\level1"
    logs_dir = models_dir

    n_timesteps = 1_000_000
    suggestions = {}

    suggestions[f"hidden_{1}"] = 128
    suggestions[f"hidden_{2}"] = 128
    suggestions[f"hidden_{3}"] = 128


    suggestions["rl_frequency"] = 15
    suggestions["learning_rate"] = 1e-9
    suggestions["speed_amplification"] = .1

    suggestions["model"] = "PPO"

    hiddens = []
    for i in range(1, len(suggestions) + 1):
        key = f"hidden_{i}"
        if key in suggestions:
            hiddens.append(suggestions[key])
        else:
            break
        
        
    vectorized_environment: VecMonitor = ReinforcementLearningPipeline.create_vectorized_environment(environment=environment, env_kwargs=suggestions)#, n_envs=1)
    specific_model_folder = ReinforcementLearningPipeline.gen_specific_folder_path(hiddens, suggestions["rl_frequency"], suggestions["learning_rate"], dir=models_dir)
    specific_log_folder = ReinforcementLearningPipeline.gen_specific_folder_path(hiddens, suggestions["rl_frequency"], suggestions["learning_rate"], dir=logs_dir)

    callback_list = ReinforcementLearningPipeline.create_callback_list(vectorized_environment, model_dir=specific_model_folder, log_dir=specific_log_folder, callbacks_to_include=[CallbackType.EVAL, CallbackType.PROGRESSBAR], n_eval_episodes=20, debug=True)
    policy_kwargs = dict(
            net_arch=dict(pi=hiddens, vf=hiddens),
        )
    model = PPO(
            "MlpPolicy",
            vectorized_environment,
            verbose=1,
            device="auto",
            tensorboard_log=specific_model_folder,
            policy_kwargs=policy_kwargs,
            learning_rate=suggestions["learning_rate"],
        )
    logging.info(model.policy)
    model = ReinforcementLearningPipeline.train_model(model, callback_list, n_timesteps=n_timesteps)

    #avg_reward, std_dev, num_episodes = ReinforcementLearningPipeline.evaluate(model, vectorized_environment, n_eval_episodes=20)
    ReinforcementLearningPipeline.save_model(model, hiddens, suggestions["rl_frequency"], suggestions["learning_rate"], 0, 0, models_dir)
    
    env = environment(GUI=True, rl_frequency=suggestions["rl_frequency"], speed_amplification=suggestions["speed_amplification"], debug=True)
    
    observation, info = env.reset()

    for steps in range(50_000):
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        print(reward, action)

        if terminated:
            print("Episode terminated")
            env.reset()
    
if __name__ == '__main__':
    main()    