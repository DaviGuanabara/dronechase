
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
from multiprocessing import cpu_count
from stable_baselines3.common.vec_env import SubprocVecEnv
from modules.factories.callback_factory import callbacklist
from modules.models.policy import CustomActorCriticPolicy, CustomCNN
import torch.nn as nn
import torch as th
import math



# ===============================================================================
# Setup
# ===============================================================================

def create_output_folder(experiment_name: str):
    # Obter a data atual
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Criar o nome da pasta com base na data atual
    #folder_name = f"output/{experiment_name}"
    folder_name = os.path.join("output", experiment_name)

    # Criar a pasta se ainda não existir
    os.makedirs(folder_name, exist_ok=True)

    # Retornar o caminho completo para a pasta
    return folder_name


def main():
    
    

    
    
    device = get_device(device='auto')
    
    if os.name == 'posix':
        device = get_device(device='mps')
        
    if os.name == 'nt':
        device = get_device(device='cuda')
        
            
    output_folder = create_output_folder("demo_training2_app")
    model_path = os.path.join(output_folder, f"models")
    log_path = os.path.join(output_folder, f"logs")
    
    
    number_of_logical_cores = cpu_count()
    n_envs = number_of_logical_cores
    
    print("device", device)
    print("cpu_count", number_of_logical_cores)

    #env_fns = []
    #for _ in range(n_envs):
    #    env_fns.append(DroneChaseEnv)
        
    env_fns = [lambda: DroneChaseEnv(GUI=True) for _ in range(n_envs)]    

    vectorized_environment = SubprocVecEnv(env_fns) # type: ignore
    

    callback_list = callbacklist(
        vectorized_environment,
        log_path=log_path, #"/output/demo_training2_app/logs/",
        model_path=model_path, #"/output/demo_training2_app/models/",
        save_freq=100_000,
    )

    nn_t = [2048, 1024, 256, 2048]
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
        net_arch=dict(pi=nn_t, vf=nn_t)
    )
    # env = DemoEnvironment(GUI=False)

    model = PPO(
        CustomActorCriticPolicy,  # "CnnPolicy",
        vectorized_environment,
        verbose=0,
        device=device,
        tensorboard_log="./logs/my_first_env/",
        policy_kwargs=policy_kwargs,
        learning_rate=0.000149267851717988 #3 * math.pow(10, -4),
    )

    print(model.policy)
    model.learn(total_timesteps=1_000_000, callback=callback_list)
    model.save("demo_trained2_model")


if __name__ == '__main__':
    # https://stackoverflow.com/questions/29690091/python2-7-exception-the-freeze-support-line-can-be-omitted-if-the-program
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
