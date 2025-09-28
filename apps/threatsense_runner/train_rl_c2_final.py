import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

import torch
import os
import logging
import warnings

from core.rl_framework.agents.policies.ppo_policies import StudentPolicy
from threatsense.level5.level5_c1_fusion_environment import Level5C1FusionEnvironment as Level5C2Fusion
from core.rl_framework.utils.pipeline import ReinforcementLearningPipeline, CallbackType
from core.rl_framework.utils.directory_manager import DirectoryManager

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_fixed_ppo(
    base_model_path: str,
    vectorized_environment,
    output_logs_dir: str,
    n_timesteps: int,
    device: torch.device,
    callback_list,
    
):

    # Modelo PPO com parâmetros fixos
    model = PPO.load(base_model_path, env=vectorized_environment, device=device)
    model.learning_rate = 1e-3
    model.batch_size = 128

    new_logger = configure(output_logs_dir, ["csv", "tensorboard"])
    model.set_logger(new_logger)

    logging.info(model.policy)
    model.learn(
        total_timesteps=n_timesteps,
        callback=callback_list,
        tb_log_name="ppo_fixed_run",
    )

    return model
    

def main():
    print("LEVEL 5 C2 - Air Combat - PPO Fixed LR=1e-3")
    n_eval_episodes = 100
    n_timesteps = 1_000_000

    # Diretórios
    input_model = "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\output\\rl\\13.09.2025\\train_rl_8\\models\\best_model.zip"

    base_dir = os.path.join("output\\rl\\c2_21.09.2025")
    DirectoryManager.create_directory(base_dir)


    # Criar ambiente vetorizado
    vectorized_environment = ReinforcementLearningPipeline.create_vectorized_environment(
        environment=Level5C2Fusion, env_kwargs={}, n_envs=4
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Treinamento
    for i in range(10):
        run_dir = os.path.join(base_dir, f"train_rl_{i+1}")
        run_log_dir = os.path.join(run_dir, "log")
        run_models_dir = os.path.join(run_dir, "models")

        DirectoryManager.create_directory(run_log_dir)
        DirectoryManager.create_directory(run_models_dir)

        callback_list = ReinforcementLearningPipeline.create_callback_list(
            vectorized_environment,
            model_dir=run_models_dir,
            log_dir=run_log_dir,
            callbacks_to_include=[
                CallbackType.EVAL, CallbackType.CHECKPOINT, CallbackType.PROGRESSBAR],
            n_eval_episodes=n_eval_episodes,
            debug=True,
            step_size_save=100_000,
            steps=n_timesteps
        )


        model = train_fixed_ppo(
            base_model_path=input_model,
            output_logs_dir=run_log_dir,
            n_timesteps=n_timesteps,
            callback_list=callback_list,
            vectorized_environment=vectorized_environment,
            device=device
        )

        model_path = os.path.join(run_models_dir, "final.zip")
        model.save(model_path)

        
        
        avg_reward, std_dev, num_episodes, episode_rewards = ReinforcementLearningPipeline.evaluate(
            model, vectorized_environment, n_eval_episodes=n_eval_episodes
        )


        results = {
            "run": i+1,
            "avg_reward": avg_reward,
            "std_dev": std_dev,
            "episodes": num_episodes
        }
        ReinforcementLearningPipeline.save_results_to_excel(
            run_log_dir, "final_summary.xlsx", results
        )

        logging.info(
            f"Treinamento {i+1}/10 concluído. Modelo salvo em {model_path}")


if __name__ == "__main__":
    main()
