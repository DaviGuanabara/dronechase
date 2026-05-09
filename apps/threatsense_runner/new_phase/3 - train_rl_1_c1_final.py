

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import DictRolloutBuffer
import logging
import warnings
from threatsense.level5.level5_c1_fusion_environment import Level5C1FusionEnvironment as Level5C1Fusion
from core.rl_framework.utils.pipeline import ReinforcementLearningPipeline, CallbackType
from core.rl_framework.utils.directory_manager import DirectoryManager
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# TODO: Colocar o train c1 no train c1, e o train c2 no train c2
# o train c1 tem um reward mais simples, que bonifica alcançar um alvo.
# e o train c2 tem um reward mais complexo, que bonifica o dano causado no alvo, e penaliza o suicídio do agente, por exemplo.


DATE = "01.05.2026"
TAG = "c1_2"
INPUT_MODEL = "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/sl/30.04.2026/train_sl_1/train_sl_1/model.zip"
# PARAMS
# model hyperparameters
N_EPOCHS = 10  # ROLLOUT_N_EPOCHS = 5
N_STEPS = 8192  # ROLLOUT_SIZE = 8192
BATCH_SIZE = 128


N_ENVS = 4
STEP_SIZE_SAVE = 100_000
N_EVAL_EPISODES = 100

SAVE_FREQ = STEP_SIZE_SAVE // N_ENVS

N_RUNS = 1
N_TIMESTEPS = 1_000_000

# Select the appropriate torch device (cuda, mps, or cpu)


def get_torch_device() -> torch.device:
    # if torch.cuda.is_available():
    #    return torch.device("cuda")
    # if torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("mps")
    # return torch.device("mps")


def train_fixed_ppo(
    base_model_path: str,
    vectorized_environment,
    output_logs_dir: str,
    n_timesteps: int,
    device: torch.device,
    callback_list,

):

    # Modelo PPO com parâmetros fixos
    model = PPO.load(
        base_model_path, env=vectorized_environment, device=device)

    # --- PPO tuning to reduce training bottleneck ---
    model.learning_rate = 1e-4
    # model.batch_size = 256          # larger batch → fewer minibatches
    # model.n_epochs = 1              # reduce number of passes over rollout

    model.n_epochs = N_EPOCHS
    model.n_steps = N_STEPS
    model.batch_size = BATCH_SIZE
    model.rollout_buffer = DictRolloutBuffer(
        model.n_steps,
        model.observation_space,  # type: ignore
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
    )

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
    print("LEVEL 5 C1 - Air Combat - PPO Fixed LR=1e-3")

    # n_timesteps = 1_000_000

    # Diretórios
    input_model = INPUT_MODEL

    base_dir = os.path.join(f"output/rlc1/{DATE}_{TAG}")
    DirectoryManager.create_directory(base_dir)

    # Criar ambiente vetorizado
    vectorized_environment = ReinforcementLearningPipeline.create_vectorized_environment(
        environment=Level5C1Fusion, env_kwargs={}, n_envs=N_ENVS
    )

    device = get_torch_device()
    logging.info(f"Using device: {device}")

    # Treinamento

    for i in range(N_RUNS):
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
            n_eval_episodes=N_EVAL_EPISODES,
            debug=True,
            save_freq=SAVE_FREQ,
            steps=N_TIMESTEPS
        )

        model = train_fixed_ppo(
            base_model_path=input_model,
            output_logs_dir=run_log_dir,
            n_timesteps=N_TIMESTEPS,
            callback_list=callback_list,
            vectorized_environment=vectorized_environment,
            device=device
        )

        model_path = os.path.join(run_models_dir, "final.zip")
        model.save(model_path)

        avg_reward, std_dev, num_episodes, episode_rewards = ReinforcementLearningPipeline.evaluate(
            model, vectorized_environment, n_eval_episodes=N_EVAL_EPISODES
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
            f"Treinamento {i+1}/{N_RUNS} concluído. Modelo salvo em {model_path}")


if __name__ == "__main__":
    main()
