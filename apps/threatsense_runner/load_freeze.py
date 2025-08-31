"""
Base Model Choice for ThreatSense – Experiment 03

We adopt the best model generated in Experiment 03 as the base for ThreatSense.

Rationale:
- Experiment 03 trained the agent in scenarios where the ally followed a Behavior Tree,
  i.e., the ally was moving during training.
- This forced the policy to learn more generalizable embeddings, since the agent had to
  disambiguate relative movement (own velocity vs. ally velocity) without explicit temporal
  cues in the LiDAR.
- In contrast, Experiment 04 trained with the ally static and tested with the ally moving.
  While it achieved higher mean performance, it introduced a bias: the agent often relied
  on spurious correlations tied to a static ally.
- Experiment 07 confirmed this fragility: using the Exp. 04 model in static-ally tests
  resulted in lower mean performance with very high variance (occasionally good, often poor).
- Therefore, the Exp. 03 model is more robust and stable, making it a stronger foundation
  for the new architecture with temporal LiDAR fusion, which can now resolve the temporal
  ambiguity that limited Exp. 03.

Conclusion:
→ Using Experiment 03 as the base balances generalization and stability,
  providing a reliable starting point for ThreatSense.
"""

import os
import logging
import warnings
from typing import Tuple

import numpy as np
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

import torch
from torch import nn

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor

# === ajuste esses imports de acordo com seu projeto ===
from threatsense.level5.level5_fusion_environment import Level5FusionEnvironment as Level5_Fusion

from core.rl_framework.utils.pipeline import (
    ReinforcementLearningPipeline,
    CallbackType,
)
from core.rl_framework.utils.directory_manager import DirectoryManager

from core.rl_framework.agents.policies.fused_lidar_extractor import FLIAExtractor  # <--
# importe o seu FLIAExtractor (ajuste o path/construtor conforme seu projeto)

from gymnasium import spaces  # para type-guard do observation_space


# --- antes do PPO.load(...), coloque este bloco de compat ---
import sys, importlib, logging

def alias(old: str, new: str):
    try:
        sys.modules[old] = importlib.import_module(new)
        logging.info(f"[compat] {old} -> {new}")
    except Exception as e:
        logging.warning(f"[compat] Falhou alias {old}->{new}: {e}")



# === Caminho do modelo base (Exp03) ===
BASE_MODEL_PATH = Path(
    "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\experiment_results\\threat_engage\\Etapa 03 - Level 4\\EXP03\\Treinamento - 1RL 1BT - EXP03\\31_01_2024_level4_2.00M_exp03_vFinal\\Trial_8\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t8_PPO_r6995.40.zip"
)

# === Config geral de logs/avisos ===
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# ------------------------------------------------------------
# Utilitário: substitui extractor, congela tudo exceto LiDAR, recria optimizer
# ------------------------------------------------------------
def swap_extractor_and_freeze(model: PPO, environment, learning_rate: float, features_dim: int = 512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) type-guard
    obs_space = environment.observation_space
    assert isinstance(
        obs_space, spaces.Dict), "Esperado observation_space do tipo gym.spaces.Dict"

    # 2) log informativo caso mude o features_dim
    old_feat_dim = int(
        getattr(model.policy.features_extractor, "features_dim", features_dim))
    #if old_feat_dim != features_dim:
        #logging.warning(f"[FLIA] features_dim salvo={old_feat_dim} difere do solicitado={features_dim}.")

    # 3) injeta novo extractor
    new_extractor = FLIAExtractor(
        observation_space=obs_space, features_dim=features_dim).to(device)
    model.policy.features_extractor = new_extractor
    # <- garante treino no módulo que vamos otimizar
    model.policy.features_extractor.train()

    # 4) congela tudo, exceto o LiDAR do FLIA
    for name, p in model.policy.named_parameters():
        if name.startswith("features_extractor.lidar_feature_extractor"):
            p.requires_grad = True
        # (opcional) para adaptar a fusão, libere também:
        # elif name.startswith("features_extractor.final_layer"):
        #     p.requires_grad = True
        else:
            p.requires_grad = False

    # módulos congelados em eval
    if hasattr(model.policy, "mlp_extractor"):
        model.policy.mlp_extractor.eval()
    if hasattr(model.policy, "action_net"):
        model.policy.action_net.eval()
    if hasattr(model.policy, "value_net"):
        model.policy.value_net.eval()

    # 5) recria o otimizador só com params treináveis (LR constante)
    trainable_params = [
        p for p in model.policy.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            "Nenhum parâmetro com requires_grad=True após o congelamento.")
    model.policy.optimizer = torch.optim.Adam(
        trainable_params, lr=float(learning_rate))

    # 6) dry-run para pegar problemas de shape cedo
    with torch.no_grad():
        sample = obs_space.sample()
        sample_t, _ = model.policy.obs_to_tensor(sample)
        _ = model.policy.features_extractor(sample_t)

    #logging.info("Parâmetros treináveis após swap:")
    #for n, p in model.policy.named_parameters():
    #    if p.requires_grad:
    #        logging.info("  • %s", n)


# ------------------------------------------------------------
# Optuna helpers
# ------------------------------------------------------------
def suggest_parameters(trial: Trial) -> dict:
    suggestions = {}
    # 1e-4 a 1e-3
    suggestions["learning_rate"] = 10 ** trial.suggest_int("exponent", -4, -3)
    suggestions["batch_size"] = trial.suggest_categorical(
        "batch_size", [128, 256, 512, 1024, 2048])
    suggestions["rl_frequency"] = trial.suggest_categorical(
        "rl_frequency", [15])
    return suggestions


def log_suggested_parameters(suggestions: dict):
    info_message = "Suggested Parameters:\n" + \
        "".join([f"  - {k}: {v}\n" for k, v in suggestions.items()])
    logging.info(info_message)


def get_hiddens(suggestion: dict):
    hiddens = []
    for i in range(1, len(suggestion) + 1):
        key = f"hidden_{i}"
        if key in suggestion:
            hiddens.append(suggestion[key])
        else:
            break
    return hiddens


def directories(study_name: str):
    app_name = "level5_fusion_optuna"
    output_folder = os.path.join("output", app_name, study_name)
    DirectoryManager.create_directory(output_folder)
    return output_folder


# ------------------------------------------------------------
# Pipelines
# ------------------------------------------------------------
def rl_pipeline(
    suggestions: dict,
    n_timesteps: int,
    models_dir: str,
    logs_dir: str,
    n_eval_episodes: int = 100,
    trial_number: int = 0,
) -> Tuple[float, float, float]:

    frequency = suggestions["rl_frequency"]
    learning_rate = suggestions["learning_rate"]
    hiddens = get_hiddens(suggestions)

    # Cria env
    vectorized_environment: VecMonitor = ReinforcementLearningPipeline.create_vectorized_environment(
        environment=Level5_Fusion, env_kwargs=suggestions
    )

    # Callbacks
    callback_list = ReinforcementLearningPipeline.create_callback_list(
        vectorized_environment,
        model_dir=models_dir,
        log_dir=logs_dir,
        callbacks_to_include=[CallbackType.EVAL, CallbackType.PROGRESSBAR],
        n_eval_episodes=n_eval_episodes,
        debug=True,
    )

    # Carrega PPO base (Exp03)


    # mapeia prefixos que existiam no modelo salvo para os caminhos atuais:
    # --- compat para módulos antigos salvos no zip ---
    alias('loyalwingmen', 'core')  # mantenha se o zip referenciar esse prefixo

    # lr constante no objeto salvo (evita desserializar função antiga)
    custom_objects = {
        "lr_schedule": (lambda _: suggestions["learning_rate"]),
    }

    # 1) CARREGAR SEM ENV PARA EVITAR O CHECK DE SPACES
    model = PPO.load(str(BASE_MODEL_PATH), custom_objects=custom_objects)


    # 2) ACOPLAR O NOVO ENV SEM CHECK DE SPACES
    model.env = vectorized_environment
    model.observation_space = vectorized_environment.observation_space
    model.action_space = vectorized_environment.action_space
    model.policy.observation_space = vectorized_environment.observation_space


    # 3) AJUSTES DE TREINO
    model.learning_rate = suggestions["learning_rate"]   # constante
    model.batch_size = suggestions["batch_size"]
    model.tensorboard_log = logs_dir
    new_logger = configure(logs_dir, ["csv", "tensorboard"])
    model.set_logger(new_logger)

    # 4) TROCAR EXTRACTOR + CONGELAR TUDO EXCETO LIDAR
    swap_extractor_and_freeze(
        model,
        vectorized_environment,
        learning_rate=suggestions["learning_rate"],
        features_dim=512
    )


    # Treina (apenas o LiDAR extractor recebe gradiente)
    #logging.info(model.policy)
    model.learn(
        total_timesteps=n_timesteps,
        callback=callback_list,
        tb_log_name="stage1_first_run",
    )

    # Avalia e salva
    avg_reward, std_dev, num_episodes, episode_rewards = ReinforcementLearningPipeline.evaluate(
        model, vectorized_environment, n_eval_episodes=n_eval_episodes
    )
    ReinforcementLearningPipeline.save_evaluation(episode_rewards, logs_dir)
    ReinforcementLearningPipeline.save_model(
        model,
        hiddens,
        frequency,
        learning_rate,
        avg_reward,
        std_dev,
        models_dir,
        trial_number=trial_number,
    )
    return avg_reward, std_dev, num_episodes


def objective(
    trial: Trial,
    n_timesteps: int,
    study_name: str,
    output_folder: str,
) -> float:
    suggestions: dict = suggest_parameters(trial)
    #log_suggested_parameters(suggestions)

    specific_output_folder = os.path.join(
        output_folder, f"Trial_{trial.number}")
    specific_model_folder = os.path.join(specific_output_folder, "models_dir")
    specific_log_folder = os.path.join(specific_output_folder, "logs_dir")

    DirectoryManager.create_directory(specific_output_folder)
    DirectoryManager.create_directory(specific_model_folder)
    DirectoryManager.create_directory(specific_log_folder)

    print(f"Directory {specific_output_folder} created")
    print(f"Directory {specific_model_folder} created")
    print(f"Directory {specific_log_folder} created")

    avg_score, std_deviation, n_episodes = rl_pipeline(
        suggestions,
        n_timesteps=n_timesteps,
        models_dir=specific_model_folder,
        logs_dir=specific_log_folder,
        n_eval_episodes=100,
        trial_number=trial.number,
    )
    logging.info(f"Avg score: {avg_score}")

    # salva resultados do trial
    suggestions["avg_score"] = avg_score
    suggestions["std_deviation"] = std_deviation
    suggestions["trial_number"] = trial.number

    ReinforcementLearningPipeline.save_results_to_excel(
        output_folder, f"results_{study_name}.xlsx", suggestions
    )

    return avg_score


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("Level 5 - Fusion - Air Combat - 1RL and 10BT")
    n_trials = 10
    n_timesteps = 2_000_000
    n_timesteps_in_millions = n_timesteps / 1e6
    study_name = f"31_08_2025_level5_{n_timesteps_in_millions:.2f}M_level5_fusion_3"

    print("Baysian Optimizer")
    print(f"number of trials: {n_trials}")
    print(f"number of timesteps: {n_timesteps_in_millions}M")
    print(f"study name: {study_name}")

    output_folder = directories(study_name)

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(), study_name=study_name
    )

    study.optimize(
        lambda trial: objective(
            trial,
            n_timesteps,
            study_name,
            output_folder,
        ),
        n_trials=n_trials,
    )

    # exporta CSV com histórico dos trials
    trial_data_save_path_file = os.path.join(
        output_folder, "optuna_study_results.csv")
    trial_data = study.trials_dataframe(
        attrs=("number", "value", "params", "state"))
    trial_data.to_csv(trial_data_save_path_file, index=False)


if __name__ == "__main__":
    main()
