# migrate_ppo.py
import os
import torch
from typing import Iterable, Optional, Union, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium import spaces

import torch
from torch import nn

from pathlib import Path


from core.rl_framework.utils.directory_manager import DirectoryManager
from gymnasium import spaces  # para type-guard do observation_space
from core.rl_framework.utils.pipeline import (
    ReinforcementLearningPipeline,
    CallbackType,
)
from stable_baselines3 import PPO
from core.rl_framework.agents.policies.fused_lidar_extractor import FLIAExtractor
from threatsense.level5.level5_fusion_environment import Level5FusionEnvironment

from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor

# === ajuste esses imports de acordo com seu projeto ===
from threatsense.level5.level5_fusion_environment import Level5FusionEnvironment as Level5_Fusion

# --- antes do PPO.load(...), coloque este bloco de compat ---
import sys, importlib, logging

def alias(old: str, new: str):
    try:
        sys.modules[old] = importlib.import_module(new)
        logging.info(f"[compat] {old} -> {new}")
    except Exception as e:
        logging.warning(f"[compat] Falhou alias {old}->{new}: {e}")

def directories(study_name: str):
    app_name, _ = os.path.splitext(os.path.basename(__file__))
    output_folder = os.path.join("output", app_name, study_name)
    print(output_folder)
    DirectoryManager.create_directory(output_folder)

    return output_folder

# === Caminho do modelo base (Exp03) ===
BASE_MODEL_PATH = Path(
    "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\experiment_results\\threat_engage\\Etapa 03 - Level 4\\EXP03\\Treinamento - 1RL 1BT - EXP03\\31_01_2024_level4_2.00M_exp03_vFinal\\Trial_8\\models_dir\\t8_PPO_r6995.40.zip"
)


PolicySpec = Union[str, type]  # "MlpPolicy", "MultiInputPolicy" ou a classe

def migrate_and_freeze_ppo(
    old_model_path: str,
    env,
    new_policy: PolicySpec,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    device: str = "auto",
    # tudo que começar com esses prefixos NÃO será carregado do modelo antigo
    skip_prefixes: Iterable[str] = (
        "features_extractor.lidar_feature_extractor",  # extrator antigo de LiDAR
    ),
    # tudo que começar com esses prefixos FICARÁ treinável no novo modelo
    trainable_prefixes: Iterable[str] = (
        "features_extractor.lidar_feature_extractor",  # extrator novo de LiDAR
    ),
    learning_rate: Optional[float] = None,  # se None, usa do PPO
) -> PPO:
    """
    Cria um PPO novo, carrega pesos compatíveis do PPO antigo, mas
    - ignora (não copia) módulos com `skip_prefixes`;
    - congela tudo que foi copiado;
    - deixa treinável apenas o que casar com `trainable_prefixes`;
    - recria o otimizador só com os parâmetros treináveis.

    Retorna o PPO pronto pra `learn()`.
    """
    # 1) carrega o modelo antigo (sem acoplar env)
    alias('loyalwingmen', 'core')  # mantenha se o zip referenciar esse prefixo
    old_model: BaseAlgorithm = PPO.load(old_model_path, device=device)

    # 2) cria o modelo novo já com o env e a policy desejada
    new_model = PPO(
        new_policy,
        env,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs or {},
    )

    # sanity-check: o env precisa ser Dict se a policy/extractor novo espera Dict
    if hasattr(new_model.observation_space, "spaces"):
        assert isinstance(new_model.observation_space, spaces.Dict), \
            "Sua policy nova espera Dict observation_space."

    # 3) prepara dicionários de pesos
    old_state = old_model.policy.state_dict()
    new_state = new_model.policy.state_dict()

    # helper para decidir se uma key deve ser pulada
    def _skip_key(name: str) -> bool:
        return any(name.startswith(pref) for pref in skip_prefixes)

    transferred, skipped = [], []
    for k, v in old_state.items():
        if _skip_key(k):
            skipped.append(k)
            continue
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v.clone()
            transferred.append(k)
        else:
            skipped.append(k)

    # 4) injeta o state_dict (strict=False para aceitar faltantes/diferentes)
    new_model.policy.load_state_dict(new_state, strict=False)

    # 5) congela os copiáveis por default…
    for name, param in new_model.policy.named_parameters():
        param.requires_grad = False

    # …e libera apenas os módulos marcados como treináveis
    def _mark_trainable(name: str) -> bool:
        return any(name.startswith(pref) for pref in trainable_prefixes)

    for name, param in new_model.policy.named_parameters():
        if _mark_trainable(name):
            param.requires_grad = True

    # 6) recria o otimizador só com os params treináveis
    trainable_params = [p for p in new_model.policy.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("Nenhum parâmetro treinável após a migração. Ajuste 'trainable_prefixes'.")

    lr = float(learning_rate) if learning_rate is not None else 0.01
    new_model.policy.optimizer = torch.optim.Adam(trainable_params, lr=lr)

    # 7) dry-run com um sample do obs_space (pega erro de chave/shape cedo)
    with torch.no_grad():
        sample = env.observation_space.sample()
        sample_t, _ = new_model.policy.obs_to_tensor(sample)
        _ = new_model.policy.extract_features(sample_t)

    print(f"[Migration] transferidos: {len(transferred)} | ignorados: {len(skipped)}")
    return new_model

def main():
    study_name = f"01_09_2025_level5_{2_000_000:.2f}M_level5_fusion_1"

    print(f"number of trials: {1}")
    print(f"number of timesteps: {2_000_000}M")
    print(f"study name: {study_name}")

    output_folder = directories(study_name)

    specific_output_folder = os.path.join(output_folder, f"Trial_{1}")
    specific_model_folder = os.path.join(specific_output_folder, "models_dir")
    specific_log_folder = os.path.join(specific_output_folder, "logs_dir")

    DirectoryManager.create_directory(specific_output_folder)
    DirectoryManager.create_directory(specific_model_folder)
    DirectoryManager.create_directory(specific_log_folder)

    # ✅ Criar ambiente vetorizado
    vectorized_environment: VecMonitor = ReinforcementLearningPipeline.create_vectorized_environment(
        environment=Level5_Fusion, n_envs=6
    )

    # ✅ Callbacks
    callback_list = ReinforcementLearningPipeline.create_callback_list(
        vectorized_environment,
        model_dir=specific_model_folder,
        log_dir=specific_log_folder,
        callbacks_to_include=[CallbackType.EVAL, CallbackType.PROGRESSBAR],
        n_eval_episodes=100,
        debug=True,
    )

    policy_kwargs = dict(
        features_extractor_class=FLIAExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )

    new_model = migrate_and_freeze_ppo(
        old_model_path=str(BASE_MODEL_PATH),
        env=vectorized_environment,
        new_policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        device="cuda",
        skip_prefixes=("features_extractor.lidar_feature_extractor",), #→ “não copie esses pesos do modelo antigo” (normalmente módulos incompatíveis).
        trainable_prefixes=("features_extractor.lidar_feature_extractor",), #→ “deixe esses módulos novos treináveis” (normalmente o novo FLIAExtractor).
        learning_rate=1e-3,
    )

    # ✅ Treinar
    new_model.learn(total_timesteps=2_000_000, tb_log_name="stage1_first_run", callback=callback_list)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()   # ⚠️ necessário no Windows
    main()
