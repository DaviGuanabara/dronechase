from stable_baselines3 import PPO
import torch.nn.functional as F
from core.rl_framework.utils.io_data import IOData
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import time

import os
from typing import Dict, Optional, Tuple

import torch



from torch import nn, optim
from torch.utils.data import DataLoader


from torch.utils.data import Dataset, DataLoader, default_collate

from core.rl_framework.utils.io_data import IOData
from threatsense.level5.level5_dumb_multiobs import Level5DumbMultiObs
from threatsense.level5.level5_fusion_environment import Level5FusionEnvironment as Level5Fusion
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils as nn_utils
from collections import deque


class SupervisedImitationTrainer:
    """
    Collect, save and load data to/from HDF5 files.
    """

    def __init__(self, folder_path: str):
        self.set_folder_path(folder_path)

    def set_folder_path(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        self.folder_path = folder_path

        self.io_data = IOData(folder_path)
        self.loader = self.io_data.get_loader(batch_size=1024, shuffle=True)

    def kill_high_correlation(
        self, obs, target, device, rng: np.random.Generator, alpha: float = 0.1
    ):
        """
        Quebra correlação forte do last_action:
        - 20% completamente aleatório
        - 40% mantém a ação real
        - 40% adiciona ruído gaussiano (moderado)
        
        Args:
            obs (dict): observações do batch
            target (Tensor): ações alvo
            device (torch.device): dispositivo
            rng (np.random.Generator): gerador aleatório para reprodutibilidade
            alpha (float): fração do range da ação usada como desvio padrão (default=0.1)
        """
        batch_size = target.shape[0]
        p = rng.random()

        if p < 0.2:
            # 20% completamente aleatório
            obs["last_action"] = torch.empty(
                (batch_size, 4), dtype=torch.float32, device=device
            ).uniform_(-1, 1)

        elif p < 0.6:
            # 40% mantém a ação real
            obs["last_action"] = obs["last_action"].to(device)

        else:
            # 40% adiciona ruído gaussiano
            std = alpha * 2.0  # range = 2 se ações normalizadas [-1,1]
            noise = torch.from_numpy(rng.normal(
                0, std, size=(batch_size, 4))).float().to(device)
            obs["last_action"] = obs["last_action"].to(device) + noise
            obs["last_action"] = torch.clamp(
                obs["last_action"], -1, 1)  # garante range válido

        return obs


    def loss_mae_direction(self, command_pred, u_label, w_dir=0.8, w_mag=0.2, w_raw_direction = 0.1, eps=1e-8):
        """
        command_pred: (B,4) -> [dx, dy, dz, m_pred]
        u_label     : (B,4) -> [dx, dy, dz, m_true]
        """
        # split
        d_pred = command_pred[:, :3]           # (B,3) -> direction
        m_pred = command_pred[:, 3:4]          # (B,1) - magnitude
        d_true = u_label[:, :3]                # (B,3) - direction
        m_true = u_label[:, 3:4]               # (B,1) - magnitude

        # normaliza direções (independente de como venham no dataset)
        d_pred_hat = F.normalize(d_pred, dim=1, eps=eps)
        d_true_hat = F.normalize(d_true, dim=1, eps=eps)

        # 1) termo direcional: 1 - cos
        cos = (d_pred_hat * d_true_hat).sum(dim=1).clamp(-1.0, 1.0)
        L_dir = (1.0 - cos).mean()

        # 2) termo de magnitude: MAE entre m_pred e m_true
        L_mag = F.l1_loss(m_pred, m_true)

        L_raw_direction = F.l1_loss(d_pred, d_true)

        return w_dir*L_dir + w_mag*L_mag + w_raw_direction*L_raw_direction





# ==== Training Loop (supervised imitation) ====

    
    def train_loop(self, env, policy_kwargs, sample_size, n_runs, output_dir, epochs=1, batch_size=256, lr=1e-3, rng=None):
 
        
        rng = np.random.default_rng(42) if rng is None else rng
 

        for run in range(n_runs):
            run_name = f"train_sl_{run+1}"
            print(f"=== Run {run+1}/{n_runs}: {run_name} ===")

            model = PPO(
                "MultiInputPolicy",
                env,  # seu Level5Fusion ou outro env compatível
                policy_kwargs=policy_kwargs,
                verbose=1,
                learning_rate=3e-4,
            )

            loader = self.io_data.get_loader_limit(
                limit_size=sample_size, batch_size=batch_size, shuffle=True, rng=rng
            )

            print("Model Created")

            self.train_student_ppo_wrapped(model, loader, run_name, epochs=epochs,
                                           batch_size=batch_size, lr=lr, output_dir=output_dir, limit_size=500_000, rng=rng)
            print("====================================\n")

    def train_student_ppo_wrapped(
        self, model: PPO, loader, run_name, epochs=1,
        batch_size=256, lr=1e-3, output_dir="/train_sl", limit_size=500_000, rng=None):
        

        # === Dataset ===

        model, logs = self._train_ppo(model, loader, epochs=epochs, lr=lr, rng=rng, run_name=run_name)

        # === Salva modelo e logs ===
        run_output_dir = os.path.join(output_dir, run_name)
        self.create_directory(run_output_dir)

        model_path = os.path.join(run_output_dir, "model.zip")
        log_path = os.path.join(run_output_dir, "log.xlsx")

        model.save(model_path)

        log_df = pd.DataFrame(logs)
        log_df.to_excel(log_path, index=False)

        return model


    def _train_ppo(
        self, model: PPO, train_loader, rng, epochs=1, lr=1e-3, run_name="train_sl"):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(model.policy.parameters(), lr=lr)
        global_batch = 0

        logs = []
        loss_window = deque(maxlen=50)  # sempre guarda no máximo 50 valores

        for epoch in range(epochs):
            model.policy.train()
            total_loss = 0
            start_time = time.time()
            num_batches = len(train_loader)

            for i, (obs, target) in enumerate(train_loader, 1):
                global_batch += 1

                # Corrige last_action com ruído/dropout
                obs = self.kill_high_correlation(obs, target, device, rng=rng)

                obs = {k: v.to(model.policy.device) for k, v in obs.items()}
                target = target.to(model.policy.device)

                features = model.policy.extract_features(obs)
                latent_pi, _ = model.policy.mlp_extractor(features)
                pred_actions = model.policy.action_net(latent_pi)

                loss = self.loss_mae_direction(
                    pred_actions, target, w_mag=1, w_dir=100, w_raw_direction=60)

                optimizer.zero_grad()
                loss.backward()
                nn_utils.clip_grad_norm_(model.policy.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * target.size(0)

                loss_window.append(loss.item())  # adiciona no fim
                avg_loss_50 = sum(loss_window) / len(loss_window)  # média da janela

                # Tempo decorrido e ETA (por batch)
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                eta = avg_time * (num_batches - i)

                msg = (
                    f"Run: {run_name} | "
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{i}/{num_batches}] "
                    #f"Loss: {loss.item():.4f} "
                    f"Avg Loss (50): {avg_loss_50:.4f} "
                    f"Elapsed: {elapsed:.1f}s "
                    f"ETA: {eta/60:.1f}min"
                )


                print(msg.ljust(120), end="\r", flush=True)

                logs.append({
                    "epoch": epoch + 1,
                    "batch": i,
                    "global_batch": global_batch,
                    "loss": loss.item(),
                    "avg_loss_50": avg_loss_50,
                    "elapsed_time_sec": round(elapsed, 2),
                    "eta_min": round(eta / 60, 2)
                })

        # Salva histórico e modelo final
        return model, logs

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Directory {path} created.")