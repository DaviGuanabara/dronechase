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

    def get_training_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def kill_high_correlation(self, obs, target, device, rng, alpha=0.1):
        """
        Makes last_action deliberately unreliable during supervised imitation.

        Distribution per sample:
        - 60% fully random action
        - 20% zero action
        - 20% noisy version of the real last_action
        - 0% clean keep

        This keeps the last_action input present during training, but prevents the
        model from solving the task by simply copying the previous action.
        """
        batch_size = target.shape[0]
        last_action = obs["last_action"].to(device=device, dtype=torch.float32)

        probs = rng.random(batch_size)
        mask_rand = probs < 0.6
        mask_zero = (probs >= 0.6) & (probs < 0.8)
        mask_noise = probs >= 0.8

        new_last_action = torch.zeros_like(last_action)

        if mask_rand.any():
            rand_count = int(mask_rand.sum())
            rand = torch.empty(
                (rand_count, last_action.shape[1]), device=device)
            rand[:, :3].uniform_(-1.0, 1.0)
            rand[:, 3:] = torch.rand((rand_count, 1), device=device)
            new_last_action[mask_rand] = rand

        if mask_zero.any():
            new_last_action[mask_zero] = 0.0

        if mask_noise.any():
            noise_count = int(mask_noise.sum())
            noise = torch.zeros(
                (noise_count, last_action.shape[1]), device=device)
            noise[:, :3] = torch.as_tensor(
                rng.normal(0.0, alpha, size=(noise_count, 3)),
                device=device,
                dtype=torch.float32,
            )
            noise[:, 3:] = torch.as_tensor(
                rng.normal(0.0, alpha * 0.5, size=(noise_count, 1)),
                device=device,
                dtype=torch.float32,
            )
            new_last_action[mask_noise] = last_action[mask_noise] + noise

        new_last_action[:, :3] = torch.clamp(new_last_action[:, :3], -1.0, 1.0)
        new_last_action[:, 3:] = torch.clamp(new_last_action[:, 3:], 0.0, 1.0)

        obs["last_action"] = new_last_action
        return obs

    def loss_mae_direction(self, command_pred, u_label, w_dir=0.8, w_mag=0.2, w_raw_direction=0.1, eps=1e-8):
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
        device = self.get_training_device()
        print(f"[SL] Training device: {device}")

        for run in range(n_runs):
            run_name = f"train_sl_{run+1}"
            print(f"=== Run {run+1}/{n_runs}: {run_name} ===")

            model = PPO(
                "MultiInputPolicy",
                env,  # seu Level5Fusion ou outro env compatível
                policy_kwargs=policy_kwargs,
                verbose=1,
                learning_rate=3e-4,
                device=device,
            )
            print(f"[SL] Model policy device: {model.policy.device}")

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

        model, logs = self._train_ppo(
            model, loader, epochs=epochs, lr=lr, rng=rng, run_name=run_name)

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

        device = model.policy.device
        print(f"[SL] _train_ppo using model.policy.device: {device}")
        optimizer = optim.Adam(model.policy.parameters(),  # type: ignore
                               lr=lr)  # type: ignore
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

                obs = {
                    k: v.to(device=device, dtype=torch.float32) if k != "validity_mask" else v.to(
                        device=device)
                    for k, v in obs.items()
                }
                target = target.to(device=device, dtype=torch.float32)

                features = model.policy.extract_features(obs)
                latent_pi, _ = model.policy.mlp_extractor(features)
                pred_actions = model.policy.action_net(latent_pi)

                loss = self.loss_mae_direction(
                    pred_actions, target, w_mag=1, w_dir=100, w_raw_direction=60)

                optimizer.zero_grad()
                loss.backward()
                nn_utils.clip_grad_norm_(
                    model.policy.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * target.size(0)

                loss_window.append(loss.item())  # adiciona no fim
                avg_loss_50 = sum(loss_window) / \
                    len(loss_window)  # média da janela

                # Tempo decorrido e ETA (por batch)
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                eta = avg_time * (num_batches - i)

                msg = (
                    f"Run: {run_name} | "
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{i}/{num_batches}] "
                    f"Loss: {loss.item():.4f} "
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
