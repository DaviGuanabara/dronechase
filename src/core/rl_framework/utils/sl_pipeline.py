import torch.nn.functional as F
from core.rl_framework.utils.io_data import IOData, MultiH5Dataset
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import time
import glob
import os
from typing import Dict, Optional, Tuple
import h5py
import torch
from torch.utils.data import Dataset

from core.rl_framework.agents.policies.fused_lidar_extractor import FLIAExtractor
from torch import nn, optim
from torch.utils.data import DataLoader


from torch.utils.data import Dataset, DataLoader, default_collate
from typing import List, Dict, Tuple
from core.rl_framework.utils.io_data import IOData
from threatsense.level5.level5_dumb_multiobs import Level5DumbMultiObs
from threatsense.level5.level5_fusion_environment import Level5FusionEnvironment as Level5Fusion
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

class StudentWithFLIA(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        self.feature_extractor = FLIAExtractor(
            observation_space, features_dim=512)

        self.mlp_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 4)  # output final = ação
        )

    def forward(self, obs):
        features = self.feature_extractor(obs)   # [B, 512]
        return self.mlp_head(features)           # [B, action_dim]


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

    

    def get_train_val_loaders(self, dataset, batch_size=1024, val_ratio=0.1):
        n = len(dataset)
        val_size = int(n * val_ratio)
        train_size = n - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        return train_loader, val_loader


    
    def train_student(self, folder_path, observation_space, action_dim=4, epochs=1,
                      batch_size=256, lr=1e-3, model_output_dir="models_all", limit_size=500_000):
        print("Iniciando treinamento do DATASET COMPLETO student...")

        io_data = IOData(folder_path)
        dataset = io_data.dataset
        n = len(dataset)

        # pega apenas 500 mil amostras aleatórias
        indices = np.random.choice(n, size=limit_size, replace=False)
        subset = Subset(dataset, indices)

        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True, persistent_workers=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StudentWithFLIA(observation_space).to(device)

        self._train(model, device, loader, val_loader=None,
                    epochs=epochs, lr=lr, fold_id=0, output_dir=model_output_dir)

        return model

    def save_model(self, model, folder_path, id, process_name="all"):
        os.makedirs(os.path.join(folder_path, "models"), exist_ok=True)
        model_path = os.path.join(
            folder_path, f"models/student_{process_name}_{id}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

    def _early_stop_mode_evaluation(self, model, criterion, best_loss, avg_train_loss, fold_id, epoch, device, val_loader, output_dir, patience, epochs_no_improve):
        # === Avaliação no fold de teste ===

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for obs, target in val_loader:
                obs = {k: v.to(device) for k, v in obs.items()}
                target = target.to(device)
                pred = model(obs)
                loss = criterion(pred, target)
                total_val_loss += loss.item() * target.size(0)

        avg_val_loss = total_val_loss / \
            len(val_loader.dataset)  # type: ignore
        print(f"[Fold {fold_id}][Epoch {epoch+1}] "
                f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # === Early stopping check ===

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            self.save_model(model, output_dir, id=fold_id,
                            process_name="best_model")
        else:
            epochs_no_improve += 1
            print(
                f"[Fold {fold_id}] No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(
                f"[Fold {fold_id}] Early stopping triggered at epoch {epoch+1}.")
            return True, best_loss, epochs_no_improve, avg_val_loss

        return False, best_loss, epochs_no_improve, avg_val_loss

    from torch.optim.lr_scheduler import ReduceLROnPlateau


    def _train(self, model, device, train_loader, val_loader: Optional[DataLoader] = None, epochs=1, lr=1e-3, fold_id=0, output_dir="models_cv", patience=5, validation=False, GUI=False, sched_patience_batches=200):

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, factor=0.2, verbose=True)

        history = []
        global_batch = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            start_time = time.time()
            num_batches = len(train_loader)

            n_batch_loss = 0
            for i, (obs, target) in enumerate(train_loader, 1):
                

                if obs["validity_mask"].any() == 0:
                    #print("Observação inválida, pulando observação...")
                    continue  # pula batchs inválidos
                
                batch_size = target.shape[0]  # ou pegue de outro tensor de obs


                obs["last_action"] = torch.empty(
                    (batch_size, 4), dtype=torch.float32, device=device
                ).uniform_(-1, 1)
            
                global_batch += 1
                obs = {k: v.to(device) for k, v in obs.items()}
                target = target.to(device)

                optimizer.zero_grad()
                pred = model(obs)
                loss = self.loss_mae_direction(pred, target, w_mag=0.1, w_dir=10, w_raw_direction=6)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * target.size(0)
                n_batch_loss += loss.item()

                # === Scheduler step a cada N batches ===
                if global_batch % sched_patience_batches == 0:
                    print(
                        f"Scheduler step at global batch {global_batch} with loss {n_batch_loss / sched_patience_batches:.6f}")
                    scheduler.step(n_batch_loss / sched_patience_batches)
                    print(
                        f"Learning rate updated to: {optimizer.param_groups[0]['lr']}")
                    n_batch_loss = 0

                # ETA logging
                if i % 50 == 0 or i == num_batches:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / i
                    eta = avg_time * (num_batches - i)
                    print(f"[Fold {fold_id}][Epoch {epoch+1}] Batch {i}/{num_batches} " f"Loss: {loss.item():.6f} | ETA: {eta/60:.1f} min")
                    print(f"Last target (u): {target[0].cpu().numpy()} | Last Pred: {pred[0].detach().cpu().numpy()}")



            # === Validação ===


            history.append({
                "epoch": epoch + 1,
                "train_loss": total_loss / len(train_loader.dataset),

            })

            self.save_model(model, output_dir, id=epoch + 1,
                            process_name=f"train_epoch{epoch+1}")

            if validation:
                self.evaluate_in_env(model, env=None, gui=GUI, max_steps=500)


        # Salva histórico
        df = pd.DataFrame(history)
        csv_path = os.path.join(output_dir, f"history_fold{fold_id+1}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Fold {fold_id}] Histórico salvo em {csv_path}")
        self.save_model(model, output_dir, id=fold_id, process_name="last_model")
        return float("inf")


    def evaluate_in_env(self, model, env, gui, max_steps=1000):
        env = Level5Fusion(GUI=gui) if env is None else env
        model.eval()
        device = next(model.parameters()).device
        rewards = []

        obs, info = env.reset()
        total_reward = 0
        for step in range(max_steps):
            with torch.no_grad():
                action = model({k: torch.tensor(v).unsqueeze(0).to(device)
                                for k, v in obs.items()}).cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
    

        avg_reward = np.mean(rewards)
        print(f"[Eval] total_reward {total_reward}")
        return avg_reward


    def cross_validate(self, observation_space, k_folds=5, epochs=10, batch_size=256, lr=1e-3, model_output_dir="models_cv"):

        loaders = self.io_data.cross_validation_loaders(k_folds, batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        results = []
        for fold, (train_loader, test_loader) in enumerate(loaders):
            print(f"\n===== Iniciando fold {fold+1}/{k_folds} =====")
            os.makedirs(self.folder_path, exist_ok=True)
            output_dir = os.path.join(self.folder_path, model_output_dir)
            os.makedirs(output_dir, exist_ok=True)



            model = StudentWithFLIA(observation_space).to(device)
            val_loss = self._train(model, train_loader=train_loader, val_loader=test_loader, device=device,
                                   epochs=epochs, lr=lr, fold_id=fold, output_dir=output_dir)

            # Salva modelo
            model_path = os.path.join(output_dir, f"student_fold{fold+1}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"[Fold {fold}] Modelo salvo em {model_path}")

            results.append(val_loss)

            print("\n===== Cross-validation concluída =====")
            print("Resultados por fold:", results)
            print(
                f"Média: {np.mean(results):.6f} | Desvio-padrão: {np.std(results):.6f}")

        # Salva métricas gerais
        np.save(os.path.join(output_dir, "results.npy"), results)

    def complete_training(self, observation_space, epochs=10, batch_size=256, lr=1e-3, model_output_dir="models_all"):
        cross = self.cross_validate(observation_space, k_folds=5, epochs=epochs, batch_size=batch_size, lr=lr, model_output_dir=model_output_dir)
        student = self.train_student(self.folder_path, observation_space, epochs=epochs, batch_size=batch_size, lr=lr, model_output_dir=model_output_dir, validation=True, GUI=True)

        return cross, student

