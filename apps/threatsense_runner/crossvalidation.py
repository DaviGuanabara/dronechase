import time
import glob
import os
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from core.rl_framework.agents.policies.fused_lidar_extractor import FLIAExtractor
from core.rl_framework.utils.io_data import IOData, MultiH5Dataset
from threatsense.level5.level5_dumb_multiobs import Level5DumbMultiObs
from core.rl_framework.agents.policies.ppo_policies import StudentWithFLIA




def train_one_fold(model, train_loader, val_loader, device, epochs=10, lr=1e-3, fold_id=0, output_dir="models_cv"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = []  # para salvar métricas por época

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        num_batches = len(train_loader)

        for i, (obs, target) in enumerate(train_loader, 1):
            obs = {k: v.to(device) for k, v in obs.items()}
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(obs)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.size(0)

            # ETA logging
            if i % 50 == 0 or i == num_batches:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                eta = avg_time * (num_batches - i)
                print(f"[Fold {fold_id}][Epoch {epoch+1}] Batch {i}/{num_batches} "
                      f"Loss: {loss.item():.6f} | ETA: {eta/60:.1f} min")

        avg_train_loss = total_loss / len(train_loader.dataset)

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

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        print(
            f"[Fold {fold_id}][Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

    # Salva histórico em CSV
    df = pd.DataFrame(history)
    csv_path = os.path.join(output_dir, f"history_fold{fold_id+1}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Fold {fold_id}] Histórico salvo em {csv_path}")

    return avg_val_loss


def cross_validate(observation_space, k_folds=5, epochs=10, batch_size=256, lr=1e-3, output_dir="models_cv"):
    os.makedirs(output_dir, exist_ok=True)

    dataset = MultiH5Dataset(
        "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\output\\collect_and_save"
    )
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_size = n // k_folds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for fold in range(k_folds):
        print(f"\n===== Fold {fold+1}/{k_folds} =====")

        # Split train/test indices
        test_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True, persistent_workers=True)

        # Modelo novo para cada fold
        model = StudentWithFLIA(observation_space).to(device)

        val_loss = train_one_fold(model, train_loader, test_loader, device,
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


if __name__ == "__main__":
    dummy_env = Level5DumbMultiObs(GUI=False)
    observation_space = dummy_env.observation_space
    dummy_env.close()

    cross_validate(observation_space, k_folds=5,
                   epochs=10, batch_size=256, lr=1e-3)
    print("Cross-validation concluída.")