import glob
import os
from typing import Dict, Tuple
import h5py
import torch
from torch.utils.data import Dataset

from core.rl_framework.agents.policies.fused_lidar_extractor import FLIAExtractor
from torch import nn, optim
from torch.utils.data import DataLoader


from torch.utils.data import Dataset, DataLoader, default_collate
from typing import List, Dict, Tuple


class MultiH5Dataset(Dataset):
    def __init__(self, folder: str, pattern: str = "*.h5"):
        # encontra todos os arquivos HDF5 dentro da pasta
        self.file_paths: List[str] = sorted(
            glob.glob(os.path.join(folder, pattern)))
        if not self.file_paths:
            raise FileNotFoundError(
                f"Nenhum arquivo encontrado em {folder} com padrão {pattern}")

        self.index_map = []  # lista de (file_id, local_idx)

        # constrói o mapa de índices
        for file_id, path in enumerate(self.file_paths):
            with h5py.File(path, "r") as f:
                n = f["teacher_actions"].shape[0]
                self.index_map.extend([(file_id, i) for i in range(n)])

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        file_id, local_idx = self.index_map[idx]
        fpath = self.file_paths[file_id]
        
        with h5py.File(fpath, "r") as f:
            print(type(f["student"]["validity_mask"][local_idx]))
            print(type(f["student"]["validity_mask"][local_idx][0]))
            print(f["student"]["validity_mask"][local_idx])
            obs = {
                "stacked_spheres": torch.tensor(f["student"]["stacked_spheres"][local_idx], dtype=torch.float32),
                "validity_mask":   torch.tensor(f["student"]["validity_mask"][local_idx], dtype=torch.bool),
                "inertial_data":   torch.tensor(f["student"]["inertial_data"][local_idx], dtype=torch.float32),
                "last_action":     torch.tensor(f["student"]["last_action"][local_idx], dtype=torch.float32),
            }
            target = torch.tensor(f["teacher_actions"]
                                  [local_idx], dtype=torch.float32)

        return obs, target
# ==== Student Network ====


class StudentWithFLIA(nn.Module):
    def __init__(self, observation_space, action_dim):
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
            nn.Linear(512, action_dim)  # output final = ação
        )

    def forward(self, obs):
        features = self.feature_extractor(obs)   # [B, 512]
        return self.mlp_head(features)           # [B, action_dim]


# ==== Training Loop (supervised imitation) ====


def train_student(dataset, observation_space, action_dim, epochs=10, batch_size=256, lr=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentWithFLIA(observation_space, action_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # ou CrossEntropy se for ação discreta

    for epoch in range(epochs):
        total_loss = 0
        for obs, target in dataloader:
            # obs deve estar no formato Dict[str, torch.Tensor] compatível com FLIAExtractor
            obs = {k: v.to(device) for k, v in obs.items()}
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(obs)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.size(0)

        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(dataset):.6f}")

    return model


# ==== Exemplo de uso com DataLoader (corrige aviso do Pylance) ====
if __name__ == "__main__":
    dataset = MultiH5Dataset("output/collect_and_save")

    dataloader = DataLoader(
        dataset,
        batch_size=1024,      # ou 1000, ou 2048
        shuffle=True,
        num_workers=4,        # paraleliza leitura de disco
        pin_memory=True,      # acelera envio para GPU
        prefetch_factor=2     # cada worker mantém 2 batches prontos
    )

    print(f"Dataset size: {len(dataloader)}")
    print(dataloader)
    for batch in dataloader:
        obs, target = batch
        print(type(obs), type(target))  # obs será Dict[str, Tensor]
        print(obs)
        break
