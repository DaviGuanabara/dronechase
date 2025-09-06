import glob
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import h5py
import numpy as np
from stable_baselines3 import PPO
import torch
from threatsense.level5.level5_dumb_multiobs import Level5DumbMultiObs
from core.utils.keyboard_listener import KeyboardListener
import sys
import os
import time
from torch.utils.data import Dataset, DataLoader, default_collate


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

            obs = {
                "stacked_spheres": torch.tensor(f["student"]["stacked_spheres"][local_idx], dtype=torch.float32),
                "validity_mask":   torch.tensor(f["student"]["validity_mask"][local_idx], dtype=torch.bool),
                "inertial_data":   torch.tensor(f["student"]["inertial_data"][local_idx], dtype=torch.float32),
                "last_action":     torch.tensor(f["student"]["last_action"][local_idx], dtype=torch.float32),
            }
            target = torch.tensor(f["teacher_actions"][local_idx], dtype=torch.float32)

        return obs, target



class IOData():
    """
    Collect, save and load data to/from HDF5 files.
    """

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        os.makedirs(folder_path, exist_ok=True)

    def collect_data(self, dumb_environment, max_observations_collected=5_000_000):
        folder_path = self.folder_path
        
        observations, info = dumb_environment.reset()
        observations_collected = 0


        start_time = time.time()
        episode_start_time = start_time
        while (observations_collected < max_observations_collected):


            observations, _, terminated, _, info = dumb_environment.step()
            observations_collected += len(observations)

            for i in range(len(observations)):
                final_obs = {}
                final_obs["teacher_observation"] = info["teacher_observation"][i]
                final_obs["student_observation"] = info["student_observation"][i]
                file_id = observations_collected // 1_000
                file_name = f"{folder_path}/io_data{file_id}.h5"
                teacher_action = info["actions"][i]
                self.save_to_hdf5(file_name, obs=final_obs, teacher_action=teacher_action)

            elapsed_time = time.time()
            if terminated:
                observation, info = dumb_environment.reset()

                global_time = time.time() - start_time
                avg_obs_per_sec = observations_collected / global_time
                eta = (max_observations_collected - observations_collected) / avg_obs_per_sec

                # tempo do episódio
                episode_time = time.time() - episode_start_time
                episode_start_time = time.time()  # reinicia pro próximo

                print(f"[INFO] Episode terminated. Total: {observations_collected}")
                print(f"[INFO] Episode duration: {episode_time:.2f} sec")
                print(f"[INFO] Avg speed: {avg_obs_per_sec:.2f} obs/sec")
                print(f"[INFO] ETA: {eta/3600:.2f} hours")

    def save_to_hdf5(self, file_path, obs, teacher_action):
        """
        Salva teacher/student em grupos no HDF5.
        obs: {"teacher_observation": dict, "student_observation": dict}
        teacher_action: ação tomada pelo professor/BT (shape ex: [4])
        """

        with h5py.File(file_path, "a") as f:

            # --- Teacher obs ---
            teacher_group = f.require_group("teacher")
            for key, value in obs["teacher_observation"].items():
                value = np.array(value, dtype=np.float32)
                if key not in teacher_group:
                    maxshape = (None,) + value.shape
                    teacher_group.create_dataset(
                        key, data=value[None], maxshape=maxshape, chunks=True
                    )
                else:
                    teacher_group[key].resize(
                        teacher_group[key].shape[0] + 1, axis=0
                    )
                    teacher_group[key][-1] = value

            # --- Student obs ---
            student_group = f.require_group("student")
            for key, value in obs["student_observation"].items():
                if key == "validity_mask":
                    value = np.asarray(value).astype(bool)  # mantém booleano
                else:
                    value = np.asarray(value, dtype=np.float32)

                if key not in student_group:
                    maxshape = (None,) + value.shape
                    student_group.create_dataset(
                        key, data=value[None], maxshape=maxshape, chunks=True
                    )
                else:
                    student_group[key].resize(
                        student_group[key].shape[0] + 1, axis=0
                    )
                    student_group[key][-1] = value


            # --- Teacher actions (target) ---
            teacher_action = np.array(teacher_action, dtype=np.float32)
            if "teacher_actions" not in f:
                f.create_dataset(
                    "teacher_actions",
                    data=teacher_action[None],
                    maxshape=(None,) + teacher_action.shape,
                    chunks=True,
                )
            else:
                f["teacher_actions"].resize(
                    f["teacher_actions"].shape[0] + 1, axis=0
                )
                f["teacher_actions"][-1] = teacher_action

        h5py.File(file_path, "a").close()

    def get_loader(self):
        dataset = MultiH5Dataset(self.folder_path)

        dataloader = DataLoader(
            dataset,
            batch_size=1,      # ou 1000, ou 2048
            shuffle=True,
            num_workers=1,        # paraleliza leitura de disco
            pin_memory=True,      # acelera envio para GPU
            prefetch_factor=1     # cada worker mantém 2 batches prontos
        )
        return dataloader


if __name__ == "__main__":
    io_data = IOData("output/collect_and_save/teste/")
    io_data.collect_data(Level5DumbMultiObs(GUI=False), max_observations_collected=10)
    loader = io_data.get_loader()
    print(f"Dataset size: {len(loader)}")
    print(loader)
    for batch in loader:
        obs, target = batch
        print(type(obs), type(target))  # obs será Dict[str, Tensor]
        print(obs)
        break
