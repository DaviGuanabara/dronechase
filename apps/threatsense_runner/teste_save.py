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

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)

# === Caminho do modelo professor (Exp03) ===
BASE_MODEL_PATH = Path(
    # "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\experiment_results\\threat_engage\\Etapa 03 - Level 4\\EXP03\\Treinamento - 1RL 1BT - EXP03\\31_01_2024_level4_2.00M_exp03_vFinal\\Trial_8\\models_dir\\t8_PPO_r6995.40.zip"
    "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\experiment_results\\threat_engage\\Etapa 03 - Level 4\\EXP03\\Treinamento - 1RL 1BT - EXP03\\31_01_2024_level4_2.00M_exp03_vFinal\\Trial_8\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t8_PPO_r6995.40.zip"
)


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


def load_teacher_model(device: torch.device) -> torch.nn.Module:
    return PPO.load(str(BASE_MODEL_PATH), device=device).policy


def alias(old: str, new: str):
    try:
        sys.modules[old] = importlib.import_module(new)
        logging.info(f"[compat] {old} -> {new}")
    except Exception as e:
        logging.warning(f"[compat] Falhou alias {old}->{new}: {e}")


def save_to_hdf5(file_path, obs, teacher_action):
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
            # mantém validity_mask como bool
            if key == "validity_mask":
                print("[DEBUG] Salvando validity_mask como bool")
                print(type(value))
                print(type(value[0]))
                print(value)
                value = np.array(value, dtype=np.bool_)
                print("[DEBUG] Após conversão:")
                print(type(value))
                print(type(value[0]))
                print(value)
                print("Salvou validity_mask como bool")
            else:
                value = np.array(value, dtype=np.float32)

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


env = Level5DumbMultiObs(GUI=False, rl_frequency=15)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alias('loyalwingmen', 'core')  # mantenha se o zip referenciar esse prefixo
# teacher_policy: torch.nn.Module = load_teacher_model(device)

if __name__ == "__main__":
    action_space = env.action_space
    observations, info = env.reset()
    data = {}
    observations_collected = 0
    max_observations_collected = 1

    start_time = time.time()
    episode_start_time = start_time
    while (observations_collected < max_observations_collected):
        # action = action_space.sample()

        observations, reward, terminated, truncated, info = env.step()
        # print(observations[0]["validity_mask"])

        # print(observations.shape, info["teacher_observation"].shape, info["student_observation"].shape)
        observations_collected += len(observations)

        for i in range(len(observations)):
            final_obs = {}
            final_obs["teacher_observation"] = info["teacher_observation"][i]
            final_obs["student_observation"] = info["student_observation"][i]
            file_id = observations_collected // 1_000
            file_name = f"output/collect_and_save/teste/05.09.2025_final_collect_and_save_output_part{file_id}.h5"
            teacher_action = info["actions"][i]
            save_to_hdf5(file_name, obs=final_obs, teacher_action=teacher_action)

        elapsed_time = time.time()
        if terminated:
            observation, info = env.reset()

            global_time = time.time() - start_time
            avg_obs_per_sec = observations_collected / global_time
            eta = (max_observations_collected -
                observations_collected) / avg_obs_per_sec

            # tempo do episódio
            episode_time = time.time() - episode_start_time
            episode_start_time = time.time()  # reinicia pro próximo

            print(f"[INFO] Episode terminated. Total: {observations_collected}")
            print(f"[INFO] Episode duration: {episode_time:.2f} sec")
            print(f"[INFO] Avg speed: {avg_obs_per_sec:.2f} obs/sec")
            print(f"[INFO] ETA: {eta/3600:.2f} hours")

    dataset = MultiH5Dataset("output/collect_and_save/teste")

    dataloader = DataLoader(
        dataset,
        batch_size=1,      # ou 1000, ou 2048
        shuffle=True,
        num_workers=1,        # paraleliza leitura de disco
        pin_memory=True,      # acelera envio para GPU
        prefetch_factor=1     # cada worker mantém 2 batches prontos
    )


    print(f"Dataset size: {len(dataloader)}")
    print(dataloader)
    for batch in dataloader:
        obs, target = batch
        print(type(obs), type(target))  # obs será Dict[str, Tensor]
        print(obs)
        break

    dataset_2 = MultiH5Dataset("output/collect_and_save/")

    dataloader_2 = DataLoader(
        dataset_2,
        batch_size=1,      # ou 1000, ou 2048
        shuffle=True,
        num_workers=1,        # paraleliza leitura de disco
        pin_memory=True,      # acelera envio para GPU
        prefetch_factor=1     # cada worker mantém 2 batches prontos
    )

    print(f"Dataset size: {len(dataloader_2)}")
    print(dataloader_2)
    for batch in dataloader_2:
        obs, target = batch
        print(type(obs), type(target))  # obs será Dict[str, Tensor]
        print(obs)
        break
