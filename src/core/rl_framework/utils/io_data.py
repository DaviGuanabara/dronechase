import glob
from typing import Dict, List, Tuple
import h5py
import numpy as np
import torch
from threatsense.level5.level5_dumb_multiobs import Level5DumbMultiObs
import os
import time
from torch.utils.data import Dataset, DataLoader


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
                n = f["teacher_actions"].shape[0]  # type: ignore
                self.index_map.extend([(file_id, i) for i in range(n)])

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        file_id, local_idx = self.index_map[idx]
        fpath = self.file_paths[file_id]

        with h5py.File(fpath, "r") as f:

            
            obs = {
    
              
                "stacked_spheres": torch.tensor(f["student"]["stacked_spheres"][local_idx], dtype=torch.float32), # type: ignore
                "validity_mask":   torch.tensor(f["student"]["validity_mask"][local_idx], dtype=torch.bool),# type: ignore
                "inertial_data":   torch.tensor(f["student"]["inertial_data"][local_idx], dtype=torch.float32), # type: ignore
                "last_action":     torch.tensor(f["student"]["last_action"][local_idx], dtype=torch.float32),# type: ignore
            }
            target = torch.tensor(f["teacher_actions"]  # type: ignore
                                  [local_idx], dtype=torch.float32)

        return obs, target


class IOData():
    """
    Collect, save and load data to/from HDF5 files.
    """

    def __init__(self, folder_path: str):
        self.set_file_path(folder_path)

    def set_file_path(self, folder_path: str):
        self.folder_path = folder_path
        os.makedirs(folder_path, exist_ok=True)

        self.dataset = MultiH5Dataset(self.folder_path)

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
                final_obs = {
                    "teacher_observation": info["teacher_observation"][i],
                    "student_observation": info["student_observation"][i],
                }
                file_id = observations_collected // 1_000
                file_name = f"{folder_path}/io_data{file_id}.h5"
                teacher_action = info["actions"][i]
                self.save_to_hdf5(file_name, obs=final_obs,
                                  teacher_action=teacher_action)

            if terminated:
                observation, info = dumb_environment.reset()

                global_time = time.time() - start_time
                avg_obs_per_sec = observations_collected / global_time
                eta = (max_observations_collected -
                       observations_collected) / avg_obs_per_sec

                # tempo do episódio
                episode_time = time.time() - episode_start_time
                episode_start_time = time.time()  # reinicia pro próximo

                print(
                    f"[INFO] Episode terminated. Total: {observations_collected}")
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
                value = np.asarray(value, dtype=np.float32)
                if key not in teacher_group:
                    maxshape = (None,) + value.shape
                    teacher_group.create_dataset(
                        key, data=value[None], maxshape=maxshape, chunks=True
                    )
                else:
                    teacher_group[key].resize(  # type: ignore
                        teacher_group[key].shape[0] + 1, axis=0  # type: ignore
                    )
                    teacher_group[key][-1] = value  # type: ignore

            # --- Student obs ---
            student_group = f.require_group("student")
            for key, value in obs["student_observation"].items():
                if key == "validity_mask":
                    value = np.asarray(value).astype(bool)  # mantém booleano
                    # print("[DEBUG] validity_mask:", value, value.dtype)
                else:
                    value = np.asarray(value, dtype=np.float32)

                if key not in student_group:
                    maxshape = (None,) + value.shape
                    student_group.create_dataset(
                        key, data=value[None], maxshape=maxshape, chunks=True
                    )
                else:
                    student_group[key].resize(  # type: ignore
                        student_group[key].shape[0] + 1, axis=0# type: ignore
                    )
                    student_group[key][-1] = value# type: ignore

            # --- Teacher actions (target) ---
            teacher_action = np.asarray(teacher_action, dtype=np.float32)
            if "teacher_actions" not in f:
                f.create_dataset(
                    "teacher_actions",
                    data=teacher_action[None],
                    maxshape=(None,) + teacher_action.shape,
                    chunks=True,
                )
            else:
                f["teacher_actions"].resize(  # type: ignore
                    f["teacher_actions"].shape[0] + 1, axis=0  # type: ignore
                )
                f["teacher_actions"][-1] = teacher_action # type: ignore

    def get_loader(self, batch_size=1024, shuffle=True) -> DataLoader:

        return self._custom_loader(self.dataset, batch_size, shuffle)
    
    def _custom_loader(self, dataset, batch_size=1024, shuffle=True) -> DataLoader:

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=8,  # ajuste para seu número de núcleos
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

    def _compute_fold_size(self, k_folds):
        n = len(self.dataset)
        indices = np.arange(n)
        np.random.shuffle(indices)

        fold_size = n // k_folds
        return fold_size, indices
    
    def _get_train_test_loaders(self, k_folds, fold_size, indices, batch_size=256):
        for fold in range(k_folds):
            test_idx = indices[fold * fold_size: (fold + 1) * fold_size]
            train_subset, test_subset = self._compute_subset(indices, test_idx)
            train_loader = self._custom_loader(train_subset, batch_size=batch_size)
            test_loader = self._custom_loader(test_subset, batch_size=batch_size)
            yield (train_loader, test_loader)



    def _compute_subset(self, indices, test_idx)-> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
        train_idx = np.setdiff1d(indices, test_idx)
        train_subset = torch.utils.data.Subset(self.dataset, train_idx) # type: ignore
        test_subset = torch.utils.data.Subset(self.dataset, test_idx)
        return (train_subset, test_subset)

    def cross_validation_loaders(self, k_folds=5, batch_size=256):

        fold_size, indices = self._compute_fold_size(k_folds)
        return self._get_train_test_loaders(k_folds, fold_size, indices, batch_size=batch_size)

    


if __name__ == "__main__":
    #io_data = IOData("output/collect_and_save/")
    io_data = IOData("C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\output\\collect_and_save")
    loader = io_data.get_loader()

    


    for obs, target in loader:
        print("Batch shape stacked_spheres:", obs["stacked_spheres"].shape)
        print("Batch shape validity_mask:", obs["validity_mask"].shape)
        print("validity_mask sample:", obs["validity_mask"])

        break

