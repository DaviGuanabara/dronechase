import importlib
import logging
from pathlib import Path

import h5py
import numpy as np

import torch
from threatsense.level5.level5_dumb_multiobs import Level5DumbMultiObs
from core.rl_framework.utils.pipeline import ReinforcementLearningPipeline

import sys
import os
import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor


current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)


def alias(old: str, new: str):
    try:
        sys.modules[old] = importlib.import_module(new)
        logging.info(f"[compat] {old} -> {new}")
    except Exception as e:
        logging.warning(f"[compat] Falhou alias {old}->{new}: {e}")


def save_to_hdf5(file_path, obs_list, teacher_actions):
    """
    Salva lotes de observações e ações em grupos no HDF5.
    obs_list: lista de dicionários com observações
    teacher_actions: lista de arrays (ações do professor)
    """

    with h5py.File(file_path, "a") as f:
        student_group = f.require_group("student")

        for i in range(len(obs_list)):
            obs = obs_list[i]
            action = teacher_actions[i]

            # --- Student obs ---
            for key, value in obs.items():
                if key == "validity_mask":
                    value = np.array(value, dtype=np.bool_)
                else:
                    value = np.array(value, dtype=np.float32)

                if key not in student_group:
                    maxshape = (None,) + value.shape
                    student_group.create_dataset(
                        key, data=value[None], maxshape=maxshape, chunks=True
                    )
                else:
                    student_group[key].resize(  # type: ignore
                        student_group[key].shape[0] + 1, axis=0  # type: ignore
                    )
                    student_group[key][-1] = value  # type: ignore

            # --- Teacher actions (target) ---
            action = np.array(action, dtype=np.float32)
            if "teacher_actions" not in f:
                f.create_dataset(
                    "teacher_actions",
                    data=action[None],
                    maxshape=(None,) + action.shape,
                    chunks=True,
                )
            else:
                f["teacher_actions"].resize(  # type: ignore
                    f["teacher_actions"].shape[0] + 1, axis=0  # type: ignore
                )
                f["teacher_actions"][-1] = action  # type: ignore


def count_observations_in_hdf5_files(dir_path: Path) -> int:
    total = 0
    for file_path in sorted(dir_path.glob("part*.h5")):
        try:
            with h5py.File(file_path, "r") as f:
                teacher_actions = f.get("teacher_actions")
                if isinstance(teacher_actions, h5py.Dataset):
                    total += teacher_actions.shape[0]
        except OSError as error:
            print(f"[WARNING] Could not read {file_path.name}: {error}")
    return total


def drop_invalid_student_obs(info):
    validitys = [obs["validity_mask"] for obs in info["student_observations"]]

    # garante 2D se todos têm mesmo shape
    try:
        validitys = np.stack(validitys)
        valid_indices = np.where(validitys.any(axis=1))[0]
    except ValueError:
        # fallback: tamanho variável → usa list comprehension
        valid_flags = [mask.any() for mask in validitys]
        valid_indices = np.where(valid_flags)[0]

    info["student_observations"] = [info["student_observations"][i]
                                    for i in valid_indices]
    info["teacher_actions"] = [info["teacher_actions"][i]
                               for i in valid_indices]

    if len(valid_indices) == 0:
        print("[WARNING] Nenhuma observação válida neste passo.")
        info["student_observations"] = [{
            "stacked_spheres": np.zeros((2, 13, 26), dtype=np.float32),
            "validity_mask": np.zeros((2,), dtype=bool),
            "inertial_data": np.zeros((10,), dtype=np.float32),
            "last_action": np.zeros((4,), dtype=np.float32),
        }]
        info["teacher_actions"] = [np.zeros((4,), dtype=np.float32)]

    return info


if __name__ == "__main__":

    """
    This environment is designed to collect data for training a student model to mimic a teacher's behavior. 
    The teacher is the pursuer's policy (Behavior Tree).
    It collects the observation and the action of each pursuer at each step. 
    in task, we can check the number of pursuers. Each step will generate n pursuers observation-action pairs.

    """

    pipeline = ReinforcementLearningPipeline()
    vec_env = pipeline.create_vectorized_environment(
        environment=Level5DumbMultiObs, n_envs=4, GUI=False)

    # env = Level5DumbMultiObs(GUI=False, rl_frequency=15)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")

    alias('loyalwingmen', 'core')  # mantenha se o zip referenciar esse prefixo

    action_space = vec_env.action_space
    # observation, info = env.reset()
    observation = vec_env.reset()

    target_total_observations = 5_000_000

    batch_obs = []
    batch_actions = []

    episode_start_time = time.time()

    dir_path = Path("output/collect_and_save/29.04.2026")
    dir_path.mkdir(parents=True, exist_ok=True)

    executor = ThreadPoolExecutor(max_workers=4)

    # Verify existing files to determine the next file_id.
    # This avoids writing new batches over files already collected in previous runs.
    existing_files = sorted(dir_path.glob("part*.h5"))

    if existing_files:
        last_id = max(int(f.stem.replace("part", "")) for f in existing_files)
        file_id = last_id + 1
    else:
        file_id = 0

    previously_collected = count_observations_in_hdf5_files(dir_path)
    observations_collected = 0
    run_start_time = time.time()

    print(f"[INFO] Existing observations in directory: {previously_collected}")
    print(f"[INFO] Starting from file_id: {file_id}")

    print("Collecting data... Press Ctrl+C to stop.")
    while (previously_collected + observations_collected < target_total_observations):

        observations, reward, terminated, infos = vec_env.step(
            np.zeros(4))

        student_observations = []
        labels = []

        for info in infos:
            info = drop_invalid_student_obs(info)
            student_observations.extend(info["student_observations"])
            labels.extend(info["teacher_actions"])

        observations_collected += len(student_observations)  # type: ignore

        batch_obs.extend(student_observations)  # type: ignore
        batch_actions.extend(labels)  # type: ignore

        if len(batch_obs) >= 1000:

            file_path = dir_path / f"part{file_id}.h5"
            file_name = str(file_path)

            # save_to_hdf5(file_name, obs_list=batch_obs, teacher_actions=batch_actions)
            # cria cópia local dos dados antes de limpar
            obs_copy = list(batch_obs)
            actions_copy = list(batch_actions)

            batch_obs.clear()
            batch_actions.clear()
            file_id += 1

            executor.submit(save_to_hdf5, file_name, obs_copy, actions_copy)

         # métricas globais (sempre atualiza)
        run_elapsed_time = time.time() - run_start_time
        total_observations_collected = previously_collected + observations_collected
        avg_obs_per_sec = observations_collected / \
            run_elapsed_time if run_elapsed_time > 0 else 0.0

        remaining_observations = max(
            target_total_observations - total_observations_collected,
            0,
        )
        eta = remaining_observations / avg_obs_per_sec if avg_obs_per_sec > 0 else 0.0

        print(
            f"[INFO] Run collected {observations_collected} | "
            f"Total collected {total_observations_collected} / {target_total_observations} "
            f"observations, Run elapsed: {run_elapsed_time/3600:.2f} hours, "
            f"Avg speed: {avg_obs_per_sec:.2f} obs/sec, "
            f"ETA: {eta/3600:.2f} hours, ",
            # f"File ID: {file_id}",
            end="\r"
        )

        # print("")
        # print(terminated)

        if np.any(terminated):
            episode_time = time.time() - episode_start_time
            episode_start_time = time.time()
            observation = vec_env.reset()

            print(
                f"\n[INFO] Episode terminated. Run total: {observations_collected}")
            print(
                f"[INFO] Episode terminated. Dataset total: {total_observations_collected}")
            print(f"[INFO] Episode duration: {episode_time:.2f} sec")
            print(f"[INFO] Avg speed: {avg_obs_per_sec:.2f} obs/sec")
            print(f"[INFO] ETA: {eta/3600:.2f} hours")

    executor.shutdown(wait=True)  # espera todas as tarefas terminarem
    print("[INFO] Todos os arquivos foram salvos com segurança.")
