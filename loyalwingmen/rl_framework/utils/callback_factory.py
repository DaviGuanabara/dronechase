import os
from typing import Optional, Tuple
import numpy as np
import math
from dataclasses import dataclass

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    ProgressBarCallback,
    BaseCallback,
)
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import VecEnv

from typing import List, Tuple

from enum import Enum
from typing import List
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecEnv
import threading


class CallbackType(Enum):
    EVAL = "eval"
    CHECKPOINT = "checkpoint"
    PROGRESSBAR = "progressbar"


# Create a mutex to ensure mutual exclusion during directory creation
mutex = threading.Lock()


def create_directories_if_not_exist(path: str) -> bool:
    with mutex:
        try:
            # Try to create the directories if they don't exist
            os.makedirs(path, exist_ok=True)
            return True
        except OSError as e:
            # Handle exceptions if an error occurs while creating directories
            print(f"Error creating directories at {path}: {e}")
            return False


def check_directories(path: str, path_name: str) -> bool:
    # Check if the paths are writable directories
    try:
        assert os.access(path, os.W_OK), f"{path_name} must be a writable directory"
        return True
    except AssertionError as e:
        print(f"Error checking write access: {e}")
        return False


def callbacklist(
    env: VecEnv,
    log_path: str = "./logs/",
    model_path: str = "./models/",
    save_freq: int = 10_000,
    callbacks_to_include: Optional[List[CallbackType]] = None,
    n_eval_episodes: int = 10,
    debug: bool = False,
) -> CallbackList:
    if callbacks_to_include is None:
        callbacks_to_include = [
            CallbackType.EVAL,
            CallbackType.CHECKPOINT,
            CallbackType.PROGRESSBAR,
        ]
    list_callbacks = []

    if CallbackType.EVAL in callbacks_to_include:
        create_directories_if_not_exist(log_path)
        create_directories_if_not_exist(model_path)

        check_directories(log_path, "log_path")
        check_directories(model_path, "model_path")

        # Check if eval_freq is valid
        assert (
            isinstance(save_freq, int) and save_freq > 0
        ), "save_freq must be a positive integer"

        stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=3, min_evals=5, verbose=1  # 0
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path=model_path,
            n_eval_episodes=n_eval_episodes,
            log_path=log_path,
            eval_freq=save_freq,
            deterministic=True,
            render=False,
            callback_after_eval=stop_train_callback,
            verbose=1,  # int(debug)
        )

        list_callbacks.append(eval_callback)

    if CallbackType.CHECKPOINT in callbacks_to_include:
        # Check if save_path is writable
        assert os.access(model_path, os.W_OK), "model_path must be a writable directory"

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq, save_path=model_path
        )

        list_callbacks.append(checkpoint_callback)

    if CallbackType.PROGRESSBAR in callbacks_to_include:
        progressbar_callback = ProgressBarCallback()
        list_callbacks.append(progressbar_callback)

    return CallbackList(list_callbacks)
