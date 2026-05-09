from pathlib import Path
from typing import Any, cast
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

try:
    from scipy import stats
except ImportError:
    stats = None

from core.rl_framework.utils.pipeline import ReinforcementLearningPipeline
from threatsense.level5.level5_c1_fusion_environment import (
    Level5C1FusionEnvironment as Level5C2Fusion,
)


current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)


N_EVAL_EPISODES = 100

INPUT_MODEL = Path(
    "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/rl_2/c2_27.04.2026/train_rl_1/models/final.zip"
)
OUTPUT_DIR = Path(
    "/Users/Davi/Documents/GitHub/dronechase/apps/threatsense_runner/new_phase/output/evaluation/baseline_no_action"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "baseline_no_action_episode_metrics.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "baseline_no_action_summary_metrics.csv"
OUTPUT_DESCRIPTIVE_STATS_CSV = OUTPUT_DIR / \
    "baseline_no_action_descriptive_statistics.csv"
OUTPUT_HYPOTHESIS_TESTS_CSV = OUTPUT_DIR / \
    "baseline_no_action_hypothesis_tests.csv"


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---- Descriptive statistics and hypothesis testing ----

def compute_numeric_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "episode_reward",
        "episode_duration_steps",
        "agent_kills",
        "allies_kills",
        "total_kills",
        "kill_ratio",
        "agent_kill_ratio",
        "allies_kill_ratio",
        "kills_per_defender",
        "kills_per_agent",
        "kills_per_ally",
        "pursuers_exploded",
        "pursuer_suicides",
        "agent_suicides",
    ]
    available_columns = [
        column for column in numeric_columns if column in df.columns]

    rows = []
    for column in available_columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        n = len(values)
        mean = values.mean() if n > 0 else np.nan
        std = values.std(ddof=1) if n > 1 else np.nan
        standard_error = std / np.sqrt(n) if n > 1 else np.nan
        ci95_margin = 1.96 * standard_error if n > 1 else np.nan

        rows.append(
            {
                "metric": column,
                "n": n,
                "mean": mean,
                "std": std,
                "min": values.min() if n > 0 else np.nan,
                "median": values.median() if n > 0 else np.nan,
                "max": values.max() if n > 0 else np.nan,
                "standard_error": standard_error,
                "ci95_low": mean - ci95_margin if n > 1 else np.nan,
                "ci95_high": mean + ci95_margin if n > 1 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_success_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()


def save_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    descriptive_df = pd.concat(
        [
            compute_numeric_descriptive_statistics(df),
            compute_success_statistics(df),
        ],
        ignore_index=True,
    )
    descriptive_df.to_csv(OUTPUT_DESCRIPTIVE_STATS_CSV, index=False)
    return descriptive_df


def mann_whitney_test(proposed_df: pd.DataFrame, baseline_df: pd.DataFrame, metric: str):
    if stats is None or metric not in proposed_df.columns or metric not in baseline_df.columns:
        return None

    proposed_values = pd.to_numeric(
        proposed_df[metric], errors="coerce").dropna()
    baseline_values = pd.to_numeric(
        baseline_df[metric], errors="coerce").dropna()

    if len(proposed_values) == 0 or len(baseline_values) == 0:
        return None

    result = stats.mannwhitneyu(
        proposed_values,
        baseline_values,
        alternative="two-sided",
    )

    return {
        "metric": metric,
        "test": "Mann-Whitney U",
        "baseline_file": None,
        "proposed_n": len(proposed_values),
        "baseline_n": len(baseline_values),
        "proposed_mean": proposed_values.mean(),
        "baseline_mean": baseline_values.mean(),
        "statistic": float(result[0]),
        "p_value": float(result[1]),
    }


def fisher_success_test(proposed_df: pd.DataFrame, baseline_df: pd.DataFrame):
    return None


def save_hypothesis_tests(proposed_df: pd.DataFrame) -> pd.DataFrame:
    baseline_files = sorted(OUTPUT_DIR.glob("baseline_*_episode_metrics.csv"))
    if not baseline_files:
        empty_df = pd.DataFrame(
            columns=[
                "metric",
                "test",
                "baseline_file",
                "proposed_n",
                "baseline_n",
                "proposed_mean",
                "baseline_mean",
                "statistic",
                "p_value",
            ]
        )
        empty_df.to_csv(OUTPUT_HYPOTHESIS_TESTS_CSV, index=False)
        return empty_df

    test_rows = []
    metrics_for_mann_whitney = [
        "episode_reward",
        "episode_duration_steps",
        "total_kills",
        "agent_kills",
        "allies_kills",
        "kill_ratio",
        "agent_kill_ratio",
        "allies_kill_ratio",
        "kills_per_defender",
        "kills_per_agent",
        "kills_per_ally",
    ]

    for baseline_file in baseline_files:
        baseline_df = pd.read_csv(baseline_file)

        for metric in metrics_for_mann_whitney:
            result = mann_whitney_test(proposed_df, baseline_df, metric)
            if result is not None:
                result["baseline_file"] = baseline_file.name
                test_rows.append(result)

    hypothesis_df = pd.DataFrame(test_rows)
    hypothesis_df.to_csv(OUTPUT_HYPOTHESIS_TESTS_CSV, index=False)
    return hypothesis_df


def main() -> None:
    vectorized_environment = ReinforcementLearningPipeline.create_vectorized_environment(
        environment=Level5C2Fusion,
        env_kwargs={},
        n_envs=1,
    )

    device = get_torch_device()

    observation = vectorized_environment.reset()
    episode_rows = []
    episode_count = 0
    current_episode_reward = 0.0
    current_episode_length = 0

    # ETA tracking
    start_time = time.time()

    partial_output_csv = OUTPUT_DIR / "baseline_no_action_episode_metrics_partial.csv"
    if partial_output_csv.exists():
        partial_output_csv.unlink()

    while episode_count < N_EVAL_EPISODES:

        # No-action baseline: the RL-controlled agent remains inert.
        # The BT-controlled ally still acts normally.
        action = np.zeros((1, 4), dtype=np.float32)
        observation, reward, done, info = vectorized_environment.step(action)
        current_episode_reward += float(reward[0])
        current_episode_length += 1

        if done[0]:
            episode_info = dict(info[0])
            episode_info["episode"] = episode_count + 1
            episode_info["episode_reward"] = current_episode_reward
            episode_info["episode_duration_steps"] = current_episode_length

            agent_kills = float(episode_info.get("agent_kills", 0) or 0)
            allies_kills = float(episode_info.get("allies_kills", 0) or 0)
            total_kills = float(episode_info.get(
                "total_kills", agent_kills + allies_kills) or 0)

            num_allies = float(episode_info.get("num_allies", 1) or 1)
            num_defenders = float(episode_info.get(
                "num_defenders", 1 + num_allies) or (1 + num_allies))

            episode_info["agent_kills"] = agent_kills
            episode_info["allies_kills"] = allies_kills
            episode_info["total_kills"] = total_kills
            episode_info["num_allies"] = num_allies
            episode_info["num_defenders"] = num_defenders
            episode_info["kills_per_agent"] = agent_kills
            episode_info["kills_per_ally"] = allies_kills / \
                num_allies if num_allies > 0 else 0.0
            episode_info["kills_per_defender"] = total_kills / \
                num_defenders if num_defenders > 0 else 0.0

            episode_rows.append(episode_info)
            pd.DataFrame(episode_rows).to_csv(partial_output_csv, index=False)

            elapsed = time.time() - start_time
            episodes_done = episode_count
            eta_seconds = (elapsed / episodes_done) * (N_EVAL_EPISODES -
                                                       episodes_done) if episodes_done > 0 else 0

            print(
                f"Episode {episode_count}/{N_EVAL_EPISODES} | "
                f"ETA: {eta_seconds/60:.1f} min | "
                f"kills={episode_info.get('total_kills')} | "
                f"agent_kills={episode_info.get('agent_kills')} | "
                f"allies_kills={episode_info.get('allies_kills')} | "
                f"kills/defender={episode_info.get('kills_per_defender')} | "
                f"kills/ally={episode_info.get('kills_per_ally')} | "
                f"duration={episode_info.get('episode_duration_steps')}"
            )
            episode_count += 1
            current_episode_reward = 0.0
            current_episode_length = 0

    df = pd.DataFrame(episode_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    df.to_csv(partial_output_csv, index=False)

    summary = {
        "n_episodes": len(df),
        "mean_episode_duration_steps": df["episode_duration_steps"].mean()
        if "episode_duration_steps" in df
        else None,
        "std_episode_duration_steps": df["episode_duration_steps"].std()
        if "episode_duration_steps" in df
        else None,
        "mean_total_kills": df["total_kills"].mean() if "total_kills" in df else None,
        "std_total_kills": df["total_kills"].std() if "total_kills" in df else None,
        "mean_agent_kills": df["agent_kills"].mean() if "agent_kills" in df else None,
        "std_agent_kills": df["agent_kills"].std() if "agent_kills" in df else None,
        "mean_allies_kills": df["allies_kills"].mean() if "allies_kills" in df else None,
        "std_allies_kills": df["allies_kills"].std() if "allies_kills" in df else None,
        "mean_kill_ratio": df["kill_ratio"].mean() if "kill_ratio" in df else None,
        "std_kill_ratio": df["kill_ratio"].std() if "kill_ratio" in df else None,
        "mean_agent_kill_ratio": df["agent_kill_ratio"].mean()
        if "agent_kill_ratio" in df
        else None,
        "std_agent_kill_ratio": df["agent_kill_ratio"].std()
        if "agent_kill_ratio" in df
        else None,
        "mean_allies_kill_ratio": df["allies_kill_ratio"].mean()
        if "allies_kill_ratio" in df
        else None,
        "std_allies_kill_ratio": df["allies_kill_ratio"].std()
        if "allies_kill_ratio" in df
        else None,
        "mean_kills_per_defender": df["kills_per_defender"].mean()
        if "kills_per_defender" in df
        else None,
        "std_kills_per_defender": df["kills_per_defender"].std()
        if "kills_per_defender" in df
        else None,
        "mean_kills_per_agent": df["kills_per_agent"].mean()
        if "kills_per_agent" in df
        else None,
        "std_kills_per_agent": df["kills_per_agent"].std()
        if "kills_per_agent" in df
        else None,
        "mean_kills_per_ally": df["kills_per_ally"].mean()
        if "kills_per_ally" in df
        else None,
        "std_kills_per_ally": df["kills_per_ally"].std()
        if "kills_per_ally" in df
        else None,
        "mean_agent_suicides": df["agent_suicides"].mean()
        if "agent_suicides" in df
        else None,
        "std_agent_suicides": df["agent_suicides"].std()
        if "agent_suicides" in df
        else None,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    descriptive_df = save_descriptive_statistics(df)
    hypothesis_df = save_hypothesis_tests(df)

    print(f"Episode metrics saved to: {OUTPUT_CSV}")
    print(f"Partial episode metrics saved to: {partial_output_csv}")
    print(f"Summary metrics saved to: {OUTPUT_SUMMARY_CSV}")
    print(f"Descriptive statistics saved to: {OUTPUT_DESCRIPTIVE_STATS_CSV}")
    print(f"Hypothesis tests saved to: {OUTPUT_HYPOTHESIS_TESTS_CSV}")
    print(f"Descriptive statistics rows: {len(descriptive_df)}")
    print(f"Hypothesis test rows: {len(hypothesis_df)}")

    vectorized_environment.close()


if __name__ == "__main__":
    main()
