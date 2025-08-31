import numpy as np
from core.rl_framework.utils.pipeline import ReinforcementLearningPipeline
from threatsense.level5.level5_environment import Level5Environment


def debug_vec_env(n_steps=10):
    env = ReinforcementLearningPipeline.create_vectorized_environment(
        environment=Level5Environment,
        env_kwargs=dict(dome_radius=20, rl_frequency=15),
        n_envs=2,
        GUI=False
    )

    obs = env.reset()
    print("\n[RESET]")
    for key, arr in obs.items():
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
        if key == "validity_mask":
            print("mask values:", arr)
            # sanity check no reset
            if arr.ndim != 2 or arr.shape[1] != 6:
                raise ValueError(
                    f"[RESET] Mask shape inconsistente: {arr.shape}")

    for step in range(n_steps):
        actions = [env.action_space.sample() for _ in range(env.num_envs)]
        obs, rewards, dones, infos = env.step(actions)

        print(f"\n[STEP {step}] rewards={rewards}, dones={dones}")
        for key, arr in obs.items():
            print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
            if key == "validity_mask":
                print("mask values:", arr)
                # sanity check a cada step
                if arr.ndim != 2 or arr.shape[1] != 6:
                    raise ValueError(
                        f"[STEP {step}] Mask shape inconsistente: {arr.shape}")

    env.close()


if __name__ == "__main__":
    debug_vec_env(20)
