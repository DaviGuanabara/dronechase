# Stage 01 – Reference Configuration

This configuration reproduces the experimental setup described in **Stage 1** of the dissertation.

- Objective: Optimize the hyperparameters of the PPO agent in a simplified scenario with a static loitering munition (LM).
- Scenario: One pursuer and one static invader within a dome-shaped arena.
- Method: Bayesian Optimization (BO) with 10 trials.
- Each trial: 2 million timesteps.
- Hyperparameter search space:
  - Batch size: [128, 256, 512, 1024]
  - Hidden layers: [128, 256, 512] in 3-layer MLP
  - Learning rate: Fixed at 1e-3

📄 Refer to **Section 4.2.1** of the dissertation for a detailed description.
