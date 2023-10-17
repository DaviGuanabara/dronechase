## PID TUNNING PROJECT

## Comments

Only the agent's quadcopter is attempting to replicate the target velocity vector. The observation is the target velocity vector, and the action is the PID constants—three-dimensional values that will fulfill the controller parameters. The reward is the inverse of the difference between the target velocity vector and the quadcopter velocity vector.

The agent's quadcopter starts at the origin, the `target_velocity` is applied to the quadcopter after the controller settings. The controller settings come from the action chosen by the agent. The agent aims to minimize the difference between the target velocity vector and the quadcopter velocity vector.

## Objective

Train a reinforcement learning (RL) agent to determine optimal PID coefficients so that a drone can achieve a desired target velocity as quickly and accurately as possible, even from varying initial velocities.

## Environment and Simulation

### State (Observation):

- **Target Velocity**: A 4D vector representing the desired velocity of the drone (3D direction + magnitude). This will be randomized across episodes and also within episodes after each successful velocity achievement.
- **Current Velocity**: The drone's current velocity to give context to the agent.
- **External Factors (if applicable)**: Factors like wind speed and direction that might influence the drone's movement.

### Action:

- The RL agent will decide on PID coefficients for the drone controller. These coefficients will be applied to the drone's PID controller to achieve the desired velocity.

### Reward:

- The reward is given at regular intervals or steps, calculated as the negative of the difference between the achieved velocity and the target velocity. The closer the drone's velocity is to the target velocity, the higher the reward.
- Additional rewards or penalties can be introduced based on the drone's behavior during the interval, e.g., penalizing extreme PID values or erratic behaviors.
- A bonus reward can be provided each time the drone successfully reaches the target velocity, prompting the generation of a new target.

### Episode Termination:

- Each episode has a maximum duration or step count. If the drone can achieve multiple target velocities within this timeframe, the episode continues.
- Episodes can also be terminated early if the drone exhibits unsafe or erratic behavior or if it fails to achieve the target velocity within a specified time frame or number of steps.

## Additional Considerations:

- **Yaw Orientation**: The drone will maintain its current yaw orientation when adjusting its velocity, preventing unnecessary rotations. However, this can be made flexible to test the robustness of learned policies.
- **Safety Constraints**: Ensure mechanisms are in place to prevent the application of extreme or unsafe PID values. Early episode termination should be considered if the drone behaves erratically.
- **Target Velocity**: Depending on the application specifics, consider expressing the target velocity as a relative change from the current velocity.
- **Logging and Visualization**: Important metrics, like the difference between target and achieved velocities or the magnitude of PID values, should be logged and visualized to monitor the learning progress.
- **Safety Mechanisms**: Ensure that there are safety checks in place for the RL agent's actions. For example, the PID coefficients that the agent can set should be within reasonable bounds to prevent erratic or unsafe behavior.
- **Logging**: Consider adding logging throughout the simulation and environment. This will help in debugging and understanding the behavior of the RL agent as it interacts with the environment.
- **Visualization**: If possible, provide a visualization of the drone's movement, target velocity, and achieved velocity. This visual feedback can be extremely useful in understanding how well the RL agent is performing.
- **Error Handling**: Add error handling mechanisms, especially for critical sections of the code. This will make your simulation robust to unexpected issues.
- **Performance**: The RL training process can be computationally intensive. Consider profiling your code to identify bottlenecks and optimize those parts.
- **Testing**: Ensure that you have tests in place, especially for the critical functions. This ensures that as you make changes, you don't inadvertently introduce bugs.
- **Interactivity**: If you're planning for manual interaction, this can be used for initial testing before deploying the RL agent or for comparison between human and RL agent performance.
- **Configurability**: Consider using configuration files (like YAML or JSON) for setting environment parameters. This way, you can easily adjust parameters without modifying the code directly.
- **Variable Naming**: Ensure variable names are descriptive. Short variable names can be hard to understand out of context. Consider more descriptive names.
- **State Initialization**: Ensure that the drone's initial state (position, velocity, etc.) is consistently initialized at the beginning of each episode. This consistency is crucial for the RL agent to learn effectively.
- **Reward Engineering**: Monitor the agent's performance and be prepared to adjust the rewards if necessary.



It was chosen 100.000 to be the maxium gain of the pid coefficients.