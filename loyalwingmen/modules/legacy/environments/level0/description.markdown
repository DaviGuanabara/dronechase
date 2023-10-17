# **Environment Name**: Level1

## **Development Status**: 

Finished. 
The environment is ready to be used with RL libraries, particularly Stable Baselines 3 (version 2.0.0 or higher).

## **Purpose**:
The `Level1` class simulates an engagement scenario between one Loyal Wingman and one Loitering Munition. It's designed for compatibility with Reinforcement Learning (RL) libraries, particularly Stable Baselines 3 (version 2.0.0 or higher).
In this level, a loyalwingman has to engage a loitering munition that is passing through the environment area. 
**PHYSICS AND GRAVITY ARE DISABLED**, so, the action, the direction and intensity of the velocity, is directly applied to the quadcopter by the
 'pybullet reset base velocity'. 


## **Inspiration**:
The design is inspired by the [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones) project on GitHub, and the CTEDS project by the Instituto Tecnológico de Aeronáutica.

## **Key Features**:

Environment Class is a subclass of the `gym.Env` class from the OpenAI Gym library. It is compatible with RL libraries, particularly Stable Baselines 3 (version 2.0.0 or higher). It is a interface between the class simulation, where the pybullet engine is running, and the RL library.
The environment Class uses the simulation class as a component.

As the simulation class and the environment class are closed related, there are 1 to 1, 1 simulation to 1 environment.
The RL is in 30hz, and the environment is running in 240hz.

### 1. **Initialization**:
   - Set simulation and RL frequencies.
   - Toggle graphical user interface (GUI) for visualizations.
   - Activate debug mode for insights and troubleshooting.


   - setup_pybullet_DIRECT and setup_pybulley_GUI methods: Connect to the pybullet physics engine in either direct mode or GUI mode.
   - gen_initial_position: Generate random positions within the dome.
   - _housekeeping: Some initialization tasks for the entities.
   - reset: Resets the simulation environment to its initial state.

### 2. **Physics and Dynamics**:
   - Uses the PyBullet engine for drone dynamics.
   - Environment restricted to a specified dome radius.

### 3. **Action Space**:
   - Actions: direction and intensity for velocity (direction: -1 to 1, intensity: 0 to 1).

### 4. **Observation Space**:
   - Observations: Loyal Wingman's inertial data, Loitering Munition's inertial data, direction to target, distance to target, and last action of Loyal Wingman.

### 5. **Environment Interaction**:
   - `reset()`: Reinitialize the environment for a new starting state.
   - `step(rl_action)`: Progress the simulation by one time step based on the RL action.
   - `close()`: Safely terminate the environment.

### 6. **Utilities**:
   - Retrieve the PyBullet Client ID.
   - User-friendly keymap for drone control via keyboard inputs.

### 7. Reward Computation:

compute_reward: Computes the reward for the current state of the simulation. The reward is based on the distance between the Loyal Wingman and the Loitering Munition. There are bonuses and penalties:
Bonus if the Loyal Wingman is getting closer to the Loitering Munition.
Score based on how close the Loyal Wingman is to the Loitering Munition.
Large bonus if the distance between them is less than 0.2 units.
Penalty if the distance exceeds the dome radius.
The final reward is score + bonus - penalty.

**Disclaimer**: The developer notes that parts of the code might seem unoptimized due to time constraints.


### Aggregate Physics Steps (`aggregate_steps`)

In the simulation environment, there's a concept of advancing time in the physics engine. In real-world scenarios, drones and other entities change their states (like position, velocity, etc.) continuously over time. However, in a simulated environment, this continuous change is approximated by "stepping" through time in small increments. 

The `aggregate_steps` refers to the number of these small increments (or "steps") that the simulation advances for every call to the `step()` method of the environment.

#### Why is it Important?

- **Higher Frequency Approximation**: By aggregating multiple steps, the simulation can more accurately approximate the continuous behavior of entities in the real world. Each of these steps computes the new state of the system based on the physics equations.

- **Performance vs Accuracy Trade-off**: There's a trade-off between the number of aggregate steps and computational performance. More steps mean more accurate simulation but at the cost of computational resources.

In the provided code, every time the `step()` function is called, the simulation is advanced by `aggregate_physics_steps` times using the line:

```python
for _ in range(self.environment_parameters.aggregate_physics_steps):
    p.stepSimulation()
```

## **Future Steps**
The next levels will have physics and gravity enabled. The action will be the RPMs of the motors, and the reward will be based on the distance to the target and the time to reach it. There will be plenty of loyalwingmen and loitering munition, and a building where loitering munition desire to destroy.
The Observation will be the lidar. So, lots of stuff coming soon.