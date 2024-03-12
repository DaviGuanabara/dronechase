This project was made for a dissertation. If you have any questions, do not hesitate to contact me.
I want to start by apologizing for a mess code, typos, or any library missing. It will be improved soon.
This project was inspired in PyFlyt https://github.com/jjshoots/PyFlyt (I had mistakenly imported its whole project into the folder PyFlyt. In fact, I only need the QuadX library. This dependency will be fixed soon.)

## Folders

This project has an application folder, apps, and the environment implementation folder, loyal wingmen/environments. So, each level in apps has its equivalent in loyalwingmen/environments.
The folder environments hold all the implementation specific to that level. Shared implementations are in loyalwingmen folder.


Level 1 is focused on a proof of concept. It uses a simple observation space to show that it is working.
Level 2 is equivalent to Stage 1, where there are two loyal wingmen (ally not moving) and one loitering munition (not moving)
Level 3 is equivalent to Stage 2, where there are two loyal wingmen (ally not moving) and many loitering munitions (not moving)
Level 4 is equivalent to Stage 3, where 7 experiments were implemented.

To reduce the code, the term pursuer was preferred as a relatively loyal wingman and invader as loitering munition.

## Execution

In the apps folder, you can find the application levels. For example, level 1 has the files: 

interactive.py: here, you can control the loyal wingman through the keyboard (W, S, and Arrows).
learn.py: here, the loyal wingman will learn how to proceed.
load.py: here, the model generated from learn.py will be executed.

## Installation.

in the main folder:

```shell
pip install -e .
```

## Informations

Quadricopter -> QuadX (PyFlyt) -> PID (PyFlyt)
Environment -> Simulation -> PyBullet


# Level 1 (Stage 2) Environment Overview

This environment is structured to create a simulation where two entities interact: a **pursuer** and an **invader**. Specifically, it's a scenario in which a pursuer quadcopter engages an invader quadcopter. The objective seems to be for the pursuer to get close enough to the invader (within a `CATCH_DISTANCE`). At this point, the invader is possibly considered 'caught' or 'engaged' and is reset to a new random position.

## File Structure:

```
.
├── __pycache__
│   ├── pyflyt_level1_environment.cpython-38.pyc
│   ├── pyflyt_level1_simulation.cpython-38.pyc
│   └── quadcopter_manager.cpython-38.pyc
├── components
│   ├── __pycache__
│   │   ├── normalization.cpython-38.pyc
│   │   ├── pyflyt_level1_simulation.cpython-38.pyc
│   │   └── quadcopter_manager.cpython-38.pyc
│   ├── normalization.py
│   ├── pyflyt_level1_simulation.py
│   ├── quadcopter_manager.py
│   ├── test_quadcopter_manager.py
│   └── test_simulation.py
├── project_structure.py
└── pyflyt_level1_environment.py
```

## Key Components:

- **PyflytL1Enviroment**: This is the main class that defines the environment. It initializes the two entities, defines action and observation spaces, computes rewards based on their distances, and checks termination conditions.

- **AviarySimulation**: Responsible for the overarching simulation environment. It seems to be influenced or inspired by `gym-pybullet-drones`.

- **QuadcopterManager**: Manages the various quadcopters in the environment, namely the pursuer and the invader. It can spawn, drive, update, and reset them.

- **Quadcopter**: This class represents a quadcopter entity in the environment. The quadcopter's physics and behaviors are simulated using the `quadx pyflyt` class.

- **normalize_inertial_data**: This utility function normalizes the quadcopters' inertial data, producing observations within the defined space.

This environment is structured with compatibility in mind, particularly with the Stable Baselines 3 reinforcement learning framework.

## CTEDS pybullet Environment - Level 1: Cooperative Threat Engagement with Heterogeneous Drone Swarm

**Purpose**: The primary focus is on "loyal wingmen" quadcopters defending structures from threats embodied by loitering munitions. Key architectural features include:

- **Loitering Munition AI**: This AI navigates and engages based on a Machine State.
- **Loyalwingmen AI**: Operates with a Behavior Tree. Within this tree, an 'engage' behavior exists, which uses a neural network for decision-making.

**Training Goal**: The primary objective is refining the neural network within the 'engage' behavior for optimal real-world performance. The environment will be layered, allowing complexity to be gradually introduced until the final design is achieved.

**Simulation Details**:

- **Physics Engine**: Managed by pybullet.
- **Drone Model**: cf2x (Crazy Fly in X motor position).

**Upcoming Features**: Future iterations will add more features, progressing towards the complete CTEDS environment.

**Note**: This project simulates the interactions between quadcopters in a defense scenario where structures are protected from loitering munitions.

Engagement Criteria for CTEDS pybullet Environment - Loyalwingmen vs. Loitering Munition

Criteria:
Engagement success is determined by the loyal wingman's proximity to the loitering munition. It is successful when the distance between the two is less than CATCH_DISTANCE.

Python Function Representation:

```python
def replace_invader_if_close(self):
invader = self.quadcopter_manager.get_invaders()[0]
pursuer = self.quadcopter_manager.get_pursuers()[0]

invader_position = invader.inertial_data["position"]
pursuer_position = pursuer.inertial_data["position"]

if np.linalg.norm(invader_position - pursuer_position) < self.CATCH_DISTANCE:
    position = (self.dome_radius / 2) * np.random.uniform(-1, 1, 3)
    attitude = np.zeros(3)
    motion_command = np.random.uniform([-1, -1, -1, 0], [1, 1, 1, 0.4], 4)
    self.quadcopter_manager.replace_quadcopter(invader, position, attitude)
    invader.drive(motion_command)
```

Explanation:

The loitering munition and loyal wingman are represented by the names invader and persuader, respectively.

The function checks the distance between the two quadcopters.

If this distance exceeds CATCH_DISTANCE, the invader is repositioned within the environment.

A motion command is then given to the invader, initiating movement upon its new placement.

### Physics & Environment Details:

- **Wind & Obstacles**: Not present in the current level but planned for future iterations.
- **Catch Distance**: The predefined threshold determining successful engagement between the persuader and the invader. Current value is set at 1.5 meters.

### Physics & Environment Details:

- **Wind & Obstacles:** No wind or obstacle simulations exist. And there's no intention to incorporate them in the foreseeable future.
- **Quadcopter Control:** The QuadX utilizes a PID controller to regulate the motors' RPM. It takes in the desired velocity and translates this into RPMs. These RPMs are then passed to the motors, translating into respective forces and torques applied to each motor.

### QuadX Class Implementation:

The QuadX class represents a drone in the QuadX configuration and is responsible for controlling and simulating the drone's physics.

- **Initialization**: The class is initialized with parameters like the starting position, orientation, control frequency, physics frequency, camera settings, etc.

- **Motor Control**:

  - Motor IDs correspond to the quadrotor X in PX4, with a layout in the diagram within the code.
  - The `Motors` class handles motor properties and behaviors such as thrust coefficients, torque coefficients, max RPM, and noise ratios.

- **Drag Simulation**:

  - A drag coefficient is specified to simulate the drag on the main drone body.
  - The `BoringBodies` class is responsible for simulating this drag.

- **Control Parameters**:

  - Control parameters such as the proportional, integral, and derivative coefficients for angular velocity, angular position, linear velocity, and linear position are specified. These parameters are essential for the drone's PID controllers.
  - Separate PIDs for height (z position and z velocity) are implemented.

- **Camera**:
  If the drone is initialized with a camera (`use_camera=True`), the `Camera` class handles the camera's properties and behaviors. This includes gimbal support, field of view, camera angle, and resolution.

The above description provides an overview of the QuadX class and how it fits within the larger simulation. It incorporates various other classes to achieve a holistic drone simulation, from motor behaviors to camera functionalities.

The Motors class is designed to simulate an array of brushless motors, primarily found on drones. It offers a detailed representation of how motor characteristics and control signals are transformed into real-world physics actions:

Initialization: The class takes in parameters related to each motor's physical characteristics, such as the maximum RPM, thrust coefficient, torque coefficient, and thrust direction (unit vector).

Translation of Control Signals: The physics_update method accepts PWM signals (ranging from -1 to 1) for each motor. This method updates the throttle of each motor based on these PWM signals and the motor's ramp time constant (tau). Noise is also added to the throttle based on a specified noise ratio.

RPM Computation: Inside the \_jitted_compute_thrust_torque method, the throttle values are translated to RPM values by multiplying them with the maximum RPM for each motor.

Force and Torque Computation: The thrust and torque exerted by each motor are computed based on their RPMs. The formula used is quadratic, implying that the thrust and torque are proportional to the square of the RPM. The thrust is computed by multiplying the squared RPM with the thrust coefficient and direction. Similarly, the torque is derived from the squared RPM, the torque coefficient, and the thrust direction.

Application of Forces and Torques: The computed thrust and torque values are then applied to the respective motors, influencing the drone's physics simulation.

In essence, the Motors class serves as an intermediary, bridging the gap between the desired control signals (PWM values) and the real-world physics interactions (forces and torques) in a simulated environment.

1. Actions and Observations:

Actions: It's a four-dimensional numpy array, where the first three dimensions specify a direction vector, and the fourth dimension gives the magnitude of the velocity. This action can be considered a velocity vector (with its direction and magnitude) that the pursuer drone should adopt in the next step.

Observations: The observation data encapsulates the inertial information for the pursuer (the agent) and the invader (the target). This likely includes positional and velocity information for both drones. Additionally, observations might also provide the relative distance and direction between the pursuer and the invader, as well as the last action taken by the agent.

2. Reward Mechanism (compute_reward method):

This method calculates the agent's reward based on the environment's current state. Several factors influence the reward mechanism:

Base Reward: The base reward (score) is the difference between the dome's radius and the current distance between the pursuer and the invader. This means the closer the pursuer is to the invader, the higher the base reward.

Bonus for Movement Anticipation: If the current distance between the pursuer and the invader is less than the distance at the previous time step (last_distance), the pursuer has moved closer to the invader. This results in a bonus reward proportional to the distance reduction.

Proximity Bonus: If the distance between the pursuer and the invader becomes less than a defined threshold (CATCH_DISTANCE), a maximum bonus (MAX_REWARD) is awarded. This suggests that the pursuer is close to catching or has caught the invader.

Penalty for Leaving the Dome: If the pursuer goes outside the boundary defined by the dome radius, a penalty equivalent to the maximum reward (MAX_REWARD) is deducted. This encourages the agent to stay within the operational area.

The final reward is computed as the sum of the base reward and the bonuses minus any penalties.


# Level 2 Environment Overview

Building on the foundational principles of the CTEDS pybullet project's Level 1, Pyflyt Level 2 Environment enriches the simulation experience. Here, the central interaction focuses on a pursuer quadcopter tracking an invader. Level 2, however, infuses added layers of realism, aiming to mirror potential real-world scenarios closely.

## Key Additions in Level 2:

Level 2 stands out with the inclusion of a simulated LIDAR system. This system amplifies the quadcopters' ability to identify threats or invaders, showcasing a more genuine portrayal of real-world quadcopter threat detection mechanisms.

## Structural Overview:

While Level 2 retains many core elements of Level 1, it adds depth to the experience:

- **Physics Engine:** pybullet
- **Drone Model:** cf2x (Crazy Fly in X motor position)

## The Role of Simulated LIDAR:

For the pursuer quadcopters, LIDAR is a game-changer. It provides precise distance and angle measurements, giving the pursuer comprehensive environmental insights. This new layer of information could be instrumental in shaping decision-making.

## LIDAR Data Points:

- **Distance Measurements:** Current distance details concerning nearby objects and threats.
- **Angle Measurements:** Directional information of detected entities about the pursuer.

## Interaction Dynamics:

While foundational action and observation mechanics remain, LIDAR data introduces a new dimension:

- **Actions:** A 4-dimensional numpy array defines the pursuer drone's direction and speed magnitude.
- **Observations:** Alongside the traditional inertial data for the pursuer and invader, Level 2 integrates LIDAR distance and angle metrics. This adds a layer of spatial intelligence, enhancing the pursuer's strategic choices.

## Reward System:

Level 2 adopts Level 1's reward mechanics, accounting for essential rewards, bonuses, and penalties. 

## Observation:

The integration of LIDAR paves the way for more intricate observation dynamics, taking into account LIDAR precision, threat detection efficiency, and LIDAR-influenced decisions.

## Final Thoughts:

Level 2, with its simulated LIDAR system, enhances the pursuer's observational prowess, creating new avenues and hurdles in training the neural network within the 'engage' mode. As training evolves, the aim is for quadcopters to adeptly leverage LIDAR insights, boosting engagement success and laying a robust groundwork for future environment levels.

## Delving Deeper: Conceptual Insights & Real-World LIDAR Analogies:

- **Snapshotting the Environment:** Think of the `buffer_inertial_data` function as a photographer taking a complete shot of the scene. This function captures the environment, echoing the initial data capture phase.
- **Data Processing & Simulating Laser Scans:** The `update_data` function develops the buffered snapshot. It mimics the process of a real-world LIDAR sending lasers and noting their return times and reflections to chart distances and object placements.
- **3D Environment Representation:** The `self.sphere` tensor, structured as `self.sphere[channel][theta][phi]`, translates into a digital 3D map. Here:
  - **channel:** Comparable to different channels in an image, these represent varied data types.
  - **theta & phi:** Reflect the spherical coordinates, paralleling the angles a real-world LIDAR would use to get a 360-degree view.
- **Accessing the 3D Map:** The `Read Data` function facilitates extracting this 3D map, reminiscent of pulling out a processed point cloud in a real LIDAR system.
  
### Analogies:

- Buffering mirrors old-school cameras capturing light on film, freezing a moment.
- The `update_data` function is like film development, where the buffered environment is "processed" to yield a 3D map.
- The `self.sphere` structure can be visualized as a multi-layered panorama. Each layer or channel holds distinct info, and the full array offers a 360-degree environmental view.

The subsequent sections provide a deeper insight into the LIDAR class's structure, the mathematical basis for spherical coordinates, the initialization of the LIDAR system, dividing space, the parallels between the celestial sphere and the LiDAR class's conceptualization, and the embedding of entity identification within the LiDAR class.

## Training Results Overview

Our training data is structured in a unique space, where multiple inputs are transformed and processed to yield the desired outcomes. Let's dive into the specifics:

### **Observation Space**

The observation space is an intricate dictionary with the following components:

1. **Lidar Readings**: 
   - Spanning 360 degrees, it has 2 channels.
   - Dynamically sized as set by the user.
   - The first channel denotes the distance to the nearest object in a specific direction.
   - The second channel showcases the angle to that object.

2. **Inertial Readings**:
   - A 12-value set consisting of x, y, and z coordinates along with velocity (vx, vy, vz), angular orientations (roll, pitch, yaw), and angular rates (roll rate, pitch rate, yaw rate).

3. **Last Action**:
   - A 4-dimensional array with a 3D direction and a 1D intensity.

These inputs undergo processing via a feature extractor within the MultiInputMLP PPO policy.

### **Feature Extractors**

Here's how each of the observation space components is processed:

- **Lidar Feature Extractor**:
``` bash
Sequential(
(0): Conv2d(2, 32, kernel_size=(4, 4), stride=(4, 4))
(1): ReLU()
(2): Conv2d(32, 64, kernel_size=(2, 2), stride=(2, 2))
(3): ReLU()
(4): Flatten(start_dim=1, end_dim=-1)
)
```

- **Inertial Feature Extractor**:
``` bash
Sequential(
(0): Linear(in_features=12, out_features=128, bias=True)
(1): ReLU()
(2): Linear(in_features=128, out_features=128, bias=True)
(3): ReLU()
(4): Linear(in_features=128, out_features=128, bias=True)
(5): ReLU()
)
```

- **Action Feature Extractor**:
``` bash
Sequential(
(0): Linear(in_features=4, out_features=128, bias=True)
(1): ReLU()
(2): Linear(in_features=128, out_features=128, bias=True)
(3): ReLU()
(4): Linear(in_features=128, out_features=128, bias=True)
(5): ReLU()
)
```

Post-processing, these features are collated into a singular vector with 448 dimensions. This vector undergoes another transformation to adjust to the `features_dim` dimensions. The transformed data is then inputted into an MLP neural network.


- **Resultant Feature Extractor**:
  
  the in features of 448 is the sum of the output of the three feature extractors above for lidar with a resolution of 16. The output of this feature extractor is then fed into the MLP neural network.


``` bash
(final_layer): Sequential(
      (0): Linear(in_features=448, out_features=512, bias=True)
      (1): ReLU()
    )
```

### **MLP Neural Network Details**

This MLP network consists of three hidden layers, with possible neuron configurations of 128, 256, or 512. The output layer comprises 4 neurons corresponding to each action. Remember, a higher average score indicates superior performance. The total training time is 2 million timesteps.


A higher average score is better.
2 million timesteps

| hidden_1 | hidden_2 | hidden_3 | rl_frequency | learning_rate | batch_size | features_dim | avg_score   | std_deviation |
| -------- | -------- | -------- | ------------ | ------------- | ---------- | ------------ | ----------- | ------------- |
| 512      | 256      | 512      | 15           | 0.001         | 128        | 512          | 6288.217773 | 2388.927979   |
| 128      | 128      | 512      | 15           | 0.0001        | 128        | 512          | 1988.519897 | 1370.796143   |
| 256      | 256      | 128      | 15           | 0.001         | 512        | 512          | 7451.65625  | 2230.62207    |
| 256      | 512      | 128      | 15           | 0.0001        | 256        | 512          | 151.1664734 | 2943.010498   |
| 512      | 512      | 512      | 15           | 0.0001        | 1024       | 512          | 2076.185791 | 1020.19342    |
| 256      | 512      | 512      | 15           | 0.0001        | 1024       | 512          | 1231.162842 | 533.1439819   |



# Level 3 (Stage 2) Environment Overview

At the inception of this project, I committed to meticulously documenting the specifications of each level, adhering to a predefined blueprint. However, as the project evolved, it became more of an exploratory adventure rather than a linear progression of steps. This shift was particularly evident by the time I reached Level 3.

#### Initial Objectives vs. Realized Complexity

Originally, Level 3 was envisioned with a more straightforward objective, but its complexity unfolded unexpectedly as development progressed. This level was designed to feature 1 Loyal Wingman (LW, the pursuer) and 'n' Loitering Munitions (LMs, the invaders), with 'n' defaulting to 2. The goal for the invader is to detonate within a 0.2-meter proximity of the pursuer, while the LW's objective is to neutralize all LMs. The accuracy of LW's attacks is set at 90%, with an effective range of 1 meter.

Unexpectedly, the emergent behavior was a straightforward "go straight" tactic, where the LW would attempt to maintain a safe distance from LMs to engage them effectively without falling victim to their explosions. While effective in eliminating the invaders and ensuring the LW's survival, this strategy diverged from the desired behavior of maintaining proximity to the origin and avoiding the risk of exiting the operational dome.

#### Adjustments and Next Steps

To align the emergent behavior with the desired outcome, I considered several adjustments, such as reducing the dome's size, which might require recalibration of reward and penalty scales. Ultimately, I opted to introduce a new level with additional constraints and challenges to address these discrepancies.

#### Technical Implementations

- **TaskProgress Class**: Initially intended to facilitate Curriculum Learning, this structure will be repurposed to suit the project's evolving needs.

- **Observer Design Pattern**: The implementation of this pattern proved invaluable, particularly for leveraging LIDAR data and ensuring synchronization between components like the GUN class and its cooldown mechanisms.

- **ARM and DISARM Quadcopters**: This approach significantly optimized simulation efficiency, increasing the iteration speed from 70 iterations per second (it/s) to 400 it/s. Armed quadcopters are active within the simulation, while disarmed ones are excluded from the simulation dynamics.

#### L3AviarySimulation Class

This class forms the backbone of the PyFlyt system for UAV management within the PyBullet simulation environment. It encompasses functionalities crucial for physics stepping, setpoint handling, collision tracking, and more.

- **Inheritance**: Inherits from `bullet_client.BulletClient` of the PyBullet library, providing an interface to the physics simulation.

- **Initialization**: Configures the simulation environment with options for visual rendering, physics loop frequency, simulation scale, and an optional random seed.

- **Key Methods**: Include `reset` for environment reinitialization, `step` for advancing the simulation, and methods for managing active drones.

- **Attributes**: Maintains a dictionary of active drones and provides evaluation and performance profiling mechanisms.

#### PyflytL3Environment Class

They are designed for simulations involving a Loyal Wingman and Loitering Munition, compatible with Stable Baselines 3 (version 2.0.0 or higher).

- **Initialization**: Sets the dome radius, reinforcement learning frequency, and GUI preferences.

- **Functionality**: Offers comprehensive methods for simulation management, including subscriber notifications, component initialization, frequency adjustments, and debug text management.

- **Simulation Control**: Facilitates environment resetting, step progression, action space definition, observation computation, and more, ensuring a robust simulation framework.



# Level 4 (Stage 3) Environment Overview

Level 4 is the aim design.
