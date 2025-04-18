# Terminology Glossary

`CTEDS pybullet project`: A simulation project focusing on the concept of 'loyalwingmen' quadcopters defending a building against threats posed by loitering munitions.

`Loyalwingmen`: Quadcopters designed to defend a specified target, such as a building or structure.

`Loitering Munition AI`: An artificial intelligence system designed to navigate and engage targets based on a Machine State.

`Behavior Tree`: A hierarchical, graphical representation of decision-making logic used in AI systems, often in game development and robotics.

`Engage Behavior`: A specific behavior within the Behavior Tree that defines how the AI will engage or confront threats.

`pybullet`: A physics engine used for simulations, particularly in robotics and machine learning environments.

`cf2x (Crazy Fly in X motor position)`: A specific drone model with motors positioned in an "X" configuration.

`Persuer`: A quadcopter designated to chase or engage another quadcopter, typically an invader.

`Invader`: A quadcopter representing a threat or target, often tasked with invading or reaching a particular point.

`CATCH_DISTANCE`: A predefined distance threshold determining when a persuer has successfully engaged an invader.

`AviarySimulation`: An overarching simulation environment, possibly inspired by gym-pybullet-drones.

`QuadcopterManager`: A class or module responsible for managing the various quadcopters in a simulation.

`normalize_inertial_data`: A utility function used to normalize inertial data of quadcopters, helping in producing standardized observations.

Stable Baselines 3`: A Python library containing implementations of reinforcement learning algorithms.

``GYMNASIUM`: Likely a typographical error for "Gym," a toolkit by OpenAI for developing and comparing reinforcement learning algorithms. It could also refer to a new or custom library.

`PYFLYT`: A simulation or utility package used for UAV or drone simulations, as inferred from the context.

`Dome Radius`: A boundary within which quadcopters operate in the simulation environment.

`PID Controller`: A control loop mechanism employed in control systems. PID stands for Proportional, Integral, and Derivative, which are the three types of actions performed by the controller.

`PWM`: Pulse Width Modulation. A technique used to produce analog signals using a digital source.

`MAX_REWARD`: A predefined maximum reward value, often given when an agent achieves its primary objective in a reinforcement learning environment.

# Purpose:

This environment is the foundational layer of the CTEDS pybullet project, which focuses on the concept of the 'loyalwingmen' quadcopters defending a building against threats posed by loitering munitions, which are also represented by quadcopters. The unique architecture of this environment encompasses:

- **Loitering Munition AI**: Utilizes a Machine State to navigate and engage.
- **Loyalwingmen AI**: Operates based on a Behavior Tree, within which an 'engage' behavior exists. This behavior will employ a neural network for decision-making processes.

Training Goal:
The overarching objective is to train the neural network within the 'engage' behavior, ensuring optimal performance in real-world threat engagement scenarios. To facilitate effective training, the environment will be built across multiple levels, progressively approaching the comprehensive final design.

Simulation Details:

- **Physics Engine**: Leveraged by pybullet.
- **Drone Model**: cf2x (Crazy Fly in X motor position).

Upcoming Features:
Future levels will incrementally integrate more components and complexities leading to the fully realized CTEDS environment.

Note:
This project aims to simulate quadcopters within the context of the 'loyalwingmen' concept defending structures from loitering munitions, establishing a nuanced and strategic interaction between the two entities.

# Pyflyt Level 1 Environment

This environment is structured to create a simulation where two types of entities interact: a **persuer** and an **invader**. Specifically, it's a scenario in which a persuer quadcopter is tasked to engage an invader quadcopter. The objective seems to be for the persuer to get close enough to the invader (within a `CATCH_DISTANCE`), at which point the invader is possibly considered as 'caught' or 'engaged', and is subsequently reset to a new random position.

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

- **QuadcopterManager**: Manages the various quadcopters in the environment, namely the persuer and the invader. It's capable of spawning, driving, updating, and resetting them.

- **Quadcopter**: This class represents a quadcopter entity in the environment. The quadcopter's physics and behaviors are simulated using the `quadx pyflyt` class.

- **normalize_inertial_data**: A utility function used for normalizing the inertial data of the quadcopters, aiding in producing observations that fit within the defined space.

This environment is structured with compatibility in mind, particularly with the Stable Baselines 3 reinforcement learning framework.

## CTEDS pybullet Environment - Level 1: Cooperative Threat Engagement with Heterogeneous Drone Swarm

**Purpose**: The primary focus is on "loyalwingmen" quadcopters defending structures from threats, embodied by loitering munitions. Key architectural features include:

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
Engagement success is determined by the proximity of the loyalwingman to the loitering munition. Engagement is considered successful when the distance between the two is less than CATCH_DISTANCE.

Python Function Representation:

```python
def replace_invader_if_close(self):
invader = self.quadcopter_manager.get_invaders()[0]
persuer = self.quadcopter_manager.get_persuers()[0]

invader_position = invader.inertial_data["position"]
persuer_position = persuer.inertial_data["position"]

if np.linalg.norm(invader_position - persuer_position) < self.CATCH_DISTANCE:
    position = (self.dome_radius / 2) * np.random.uniform(-1, 1, 3)
    attitude = np.zeros(3)
    motion_command = np.random.uniform([-1, -1, -1, 0], [1, 1, 1, 0.4], 4)
    self.quadcopter_manager.replace_quadcopter(invader, position, attitude)
    invader.drive(motion_command)
```

Explanation:

The loitering munition and loyalwingman are represented by invader and persuer respectively.

The function checks the distance between the two quadcopters.

If this distance is less than CATCH_DISTANCE, the invader is repositioned within the environment.

A motion command is then given to the invader, initiating movement upon its new placement.

### Physics & Environment Details:

- **Wind & Obstacles**: Not present in the current level but planned for future iterations.
- **Catch Distance**: The predefined threshold which determines successful engagement between the persuer and the invader. Current value set at 1.5 meters.

### Physics & Environment Details:

- **Wind & Obstacles:** There are currently no wind or obstacle simulations. And there's no intention to incorporate them in the foreseeable future.
- **Quadcopter Control:** The QuadX utilizes a PID controller to regulate the motors' RPM. It takes in the desired velocity and translates this into RPMs. These RPMs are then passed to the motors, which in turn translate them into respective forces and torques applied to each motor.

### QuadX Class Implementation:

The QuadX class represents a drone in the QuadX configuration and is responsible for controlling and simulating the physics of the drone.

- **Initialization**: The class is initialized with parameters like the starting position, orientation, control frequency, physics frequency, camera settings, and more.

- **Motor Control**:

  - Motor ids correspond to the quadrotor X in PX4, with a layout represented in the diagram within the code.
  - The `Motors` class is utilized to handle motor properties and behaviors such as thrust coefficients, torque coefficients, max RPM, and noise ratios.

- **Drag Simulation**:

  - A drag coefficient is specified for simulating the drag on the main drone body.
  - The `BoringBodies` class is responsible for simulating this drag.

- **Control Parameters**:

  - Control parameters such as the proportional, integral, and derivative coefficients for angular velocity, angular position, linear velocity, and linear position are specified. These parameters are essential for the drone's PID controllers.
  - Separate PIDs for height (z position and z velocity) are implemented.

- **Camera**:
  - If the drone is initialized with a camera (`use_camera=True`), the `Camera` class is used to handle the camera properties and behaviors. This includes features like gimbal support, field of view, camera angle, and resolution.

The above description provides an overview of the QuadX class and how it fits within the larger simulation. It incorporates various other classes to achieve a holistic drone simulation, from motor behaviors to camera functionalities.

The Motors class is designed to simulate an array of brushless motors, primarily found on drones. It offers a detailed representation of how motor characteristics and control signals are transformed into real-world physics actions:

Initialization: The class takes in parameters related to each motor's physical characteristics, such as the maximum RPM, thrust coefficient, torque coefficient, and thrust direction (unit vector).

Translation of Control Signals: The physics_update method accepts PWM signals (ranging from -1 to 1) for each motor. This method updates the throttle of each motor based on these PWM signals and the motor's ramp time constant (tau). Noise is also added to the throttle based on a specified noise ratio.

RPM Computation: Inside the \_jitted_compute_thrust_torque method, the throttle values are translated to RPM values by multiplying them with the maximum RPM for each motor.

Force and Torque Computation: The thrust and torque exerted by each motor are computed based on their RPMs. The formula used is quadratic in nature, implying that the thrust and torque are proportional to the square of the RPM. The thrust is computed by multiplying the squared RPM with the thrust coefficient and the thrust direction. Similarly, the torque is derived from the squared RPM, the torque coefficient, and the thrust direction.

Application of Forces and Torques: The computed thrust and torque values are then applied to the respective motors, thereby influencing the physics simulation of the drone.

In essence, the Motors class serves as an intermediary, bridging the gap between the desired control signals (PWM values) and the real-world physics interactions (forces and torques) in a simulated environment.

1. Actions and Observations:

Actions: It's a 4-dimensional numpy array where the first three dimensions specify a direction vector, and the fourth dimension gives the magnitude of the velocity. In essence, this action can be thought of as a velocity vector (with its direction and magnitude) that the pursuer drone should adopt in the next time step.

Observations: The observation data encapsulate the inertial information for both the pursuer (the agent) and the invader (the target). This likely includes positional and velocity information for both drones. Additionally, observations might also provide the relative distance and direction between the pursuer and the invader, as well as the last action taken by the agent.

2. Reward Mechanism (compute_reward method):

This method calculates the reward for the agent based on the current state of the environment. The reward mechanism is influenced by several factors:

Base Reward: The base reward (score) is the difference between the dome's radius and the current distance between the pursuer and the invader. This means the closer the pursuer is to the invader, the higher the base reward will be.

Bonus for Movement Anticipation: If the current distance between the pursuer and the invader is less than the distance at the previous time step (last_distance), it means the pursuer has moved closer to the invader. This results in a bonus reward which is proportional to the reduction in distance.

Proximity Bonus: If the distance between the pursuer and the invader becomes less than a defined threshold (CATCH_DISTANCE), a maximum bonus (MAX_REWARD) is awarded. This suggests that the pursuer is very close to catching or has caught the invader.

Penalty for Leaving the Dome: If the pursuer goes outside the boundary defined by the dome radius, a penalty equivalent to the maximum reward (MAX_REWARD) is deducted. This discourages the agent from straying too far from the operational area.

The final reward is computed as the sum of the base reward and the bonuses, minus any penalties.

Dependencies and External Libraries
The following libraries and dependencies were used in the creation and functioning of this project:

SB3: Stable Baselines3 (SB3) is a set of high-quality implementations of reinforcement learning algorithms in Python. It provides utilities for building and training custom RL agents.

GYMNASIUM: While "GYMNASIUM" is not a recognized standard library in the context of reinforcement learning as of my last training data in January 2022, it's possible that it refers to a custom or recently developed package. It might also be a typographical error for "Gym," a toolkit by OpenAI for developing and comparing reinforcement learning algorithms.

PYFLYT: PYFLYT seems to be a simulation or utility package used in the context of the provided code. It's used for UAV or drone simulations based on the context given, but its exact details are not within my training data as of January 2022.

## Future Work:

Plans to iterate on the environment involve:

- **Additional Quadcopters**: Introducing more quadcopters, simulating a more complex swarm behavior.
- **Obstacles**: Adding physical barriers to test and refine navigation and decision-making skills.
- **Weather Conditions**: Introducing weather elements like wind and rain to further test quadcopter performance.

For a deeper dive into the Python source code, you can explore the files within the `components` folder.
