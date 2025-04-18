Level 3 Environment
When I started developing this project, I decided to write down each level's specifications, defining and following the definition. 


As the project grew, I realized it was more an adventure, with exploration and discovery, than a step-by-step plan.

I had a different objective in mind when I started level 3. Although it is more complex than I first imagined, it does not offer the conditions for the agent to develop a behavior that I desired. 

Level 3 shows 1 Loyal Wingman (pursuer) and n (default is 2) Loitering Munitions (invaders). The invader has to explode near (0.2 meters of range) the pursuer, and the LW aims to shoot all the LMs.

The shoot accuracy is 90%, and its range is 1 meter.

The behavior that emerged was a "go straight," hoping for the LM to get close enough to shoot but far enough not to die by an explosion. Most of the time, "go straight" could kill invaders and end up alive. We have a kind of complexity here, as the pursuer has to have enough distance and velocity to avoid hitting while keeping the invaders in the shooting range.

The behavior that I desired was keeping near the origin, or at least not risking going outside the dome. It may be corrected by reducing the dome size (which may require a reward rescale of its benefits and penalties) or not. I decided to create a next level in which there are new constraints and challenges.

The TaskProgress class idea was to implement Curriculum Learning. But I will take a different approach. All this structure will be helpful soon.

The Observer Design Pattern was very worthwhile: to use LIDAR and to broadcast the current agent step, helping synchronize components like GUN class and its cooldown.

Level 3 brought the idea of ARM and DISARM quadcopters. All the quadcopters are managed by the class Quadcopter_Manager. The armed quadcopters are inserted in the simulation list, which executes its action. The disarmed quadcopters do not exist in the simulation list; they don't have mass and cannot be collided. Their dark changes its core, and they do not move until it is where it is again. The Arm and Disarm approach improved a lot the simulation time cost (from 70 it/s to 400 it/s)



ChatGPT Generated Documentation:


Class: L3AviarySimulation
The L3AviarySimulation class serves as the core of the PyFlyt system for handling Unmanned Aerial Vehicles (UAVs) in the PyBullet simulation environment. It provides functionalities for physics stepping, setpoint handling, collision tracking, and various other features essential for controlling drones or defining tasks within a simulated environment.

Inheritance
The class inherits from bullet_client.BulletClient, which is part of the PyBullet library and provides an interface to the physics simulation.

Initialization
python
Copy code
def __init__(self, render: bool = False, physics_hz: int = 240, world_scale: float = 1.0, seed: None | int = None)
Initializes the simulation environment.

Parameters:
render (bool): Specifies whether to render the simulation visually.
physics_hz (int): The frequency (in Hz) of the physics loop. It's not recommended to change this value from the default.
world_scale (float): Determines the size of the simulation floor.
seed (None | int): Optional seed for the simulation's random number generator.
Methods
reset
Resets the simulation environment. This includes setting the gravity, camera position, and initializing the random number generator.

step
Advances the environment by one step. Each step corresponds to one control loop step and handles physics updates as well as control loop rates for each drone.

add_active_drone
Adds a drone to the active drones list within the simulation.

remove_active_drone
Removes a drone from the active drones list.

has_drones
Checks if there are any drones currently active in the simulation.

Attributes
active_drones: A dictionary holding references to active drones in the simulation, keyed by their unique IDs.
Evaluation Function
The script includes an on_avaluation_step function for testing and evaluating the simulation steps. It creates an instance of L3AviarySimulation, and then iteratively steps through the simulation for a specified number of iterations.

Main Function
The main function is set up to profile the performance of the on_avaluation_step function using Python's cProfile module. It generates performance statistics which can be used to optimize the simulation.

Class: PyflytL3Environment
PyflytL3Environment is a custom environment class designed for simulating a scenario with one Loyal Wingman and one Loitering Munition. This environment is compatible with Stable Baselines 3 (version 2.0.0 or higher) and is inspired by the Gym Pybullet Drones project.

Initialization
python
Copy code
def __init__(self, dome_radius: float = 8, rl_frequency: int = 15, GUI: bool = False)
Initializes the environment with the specified dome radius, reinforcement learning frequency, and graphical user interface settings.

Parameters:
dome_radius (float): Radius of the simulation dome.
rl_frequency (int): Frequency at which the reinforcement learning algorithm operates.
GUI (bool): Flag to enable or disable the graphical user interface.
Methods
_subscriber_notifications
Handles notifications from the simulation.

init_constants
Initializes constants used in the environment.

init_globals
Initializes global variables.

init_components
Initializes components of the simulation.

frequency_adjustments
Adjusts the frequency of the simulation based on the reinforcement learning frequency.

manage_debug_text
Manages debug text in the simulation.

close
Terminates the environment.

reset
Resets the environment for a new episode.

step
Advances the environment by one step based on the provided action.

advance_step
Advances the simulation steps.

compute_info
Computes additional information about the current state of the environment.

_action_space
Defines the action space of the environment.

compute_observation
Computes the observation of the current state of the environment.

_observation_space
Defines the observation space of the environment.

observation_shape
Determines the shape of various components in the observation space.

process_inertial_state
Processes the inertial state of a quadcopter.

convert_dict_to_array
Converts a dictionary to a numpy array.

get_keymap
Provides a keymap for manual control.

Attributes
dome_radius: Radius of the dome in the simulation.
debug_on: Flag indicating whether debugging is enabled.
show_name_on: Flag for displaying names in the simulation.
max_step_calls: Maximum number of steps per episode.
last_action: Last action taken by the agent.
step_counter: Counter for the number of steps taken in the current episode.
message_hub: A hub for handling messages and notifications within the simulation.


