Class: Gun
The Gun class represents a weapon system mounted on a quadcopter, capable of firing at targets within its range. The gun operates based on ammunition availability, cooldown periods, and firing probabilities.

Attributes:
munition: (int) The number of bullets available for firing.
cooldown_time: (float) The time required (in simulation steps or seconds) between successive shots.
last_fired_time: (float) The last time (in simulation steps or seconds) the gun was fired.
fire_probability: (float) The probability of successfully hitting a target when fired, ranging from 0 to 1.
target_range: (float) The maximum distance at which the gun can engage targets.
current_step: (int) The current simulation step, updated regularly from the simulation.

The solution presented is to use the Observer Pattern as a way to receive the current step to update the gun internal state and to hit the target.
In this manner, the class MessageHub works as a mediator between the Gun, the Simulation and others quadcopters. The Simulation publish the current step to topic "SimulationStep" and the Gun subscribe to this topic to receive the current step and timestep to make itself available if step is greater than cooldown.

So: Step, TimeStep
Simulation -> MessageHub -> Gun

The acquisition of the target happens inside the Lidar Class, with a callback function called when an object were detected inside a certain threshold. This callback is sourced by the parent quadcopter, that receives the target information and triggers the Gun to shoot against it. Although, as the threshold is a measure of inside the Gun's range, is not need to check if the target is inside the range, but it is done again, in the shoot function, to ensure this constraint.

the Shoot function will calculate the probability of hit and return True or False if the target were hit. The Gun will also update the last_fired_time and decrement the munition.

So:
Lidar -> Quadcopter -> Gun
