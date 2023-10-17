from dataclasses import dataclass
import numpy as np


@dataclass(order=True)
class QuadcopterSpecs:
    M: float
    L: float
    THRUST2WEIGHT_RATIO: float
    J: np.ndarray
    J_INV: np.ndarray
    KF: float
    KM: float
    COLLISION_H: float
    COLLISION_R: float
    COLLISION_Z_OFFSET: float
    MAX_SPEED_KMH: float
    GND_EFF_COEFF: float
    PROP_RADIUS: float
    DRAG_COEFF: np.ndarray
    DW_COEFF_1: float
    DW_COEFF_2: float
    DW_COEFF_3: float
    WEIGHT: float


''''
QuadcopterSpecs:
M: This seems to represent the mass of the quadcopter.
L: Possibly the distance between the center of the quadcopter and a motor, or the arm length.
THRUST2WEIGHT_RATIO: The ratio of the maximum thrust the quadcopter can produce to its weight.
J: The inertia tensor of the quadcopter, which describes its resistance to changes in angular speed.
J_INV: The inverse of the inertia tensor.
KF: Thrust constant; could be the proportionality constant between the square of motor speed (RPM) and the thrust they produce.
KM: Torque constant; could be the proportionality constant between the square of motor speed (RPM) and the torque they produce.
COLLISION_H: Height for collision detection.
COLLISION_R: Radius for collision detection.
COLLISION_Z_OFFSET: The offset in the Z direction for collision detection.
MAX_SPEED_KMH: The maximum speed of the quadcopter in kilometers per hour.
GND_EFF_COEFF: Ground effect coefficient, possibly representing how close to the ground the quadcopter experiences additional lift due to ground effect.
PROP_RADIUS: Radius of the propellers.
DRAG_COEFF: Coefficients for aerodynamic drag. This could be a 3D vector (for drag in the X, Y, and Z directions).
DW_COEFF_1, DW_COEFF_2, DW_COEFF_3: Coefficients related to downwash, or the downward thrust of air caused by the propellers. Could be used in more complex aerodynamic models.
WEIGHT: The weight (or gravitational force) acting on the quadcopter.

'''