import numpy as np
from dataclasses import dataclass


@dataclass(order=True)
class OperationalConstraints:
    speed_limit: float = 0
    angular_speed_limit: float = 0
    acceleration_limit: float = 0
    angular_acceleration_limit: float = 0

    weight: float = 0
    max_rpm: float = 0

    max_xy_thrust: float = 0
    max_z_thrust: float = 0

    max_thrust: float = 0
    max_z_torque: float = 0

    hover_rpm: float = 0
    gnd_eff_h_clip: float = 0
    max_xy_torque: float = 0


"""

OperationalConstraints:
speed_limit: The maximum linear speed the quadcopter is allowed to have.
angular_speed_limit: The maximum angular speed (rotational speed) the quadcopter is allowed to have.
acceleration_limit: The maximum linear acceleration the quadcopter is allowed to have.
angular_acceleration_limit: The maximum angular acceleration the quadcopter is allowed to have.
weight: The weight of the quadcopter, same as the WEIGHT in QuadcopterSpecs (seems redundant).
max_rpm: The maximum revolutions per minute of the quadcopter's motors.
max_thrust: The maximum thrust the quadcopter can produce.
max_z_torque: The maximum torque the quadcopter can produce about the Z-axis.
hover_rpm: The revolutions per minute of the motors when the quadcopter is hovering (maintaining a steady altitude).
gnd_eff_h_clip: Related to the ground effect. Perhaps the height below which the quadcopter starts experiencing a significant ground effect.
max_xy_torque: The maximum torque the quadcopter can produce about the X or Y axes.

"""


class OperationalConstraintsCalculator:
    @staticmethod
    def compute_max_torque(max_rpm, L, KF, KM):
        max_z_torque = 2 * KM * max_rpm**2
        max_xy_torque = (2 * L * KF * max_rpm**2) / np.sqrt(2)

        return max_xy_torque, max_z_torque

    @staticmethod
    def compute_thrust(max_rpm, KF, M):
        max_thrust = 4 * KF * max_rpm**2
        acceleration_limit = max_thrust / M
        return max_thrust, acceleration_limit

    @staticmethod
    def compute_rpm(weight, KF, THRUST2WEIGHT_RATIO):
        max_rpm = np.sqrt((THRUST2WEIGHT_RATIO * weight) / (4 * KF))
        hover_rpm = np.sqrt(weight / (4 * KF))

        return max_rpm, hover_rpm

    @staticmethod
    def compute_speed_limit(parameters):
        """
        QuadcopterSpecs
        """
        KMH_TO_MS = 1000 / 3600
        VELOCITY_LIMITER = 1

        return VELOCITY_LIMITER * parameters.MAX_SPEED_KMH * KMH_TO_MS

    @staticmethod
    def compute_gnd_eff_h_clip(max_rpm, KF, GND_EFF_COEFF, max_thrust, PROP_RADIUS):
        return (
            0.25
            * PROP_RADIUS
            * np.sqrt((15 * max_rpm**2 * KF * GND_EFF_COEFF) / max_thrust)
        )

    @staticmethod
    def compute_moment_of_inertia(M, L):
        # I know that this may be not the correct way to compute the accelerations and velocity limits, but it is the best I can do for now.
        I_x = I_y = (1 / 12) * M * L**2
        I_z = (1 / 6) * M * L**2

        return I_x, I_y, I_z

    @staticmethod
    def compute_angular_acceleration_limit(
        max_xy_torque, max_z_torque, I_x, I_z
    ) -> float:
        alpha_x = alpha_y = max_xy_torque / I_x
        alpha_z = max_z_torque / I_z
        return max(alpha_x, alpha_z)

    @staticmethod
    def compute_angular_speed_limit(angular_acceleration_limit, timestep) -> float:
        return angular_acceleration_limit * timestep

    @staticmethod
    def compute(parameters, environment_parameters) -> OperationalConstraints:
        # Your operational constraints logic here
        gravity_acceleration = environment_parameters.G
        timestep = environment_parameters.timestep

        KMH_TO_MS = 1000 / 3600
        VELOCITY_LIMITER = 1

        L = parameters.L
        M = parameters.M
        KF = parameters.KF
        KM = parameters.KM
        PROP_RADIUS = parameters.PROP_RADIUS
        GND_EFF_COEFF = parameters.GND_EFF_COEFF
        THRUST2WEIGHT_RATIO = parameters.THRUST2WEIGHT_RATIO

        WEIGHT = gravity_acceleration * M

        max_rpm, hover_rpm = OperationalConstraintsCalculator.compute_rpm(
            WEIGHT, KF, THRUST2WEIGHT_RATIO
        )

        speed_limit = OperationalConstraintsCalculator.compute_speed_limit(parameters)
        (
            max_thrust,
            acceleration_limit,
        ) = OperationalConstraintsCalculator.compute_thrust(max_rpm, KF, M)
        (
            max_xy_torque,
            max_z_torque,
        ) = OperationalConstraintsCalculator.compute_max_torque(max_rpm, L, KF, KM)

        gnd_eff_h_clip = OperationalConstraintsCalculator.compute_gnd_eff_h_clip(
            max_rpm, KF, GND_EFF_COEFF, max_thrust, PROP_RADIUS
        )

        I_x, I_y, I_z = OperationalConstraintsCalculator.compute_moment_of_inertia(M, L)
        angular_acceleration_limit = (
            OperationalConstraintsCalculator.compute_angular_acceleration_limit(
                max_xy_torque, max_z_torque, I_x, I_z
            )
        )
        angular_speed_limit = (
            OperationalConstraintsCalculator.compute_angular_speed_limit(
                angular_acceleration_limit, timestep
            )
        )

        # Saving constraints
        operational_constraints = OperationalConstraints()
        operational_constraints.weight = WEIGHT
        operational_constraints.max_rpm = max_rpm
        operational_constraints.max_thrust = max_thrust

        operational_constraints.max_xy_thrust = max_thrust
        operational_constraints.max_z_thrust = max_thrust - WEIGHT

        operational_constraints.max_z_torque = max_z_torque
        operational_constraints.hover_rpm = hover_rpm

        operational_constraints.speed_limit = speed_limit
        operational_constraints.acceleration_limit = acceleration_limit

        operational_constraints.angular_speed_limit = angular_speed_limit
        operational_constraints.angular_acceleration_limit = angular_acceleration_limit

        operational_constraints.gnd_eff_h_clip = gnd_eff_h_clip
        operational_constraints.max_xy_torque = max_xy_torque

        return operational_constraints
