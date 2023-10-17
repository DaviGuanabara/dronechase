import numpy as np
import pybullet as p

from .actuator_interface import ActuatorInterface
from ..controllers.DSLPIDControl import DSLPIDControl

from ..dataclasses.flight_state import FlightStateManager


from ....environments.helpers.environment_parameters import EnvironmentParameters
from ..dataclasses.quadcopter_specs import QuadcopterSpecs
from ..dataclasses.operational_constraints import OperationalConstraints

from ....utils.enums import DroneModel

from enum import Enum, auto


class CommandType(Enum):
    VELOCITY_TO_CONTROLLER = auto()
    VELOCITY_DIRECT = auto()
    RPM = auto()


class Propulsion(ActuatorInterface):
    def __init__(self, drone_id: int, drone_specs: QuadcopterSpecs, client_id: int):
        self.drone_specs = drone_specs
        self.client_id = client_id
        self.drone_id = drone_id

    def apply(self, action):
        assert False, "Not implemented"


class Motors(Propulsion):
    def __init__(self, drone_id: int, drone_specs: QuadcopterSpecs, client_id: int):
        super().__init__(drone_id, drone_specs, client_id)

        """
            This data were extracted from dls_pid_control.py.
        """
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.minimum_rpm = (self.MIN_PWM - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        self.maximum_rpm = (self.MAX_PWM - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

    def apply(
        self,
        rpm: np.ndarray,
    ):
        """Base PyBullet physics implementation.
        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position
        """

        KF = self.drone_specs.KF
        KM = self.drone_specs.KM

        # print(f"rpm: {rpm}")
        # rpm_float = rpm.astype(np.float64)
        forces = np.array(rpm**2) * KF
        torques = np.array(rpm**2) * KM
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]

        for i in range(4):
            p.applyExternalForce(
                self.drone_id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client_id,
            )

        p.applyExternalTorque(
            self.drone_id,
            4,
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self.client_id,
        )


class DirectVelocityApplier(Propulsion):
    def __init__(self, drone_id: int, drone_specs: QuadcopterSpecs, client_id: int):
        super().__init__(drone_id, drone_specs, client_id)

    def apply(
        self,
        velocity: np.ndarray,
        angularVelocity: np.ndarray = np.zeros(3),
    ):
        # A força é a plicada, mas não consegue compensar a gravidade não sei o motivo
        # assim, prefiro deixar comentado, sem gravidade, sem compensação.
        # p.applyExternalForce(
        #    self.drone_id,
        #    -1,
        #    forceObj=np.array([0, 0, self.drone_specs.WEIGHT]),
        #    posObj=[0, 0, 0],
        #    flags=p.LINK_FRAME,
        #    physicsClientId=self.client_id,
        # )

        # print(velocity)
        p.resetBaseVelocity(
            self.drone_id,
            linearVelocity=velocity,
            angularVelocity=angularVelocity,
            physicsClientId=self.client_id,
        )


class PropulsionSystem:
    def __init__(
        self,
        drone_id: int,
        drone_model: DroneModel,
        drone_specs: QuadcopterSpecs,
        environment_parameters: EnvironmentParameters,
        operational_constraints: OperationalConstraints,
        quadcopter_name: str = "",
        command_type: CommandType = CommandType.VELOCITY_TO_CONTROLLER,
    ):
        self.quadcopter_name = quadcopter_name
        self.environment_parameters = environment_parameters
        self.command_type = command_type
        self.operational_constraints = operational_constraints

        self.max_rpm = operational_constraints.max_rpm
        self.min_rpm = 0

        self.max_action_rpm = 1
        self.min_action_rpm = -1

        self.propeller = self._init_propeller(
            command_type,
            drone_id,
            drone_specs,
            drone_model,
            environment_parameters,
            operational_constraints,
        )

    def _compute_velocity_from_command(self, motion_command: np.ndarray) -> np.ndarray:
        """
        Compute the velocity vector from a given motion command.

        Parameters:
        - motion_command: The motion command where the first three elements represent direction,
        and the fourth element represents intensity or magnitude.

        Returns:
        - velocity: The computed velocity vector.
        """

        norm = np.linalg.norm(motion_command[:3])
        if norm == 0:
            return np.array([0, 0, 0])
        speed_limit = self.operational_constraints.speed_limit
        intensity = speed_limit * motion_command[3]

        return intensity * motion_command[:3] / norm

    def _init_propeller(
        self,
        command_type,
        drone_id,
        drone_specs: QuadcopterSpecs,
        drone_model: DroneModel,
        environment_parameters: EnvironmentParameters,
        operational_constraints: OperationalConstraints,
    ):
        if command_type == CommandType.VELOCITY_DIRECT:
            direct_velocity_applier = DirectVelocityApplier(
                drone_id=drone_id,
                drone_specs=drone_specs,
                client_id=environment_parameters.client_id,
            )

            # print("_init_propeller - Direct velocity applier")

            return lambda velocity_command, flight_state_manager: self._directly_apply_velocity(
                velocity_command, flight_state_manager, direct_velocity_applier
            )

        elif command_type == CommandType.VELOCITY_TO_CONTROLLER:
            controller = DSLPIDControl(
                drone_model,
                operational_constraints,
                environment_parameters,
                drone_specs,
            )  # default or custom controller
            motors = Motors(
                drone_id=drone_id,
                drone_specs=drone_specs,
                client_id=environment_parameters.client_id,
            )

            # print("_init_propeller - Velocity to controller")
            return lambda velocity_command, flight_state_manager: self._velocity_to_controller(
                velocity_command, flight_state_manager, controller, motors
            )

        elif command_type == CommandType.RPM:
            motors = Motors(
                drone_id=drone_id,
                drone_specs=drone_specs,
                client_id=environment_parameters.client_id,
            )

            # print("_init_propeller - RPM")
            return lambda rpm, flight_state_manager: self._rl_to_rpm(
                rpm, flight_state_manager, motors
            )

        else:
            raise ValueError(f"Invalid command type: {command_type}")

        # return lambda any_thing, flight_state_manager: print("Invalid command type")

    def _directly_apply_velocity(
        self,
        velocity_command,
        flight_state_manager,
        direct_velocity_applier: DirectVelocityApplier,
    ):
        velocity = self._compute_velocity_from_command(velocity_command)
        direct_velocity_applier.apply(velocity)

    def _rl_to_rpm(
        self, action_rpm, flight_state_manager: FlightStateManager, motors: Motors
    ):
        # ((action_rpm - self.min_action_rpm) / (self.max_action_rpm - self.min_action_rpm)) = rpm - self.min_rpm / (self.max_rpm - self.min_rpm)
        # rpm = ((action_rpm - self.min_action_rpm) / (self.max_action_rpm - self.min_action_rpm)) * (self.max_rpm - self.min_rpm) + self.min_rpm
        # rpm = action_rpm * self.max_rpm
        action_rpm_normalized = (action_rpm - self.min_action_rpm) / (
            self.max_action_rpm - self.min_action_rpm
        )
        rpm = action_rpm_normalized * (self.max_rpm - self.min_rpm) + self.min_rpm
        motors.apply(rpm)

    def _velocity_to_controller(
        self, velocity_command, flight_state_manager, controller, motors: Motors
    ):
        velocity = self._compute_velocity_from_command(velocity_command)

        rpms = self._apply_controller(velocity, flight_state_manager, controller)
        motors.apply(rpms)

    def _apply_controller(
        self,
        target_velocity: np.ndarray,
        flight_state_manager: FlightStateManager,
        controller: DSLPIDControl,
    ) -> np.ndarray:
        inertial_data = flight_state_manager.get_inertial_data()

        yaw = inertial_data.get("attitude", np.zeros(3))[2]
        target_rpy = np.array([0, 0, yaw])

        # aggregate_physics_steps = self.environment_parameters.aggregate_physics_steps
        # timestep = self.environment_parameters.timestep
        # control_timestep = aggregate_physics_steps * timestep
        control_timestep = self.environment_parameters.timestep

        quaternions = p.getQuaternionFromEuler(
            inertial_data.get("attitude", np.array([0, 0, 0, 1]))
        )
        rpm, _, _ = controller.computeControl(
            control_timestep,
            inertial_data.get("position", np.zeros(3)),
            quaternions,
            inertial_data.get("velocity", np.zeros(3)),
            inertial_data.get("angular_rate", np.zeros(3)),
            target_pos=inertial_data.get("position", np.zeros(3)),
            target_rpy=target_rpy,  # keep current yaw,
            target_vel=target_velocity,
        )

        # print(
        #    f"codigo 2 \n /begin: \n speedlimit: {0.25} \n control_timestep {control_timestep} \nvelocity: {target_velocity} \n rpm: {rpm} \n /end"
        # )

        # print(rpm)
        return rpm

    def propel(self, velocity: np.ndarray, flight_state_manager: FlightStateManager):
        self.propeller(velocity, flight_state_manager)
