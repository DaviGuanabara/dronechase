import numpy as np
import pybullet as p
import platform
import os
from pathlib import Path

import xml.etree.ElementTree as etxml
from typing import Union, Tuple

from .components.base.quadcopter import (
    Quadcopter,
    FlightStateDataType,
    QuadcopterSpecs,
    OperationalConstraints,
    OperationalConstraintsCalculator,
    QuadcopterType,
    DroneModel,
    CommandType,
)


from ..environments.helpers.environment_parameters import EnvironmentParameters
from .loyalwingman import LoyalWingman
from .loiteringmunition import LoiteringMunition, LoiteringMunitionBehavior
from enum import Enum, auto


class DroneURDFHandler:
    """Handler for drone's URDF files and related operations."""

    def __init__(
        self, drone_model: DroneModel, environment_parameters: EnvironmentParameters
    ):
        self.environment_parameters = environment_parameters
        self.drone_model = drone_model

    def load_model(self, initial_position, initial_quaternion):
        """Load the drone model and return its ID and parameters."""

        drone_model = self.drone_model
        environment_parameters = self.environment_parameters
        client_id = environment_parameters.client_id
        urdf_file_path = DroneURDFHandler._create_path(drone_model=drone_model)
        tree = etxml.parse(urdf_file_path)
        root = tree.getroot()

        quadcopter_id = DroneURDFHandler._load_to_pybullet(
            initial_position, initial_quaternion, urdf_file_path, client_id
        )
        quadcopter_specs = DroneURDFHandler._load_parameters(
            root, environment_parameters
        )

        return quadcopter_id, quadcopter_specs

    @staticmethod
    def _create_path(drone_model: DroneModel) -> str:
        """Generate the path for the given drone model's URDF file."""

        # base_path = Path(os.getcwd()).parent
        base_path = Path(__file__).resolve().parent.parent.parent
        urdf_name = f"{drone_model.value}.urdf"
        return str(base_path / "assets" / urdf_name)

    @staticmethod
    def _load_to_pybullet(position, attitude, urdf_file_path, client_id):
        """Load the drone model into pybullet and return its ID."""
        quarternion = p.getQuaternionFromEuler(attitude)
        return p.loadURDF(
            urdf_file_path,
            position,
            quarternion,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=client_id,
        )

    @staticmethod
    def _load_parameters(
        root, environment_parameters: EnvironmentParameters
    ) -> QuadcopterSpecs:
        """Loads parameters from an URDF file.
        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        """

        URDF_TREE = root  # self.root
        M = float(URDF_TREE[1][0][1].attrib["value"])
        L = float(URDF_TREE[0].attrib["arm"])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib["thrust2weight"])
        IXX = float(URDF_TREE[1][0][2].attrib["ixx"])
        IYY = float(URDF_TREE[1][0][2].attrib["iyy"])
        IZZ = float(URDF_TREE[1][0][2].attrib["izz"])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib["kf"])
        KM = float(URDF_TREE[0].attrib["km"])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib["length"])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib["radius"])
        COLLISION_SHAPE_OFFSETS = [
            float(s) for s in URDF_TREE[1][2][0].attrib["xyz"].split(" ")
        ]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib["max_speed_kmh"])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib["gnd_eff_coeff"])
        PROP_RADIUS = float(URDF_TREE[0].attrib["prop_radius"])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib["drag_coeff_xy"])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib["drag_coeff_z"])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib["dw_coeff_1"])
        DW_COEFF_2 = float(URDF_TREE[0].attrib["dw_coeff_2"])
        DW_COEFF_3 = float(URDF_TREE[0].attrib["dw_coeff_3"])

        WEIGHT = M * environment_parameters.G
        return QuadcopterSpecs(
            M=M,
            L=L,
            THRUST2WEIGHT_RATIO=THRUST2WEIGHT_RATIO,
            J=J,
            J_INV=J_INV,
            KF=KF,
            KM=KM,
            COLLISION_H=COLLISION_H,
            COLLISION_R=COLLISION_R,
            COLLISION_Z_OFFSET=COLLISION_Z_OFFSET,
            MAX_SPEED_KMH=MAX_SPEED_KMH,
            GND_EFF_COEFF=GND_EFF_COEFF,
            PROP_RADIUS=PROP_RADIUS,
            DRAG_COEFF=DRAG_COEFF,
            DW_COEFF_1=DW_COEFF_1,
            DW_COEFF_2=DW_COEFF_2,
            DW_COEFF_3=DW_COEFF_3,
            WEIGHT=WEIGHT,
        )


class QuadcopterFactory:
    def __init__(
        self,
        environment_parameters: EnvironmentParameters,
        drone_model: DroneModel = DroneModel.CF2X,
    ):
        self.environment_parameters = environment_parameters
        self.client_id: int = environment_parameters.client_id
        self.debug: bool = environment_parameters.debug

        self.drone_model = drone_model
        self.drone_urdf_handler = DroneURDFHandler(
            self.drone_model, self.environment_parameters
        )

        self.n_loyalwingmen = 0
        self.n_loiteringmunitions = 0

    def load_quad_attributes(
        self,
        initial_position: np.ndarray,
        initial_angular_position: np.ndarray,
        quadcopter_type: QuadcopterType,
        quadcopter_name: str,
        quadcopter_role: str = "general",
    ) -> Tuple[
        int,
        DroneModel,
        QuadcopterSpecs,
        OperationalConstraints,
        EnvironmentParameters,
        str,
    ]:
        id, parameters = self.drone_urdf_handler.load_model(
            initial_position, initial_angular_position
        )

        operational_constraints = OperationalConstraintsCalculator.compute(
            parameters, self.environment_parameters
        )

        environment_parameters = self.environment_parameters
        quadcopter_full_name = self._gen_quadcopter_name(
            quadcopter_type, quadcopter_name, quadcopter_role
        )

        return (
            id,
            self.drone_model,
            parameters,
            operational_constraints,
            environment_parameters,
            quadcopter_full_name,
        )

    def create_loyalwingman(
        self,
        position: np.ndarray,
        ang_position: np.ndarray,
        command_type: CommandType = CommandType.VELOCITY_DIRECT,
        quadcopter_name: str = "Drone",
        quadcopter_role: str = "general",
    ) -> LoyalWingman:
        attributes = self.load_quad_attributes(
            position,
            ang_position,
            QuadcopterType.LOYALWINGMAN,
            quadcopter_name,
            quadcopter_role,
        )

        return LoyalWingman(*attributes, command_type=command_type)

    def create_loiteringmunition(
        self,
        position: np.ndarray,
        ang_position: np.ndarray,
        command_type: CommandType = CommandType.VELOCITY_DIRECT,
        quadcopter_name: str = "Drone",
        quadcopter_role: str = "general",
    ) -> LoiteringMunition:
        attributes = self.load_quad_attributes(
            position,
            ang_position,
            QuadcopterType.LOITERINGMUNITION,
            quadcopter_name,
            quadcopter_role,
        )
        return LoiteringMunition(*attributes, command_type=command_type)

    def _gen_quadcopter_name(
        self,
        quadcopter_type: QuadcopterType,
        quadcopter_name: str = "Drone",
        role: str = "general",
    ):
        if quadcopter_type == QuadcopterType.LOYALWINGMAN:
            # return f"{quadcopter_type.name.lower()}_{self.n_loyalwingmen}_{quadcopter_name}"
            return f"{quadcopter_name}_{role}_{self.n_loyalwingmen}"

        if quadcopter_type == QuadcopterType.LOITERINGMUNITION:
            # return f"{quadcopter_type.name.lower()}_{self.n_loiteringmunitions}_{quadcopter_name}"
            return f"{quadcopter_name}_{role}_{self.n_loiteringmunitions}"

        return f"{quadcopter_type.name.lower()}_{quadcopter_name}_{role}"
