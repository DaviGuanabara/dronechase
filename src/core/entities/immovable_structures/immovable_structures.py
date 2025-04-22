from __future__ import annotations

"""TODO: 

 - [x] This class has to have a function that broadcast its location and dimensions like quadcopter does.
 - [x] Has to receive current step
 - [x] Variable saying the kind of entity it is.
 - [x] Variable to say its life.
 - [x] Variable to say its color.
 
"""


from pybullet_utils.bullet_client import BulletClient # type: ignore
import numpy as np


from ..entity_type import EntityType
from ...notification_system.message_hub import MessageHub
from typing import Dict, Optional


class ImmovableStructures:
    def __init__(
        self,
        simulation: BulletClient,
        id: int,
        position: list,
        type: EntityType,
        dimensions: Optional[list] = None,
    ):
        self.id = id
        self.position = position
        self.simulation = simulation
        self.type = type
        self.dimensions = dimensions

        self.init_physical_properties()
        self.setup_message_hub()

    def init_physical_properties(self):
        # print("getting dynamics info")
        self.simulation.changeDynamics(self.id, -1, mass=0)
        self.dynamics_info = self.simulation.getDynamicsInfo(self.id, -1)

        self.shape_data = self.simulation.getVisualShapeData(self.id)[0]

        if self.dimensions is None:
            self.dimensions = self.shape_data[3]

        self.num_joints = self.simulation.getNumJoints(self.id)

        self.joint_indices = list(range(-1, self.num_joints))
        self.mass = self.dynamics_info[0]
        # self._armed = True

    # ===========================================================================
    #       Communitation with the environment
    # ===========================================================================

    def setup_message_hub(self):
        self.messageHub = MessageHub()

        self.messageHub.subscribe(
            topic="SimulationStep", subscriber=self._subscriber_simulation_step
        )

    def _subscriber_simulation_step(self, message: Dict, publisher_id: int):
        "The simulation step has been updated."
        "So, it should broadcast its location and dimensions."
        self.current_step = message.get("step", 0)
        self.timestep = message.get("timestep", 0)
        self._publish_inertial_data()

    def _publish_inertial_data(self):
        """
        Publish the current flight state data to the message hub.
        """
        data = self.gen_inertial_data()
        data["dimensions"] = self.dimensions
        data["current_step"] = self.current_step

        message = {**data, "publisher_type": self.type}
        # print(message)
        # print("Publishing intertial data")
        self.messageHub.publish(
            topic="inertial_data", message=message, publisher_id=self.id
        )

    def gen_inertial_data(self):
        """
        Retrieve data related to the inertial measurement unit (IMU).

         - Consider object static
        """

        quaternion = self.simulation.getQuaternionFromEuler([0, 0, 0])
        return {
            "position": self.position,
            "velocity": np.zeros(3, dtype=np.float32),
            "attitude": np.zeros(3, dtype=np.float32),
            "quaternion": quaternion,
            "angular_rate": np.zeros(3, dtype=np.float32),
        }

    # ===========================================================================
    #      Create Kinds of objects
    # ===========================================================================

    @staticmethod
    def spawn_building(simulation: BulletClient, position: list) -> ImmovableStructures:
        body_id = simulation.loadURDF(
            "cube_small.urdf",
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
        )

        return ImmovableStructures(
            simulation, body_id, position, EntityType.PROTECTED_BUILDING
        )


    @staticmethod
    def spawn_ground(
        simulation: BulletClient, position: list, dome_radius: float
    ) -> ImmovableStructures:
        body_id = simulation.loadURDF(
            "plane.urdf",
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
        )

        dimensions = ImmovableStructures.gen_ground_dimensions(dome_radius)
        return ImmovableStructures(
            simulation, body_id, position, EntityType.GROUND, dimensions
        )

    @staticmethod
    def gen_ground_dimensions(dome_radius: float) -> list:
        return [dome_radius * 2, dome_radius * 2, 0.01]

    # ===========================================================================
    #      Life Management
    # ===========================================================================

    def setup_life(self):
        self.max_life = 1
        self.life = self.max_life

    def hit(self):
        self.life -= 1

    def is_alive(self):
        return self.life > 0


    def update_color(self):
        p = self.simulation

        if self.life / self.max_life > 0.9:
            color = [0, 1, 0, 1]  # green

        elif self.life / self.max_life > 0.5:
            color = [1, 1, 0, 1]  # yellow

        else:
            color = [1, 0, 0, 1]  # red

        p.changeVisualShape(self.id, -1, rgbaColor=color)
