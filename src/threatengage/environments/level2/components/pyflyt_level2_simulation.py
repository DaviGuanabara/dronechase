"""The Aviary class, the core of how PyFlyt handles UAVs in the PyBullet simulation environment."""
from __future__ import annotations


import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client


from core.entities.quadcopters.quadcopter import Quadcopter

class L2AviarySimulation(bullet_client.BulletClient):
    """Aviary class, the core of how PyFlyt handles UAVs in the PyBullet simulation environment.

    The `aviary` is a handler for physics stepping, setpoint handling, collisions tracking, and much more.
    It provides a common endpoint from where users may control drones or define tasks.

    Args:
        render (bool): a boolean whether to render the simulation.
        physics_hz (int): physics looprate (not recommended to be changed).
        world_scale (float): how big to spawn the floor.
        seed (None | int): optional int for seeding the simulation RNG.
    """

    def __init__(
        self,
        render: bool = False,
        physics_hz: int = 240,
        world_scale: float = 1.0,
        seed: None | int = None,
    ):
        super().__init__(p.GUI if render else p.DIRECT)

        # The ctrl_hz is setted inside the quadcopter and i have no
        # intention to change it.

        print("AviarySimulation")
        self.ctrl_hz = 120
        self.physics_hz = physics_hz
        self.physics_period = 1.0 / physics_hz

        # constants for tracking how many times to step depending on control hz
        print("self.physics_hz", self.physics_hz)
        self.updates_per_step = self.physics_hz // self.ctrl_hz
        self.update_period = 1.0 / self.updates_per_step

        # set the world scale and directories
        print("self.updates_per_step", self.updates_per_step)
        self.wind_field = None  # no wind field
        self.world_scale = world_scale
        self.setAdditionalSearchPath(pybullet_data.getDataPath())

        # render
        print("self.world_scale", self.world_scale)
        self.render = render

        print("self.render", self.render)
        self.drones: list[Quadcopter] = []
        self.reset(seed)

    def reset(self, seed: None | int = None):
        """Resets the simulation.

        Args:
            seed (None | int): seed
        """
        self.resetSimulation()
        self.setGravity(0, 0, -9.81)
        self.physics_steps: int = 0

        # reset the camera position to a sane place
        self.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 1],
        )

        # define new RNG
        self.np_random = np.random.RandomState(seed=seed)

    def has_drones(self):
        """Returns whether the simulation has any drones."""
        return len(self.drones) > 0

    def step(self):
        """Steps the environment, this automatically handles physics and control looprates, one step is equivalent to one control loop step."""

        for _ in range(self.updates_per_step):
            if self.has_drones():
                for drone in self.drones:
                    # drone.update_state()
                    # update imu is a more meaningful name
                    # describing that Lidar or other sensors are not updated.
                    drone.update_imu()
                    drone.update_control()
                    drone.update_physics()

            else:
                print("No drones in the simulation.")

            self.stepSimulation()

            self.physics_steps += 1


def on_avaluation_step():
    print("Creating environment")
    simulation = L2AviarySimulation(render=True)
    print("Environment created")

    for i in range(1000):
        print("Step", i)
        simulation.step()


def main():
    import cProfile
    import pstats

    cProfile.run("on_avaluation_step()", "result_with_lidar.prof")
    stats = pstats.Stats("result.prof")
    stats.sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    # https://stackoverflow.com/questions/29690091/python2-7-exception-the-freeze-support-line-can-be-omitted-if-the-program
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
