"""Spawn a single drone on x=0, y=0, z=1, with 0 rpy."""
import numpy as np
import time

# from PyFlyt.core import aviary_loyalwingmen
# from PyFlyt.core.aviary_loyalwingmen import Aviary


from loyalwingmen.modules.environments_pyflyt.level1.components.pyflyt_level1_simulation import (
    AviarySimulation as Aviary,
)
from loyalwingmen.modules.environments_pyflyt.level1.components.quadcopter_manager import (
    QuadcopterManager,
    QuadcopterBlueprint,
)

# environment setup
print("Creating environment")
env = Aviary(render=True)
print("Environment created")

quadcopter_manager = QuadcopterManager(env)
persuer_list = quadcopter_manager.spawn_persuer(1, np.array([[0, 0, 1]]), "persuer")
persuer = persuer_list[0]

# simulate for 1000 steps (1000/120 ~= 8 seconds)
i = 0
while True:
    if i > 5:
        break

    i += 1
    quadcopter_manager.replace_all_to_starting_position()

    for step in range(1000):
        print("Step", step)
        env.step()
        quadcopter = quadcopter_manager.get_persuers()[0]
        print(f"inertial data: {quadcopter.inertial_data} ")
        if step < 100:
            print("Ascending")
            persuer.drive(np.array([0, 0, 1, 0.5]))

        if step >= 100 and step < 100:
            print("Lefting")
            persuer.drive(np.array([0, 1, 0, 0.5]))

        if step >= 200 and step < 300:
            print("Forwarding")
            persuer.drive(np.array([1, 0, 0, 0.5]))

        if step >= 300 and step < 400:
            print("backwarding")
            persuer.drive(np.array([-1, 0, 0, 0.5]))

        if step >= 400 and step < 500:
            print("Righting")
            persuer.drive(np.array([0, -1, 0, 0.5]))

        if step >= 500 and step < 600:
            print("Descending")
            persuer.drive(np.array([0, 0, -1, 0.5]))

        if step >= 600 and step < 700:
            print("Stopping")
            persuer.drive(np.array([0, 0, 0, 0.5]))

        if step >= 700:
            print("Keep Stopping")
            persuer.drive(np.array([0, 0, 0, 0.5]))
