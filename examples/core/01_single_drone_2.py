"""Spawn a single drone on x=0, y=0, z=1, with 0 rpy."""
import numpy as np

# from PyFlyt.core import aviary_loyalwingmen
# from PyFlyt.core.aviary_loyalwingmen import Aviary

from loyalwingmen.modules.environments_pyflyt.level1.components.pyflyt_level1_simulation import (
    AviarySimulation as Aviary,
)


# environment setup
print("Creating environment")
env = Aviary(render=True)
print("Environment created")
env.create_invader(1, np.array([[0.0, 0.0, 1.0]]))
env.create_persuer(1, np.array([[1.0, 1.0, 1.0]]))
# env = AviarySimulation(render=True)

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    print("Step", i)
    env.step()
