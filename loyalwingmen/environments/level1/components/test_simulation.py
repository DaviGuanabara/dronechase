"""Spawn a single drone on x=0, y=0, z=1, with 0 rpy."""
import numpy as np
import time

# from PyFlyt.core import aviary_loyalwingmen
# from PyFlyt.core.aviary_loyalwingmen import Aviary


from loyalwingmen.modules.environments_pyflyt.level1.components.pyflyt_level1_simulation import (
    AviarySimulation as Aviary,
)

# environment setup
print("Creating environment")
env = Aviary(render=True)
print("Environment created")


# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    print("Step", i)
    time.sleep(0.1)
    env.step()
