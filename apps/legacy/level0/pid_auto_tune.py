import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)


import pybullet as p
from loyalwingmen.modules.environments.level0.pid_auto_tune import PIDAutoTuner

pid_autotuner = PIDAutoTuner(10, 30, 240, True)
pid_autotuner._reset_simulation()
for _ in range(5):
    pid_autotuner.apply_step_input()
# p.connect(p.GUI)

# import pybullet as p
# import time
# import pybullet_data

# physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
# p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
# p.setGravity(0, 0, -10)
# planeId = p.loadURDF("plane.urdf")
# startPos = [0, 0, 1]
# startOrientation = p.getQuaternionFromEuler([0, 0, 0])
# boxId = p.loadURDF("r2d2.urdf", startPos, startOrientation)
# set the center of mass frame (loadURDF sets base link frame)
# startPos / Orn
# p.resetBasePositionAndOrientation(boxId, startPos, startOrientation)

# for _ in range(10000):
#    p.stepSimulation()
#    time.sleep(1.0 / 240.0)
#    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
#    print(cubePos, cubeOrn)
