import mujoco_py
import os
import numpy as np
from scipy.spatial.transform import Rotation

def controller(model, data):
    data.ctrl[0] = 10
    data.ctrl[1] = 10

xml_path = 'model/differential.xml'

model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

def quat2euler(quat_mujoco):
    quat_scipy = np.array([quat_mujoco[3], quat_mujoco[0], quat_mujoco[1], quat_mujoco[2]])
    r = Rotation.from_quat(quat_scipy)
    euler = r.as_euler('xyz', degrees=True)
    return euler

while True:
    sim.data.ctrl[0] = 1
    sim.data.ctrl[1] = 1
    # print(sim.data.qpos[0])
    # print(sim.data.qpos[1])
    # print(sim.data.qpos[2])
    sim.step()
    viewer.render()

    quat = np.array([sim.data.qpos[3], sim.data.qpos[4], sim.data.qpos[5], sim.data.qpos[6]])
    euler = quat2euler(quat)
    # print('yaw = ', euler[2])
    print(sim.data.site_xpos[0])

    if os.getenv('TESTING') is not None:
        break