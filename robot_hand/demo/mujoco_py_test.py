import time

import mujoco_py as mp
import numpy as np
from robosuite.utils.transform_utils import mat2quat

# model = mp.load_model_from_path('../model/UR5gripper_2_finger.xml')
model = mp.load_model_from_path('../model/robot_hand/body_sim.xml')
# model = mp.load_model_from_path('../model/robot_hand/hand_sim.xml')
# model = mp.load_model_from_path('../model/quadrupeds/unitree_a1_torque.xml')
# model = mp.load_model_from_path('../model/mujoco_menagerie/kuka_iiwa_14/iiwa14.xml')
sim = mp.MjSim(model)
viewer = mp.MjViewer(sim)
start_time = time.time()

# joint_left = ["left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6"]
# joint_right = ["right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6"]
#
# joint_index = []
# for i in range(12):
#     if i < 6:
#         joint_index.append(sim.model.get_joint_qpos_addr("left_j" + str(i + 1)))
#     else:
#         joint_index.append(sim.model.get_joint_qpos_addr("right_j" + str(i - 5)))
# print(joint_index)



while time.time() - start_time < 5:
    print(sim.data.get_site_xpos('tip'))
    if time.time() - start_time < 3:
        sim.data.ctrl[0] = 1
        sim.data.ctrl[1] = 1
        sim.data.ctrl[2] = 1
        sim.data.ctrl[3] = 1
        sim.data.ctrl[4] = 1
        sim.data.ctrl[5] = 1
        sim.data.ctrl[6] = 1
    #         sim.data.ctrl[7] = 1
    #         sim.data.ctrl[8] = 1
    #         sim.data.ctrl[9] = 1
    #         sim.data.ctrl[10] = 1
    #         sim.data.ctrl[11] = 1
    #         sim.data.ctrl[12] = 1
    #         sim.data.ctrl[13] = 1
    #         sim.data.ctrl[14] = 1
    #         sim.data.ctrl[15] = 1
    else:
        sim.data.ctrl[0] = -1
        sim.data.ctrl[1] = -1
        sim.data.ctrl[2] = -1
        sim.data.ctrl[3] = -1
        sim.data.ctrl[4] = -1
        sim.data.ctrl[5] = -1
        sim.data.ctrl[6] = -1
    #         sim.data.ctrl[7] = -1
    #         sim.data.ctrl[8] = -1
    #         sim.data.ctrl[9] = -1
    #         sim.data.ctrl[10] = -1
    #         sim.data.ctrl[11] = -1
    #         sim.data.ctrl[12] = -1
    #         sim.data.ctrl[13] = -1
    #         sim.data.ctrl[14] = -1
    #         sim.data.ctrl[15] = -1
    sim.step()
    viewer.render()
