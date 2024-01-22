import time

import mujoco
import numpy as np
from collections.abc import Iterable
from robot_hand.MujocoXMLModel import MujocoXML
from robot_hand.single_arm import SingleArm
from robot_hand.controller_factory import load_controller_config
from robot_hand.opencv_renderer import OpenCVRenderer

from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim


model = MujocoXML('test3.xml')
# model = MujocoXML('../model/robot_hand/humanoid_test.xml')
# model = MujocoXML('test.xml')
xml = model.get_xml()
sim = MjSim.from_xml_string(xml)
sim.forward()
viewer = OpenCVRenderer(sim)
viewer.set_camera(0)
if sim._render_context_offscreen is None:
    render_context = MjRenderContextOffscreen(sim, device_id=-1)
sim._render_context_offscreen.vopt.geomgroup[0] = 0
sim._render_context_offscreen.vopt.geomgroup[1] = 1

control_freq = 20
controller_name = 'OSC_POSE'
controller_config = load_controller_config(default_controller=controller_name)
robot = SingleArm('UR5e', 0, controller_config)
# robot = SingleArm('Humanoid', 0, controller_config)
robot.load_model()
robot.reset_sim(sim)
robot.setup_references()
robot.reset()
# print(robot.action_dim)

def step(sim, action):
    policy_step = True
    control_timestep = 0.05
    model_timestep = 0.002
    for i in range(int(control_timestep / model_timestep)):  # 0.05 / 0.002 = 25
        sim.forward()
        robot.control(action, policy_step)
        sim.step()
        policy_step = False

start_time = time.time()
while time.time() - start_time < 5:
    action = np.zeros(6)
    # print(robot._joint_positions)
    if time.time() - start_time < 2:
        action[1] = 1.0
    # step(sim, action)
    sim.forward()
    robot.control(action, policy_step=True)
    sim.step()
    viewer.render()
# import mujoco.viewer
# m = mujoco.MjModel.from_xml_path('../model/robot_hand/body_new.xml')
# d = mujoco.MjData(m)
# with mujoco.viewer.launch_passive(m, d) as viewer:
# # Close the viewer automatically after 30 wall-seconds.
#     start = time.time()
#     while viewer.is_running() and time.time() - start < 1000:
#         step_start = time.time()
#         # print(m.data.get_site_xpos('tip'))
#         # mj_step can be replaced with code that also evaluates
#         # a policy and applies a control signal before stepping the physics.
#         mujoco.mj_step(m, d)
#
#         # Example modification of a viewer option: toggle contact points every two seconds.
#         with viewer.lock():
#             viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
#
#         # Pick up changes to the physics state, apply perturbations, update options from GUI.
#         viewer.sync()
#
#         # Rudimentary time keeping, will drift relative to wall clock.
#         time_until_next_step = m.opt.timestep - (time.time() - step_start)
#         if time_until_next_step > 0:
#             time.sleep(time_until_next_step)



