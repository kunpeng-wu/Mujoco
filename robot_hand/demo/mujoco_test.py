import time

import mujoco
import mujoco.viewer
import numpy as np
import cv2
from robot_hand.controller_factory import controller_factory, load_controller_config


# m = mujoco.MjModel.from_xml_path('test.xml')
m = mujoco.MjModel.from_xml_path('robot_two_arm_test.xml')

height = 480
width = 640
depth = False
d = mujoco.MjData(m)

# controller_config = load_controller_config(custom_fpath='osc_pose.json')
# controller_config["robot_name"] = 'UR5e'
# controller_config["sim"] = sim
# controller_config["eef_name"] = gripper.important_sites["grip_site"]
# controller_config["eef_rot_offset"] = eef_rot_offset
# controller_config["joint_indexes"] = {
#     "joints": joint_indexes,
#     "qpos": _ref_joint_pos_indexes,
#     "qvel": _ref_joint_vel_indexes,
# }
# controller_config["actuator_range"] = torque_limits
# controller_config["policy_freq"] = control_freq
# controller_config["ndim"] = len(robot_joints)
# Instantiate the relevant manipulator
# manipulator = controller_factory(controller_config["type"], controller_config)


init_qpos = np.zeros(6)
_ref_joint_pos_indexes = [0, 1, 2, 3, 4, 5]
d.qpos[_ref_joint_pos_indexes] = init_qpos

from robosuite.renderers.context.egl_context import EGLGLContext as GLContext
gl_ctx = GLContext(max_width=width, max_height=height, device_id=-1)
gl_ctx.make_current()
mujoco.mj_forward(m, d)
scn = mujoco.MjvScene(m, maxgeom=1000)
cam = mujoco.MjvCamera()
cam.fixedcamid = 0
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
vopt = mujoco.MjvOption()
vopt.geomgroup[0] = 0
vopt.geomgroup[1] = 1
pert = mujoco.MjvPerturb()
pert.active = 0
pert.select = 0
pert.skinselect = -1
con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)

viewport = mujoco.MjrRect(0, 0, width, height)
mujoco.mjv_updateScene(m, d, vopt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
mujoco.mjr_render(viewport=viewport, scn=scn, con=con)

rgb_img = np.empty((480, 640, 3), dtype=np.uint8)
depth_img = np.empty((height, width), dtype=np.float32) if depth else None
mujoco.mjr_readPixels(rgb=rgb_img, depth=depth_img, viewport=viewport, con=con)

action = np.zeros(6)

start = time.time()
while time.time() - start < 5:
    im = rgb_img[..., ::-1]
    im = np.flip(im, axis=0)
    cv2.imshow("offscreen render", im)
    key = cv2.waitKey(1)
cv2.destroyAllWindows()



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