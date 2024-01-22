import time

import mujoco
import mujoco.viewer
import numpy as np
import cv2

# m = mujoco.MjModel.from_xml_path('../model/ur5e/no_texture_robot.xml')
# m = mujoco.MjModel.from_xml_path('../model/mujoco_menagerie/franka_emika_panda/scene.xml')
# m = mujoco.MjModel.from_xml_path('../model/mujoco_menagerie/kuka_iiwa_14/scene.xml')
# m = mujoco.MjModel.from_xml_path('../model/mujoco_menagerie/universal_robots_ur5e/scene.xml')
# m = mujoco.MjModel.from_xml_path('../mujoco_menagerie/shadow_hand/scene_right.xml')
# m = mujoco.MjModel.from_xml_path('../mujoco_menagerie/unitree_go1/scene.xml')
# m = mujoco.MjModel.from_xml_path('../model/mujoco_menagerie/wonik_allegro/scene_left.xml')
# m = mujoco.MjModel.from_xml_path('../model/robot_hand/body_sim.xml')
# m = mujoco.MjModel.from_xml_path('../model/robot_hand/humanoid_test.xml')
# m = mujoco.MjModel.from_xml_path('../robot_hand/test.xml')
m = mujoco.MjModel.from_xml_path('../robot_hand/body_new.xml')
# m = mujoco.MjModel.from_xml_path('../model/robot_hand/hand_left.xml')
# m = mujoco.MjModel.from_xml_path('../robot_hand/hand_right.xml')
# m = mujoco.MjModel.from_xml_path('../model/robot_hand/hand_sim.xml')
# m = mujoco.MjModel.from_xml_path('../model/robot_hand/body_hand_sim.xml')
# m = mujoco.MjModel.from_xml_path('../model/UR5gripper_2_finger.xml')

d = mujoco.MjData(m)
height = 480
width = 640
depth = False

# from robosuite.renderers.context.egl_context import EGLGLContext as GLContext
# gl_ctx = GLContext(max_width=width, max_height=height, device_id=-1)
# gl_ctx.make_current()
# mujoco.mj_forward(m, d)
# scn = mujoco.MjvScene(m, maxgeom=1000)
# cam = mujoco.MjvCamera()
# cam.fixedcamid = 0
# cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
# vopt= mujoco.MjvOption()
# vopt.geomgroup[0] = 0
# vopt.geomgroup[1] = 1
# pert = mujoco.MjvPerturb()
# pert.active = 0
# pert.select = 0
# pert.skinselect = -1
# con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
# mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
#
# viewport = mujoco.MjrRect(0, 0, width, height)
# mujoco.mjv_updateScene(m, d, vopt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
# mujoco.mjr_render(viewport=viewport, scn=scn, con=con)
# viewport = mujoco.MjrRect(0, 0, width, height)
#
# rgb_img = np.empty((height, width, 3), dtype=np.uint8)
# depth_img = np.empty((height, width), dtype=np.float32) if depth else None
# mujoco.mjr_readPixels(rgb=rgb_img, depth=depth_img, viewport=viewport, con=con)
#
# start = time.time()
# while time.time() - start < 10:
#     im = rgb_img[..., ::-1]
#     im = np.flip(im, axis=0)
#     cv2.imshow("offscreen render", im)
#     key = cv2.waitKey(1)
# cv2.destroyAllWindows()

with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 1000:
        step_start = time.time()
        # print(m.data.get_site_xpos('tip'))
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)