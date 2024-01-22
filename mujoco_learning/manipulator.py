import mujoco_py
import os
import numpy as np
from matplotlib import pyplot as plt


xml_path = 'model/manipulator.xml'
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

hinge_joint_1 = sim.policy_net.get_joint_qpos_addr("hinge:1")
hinge_joint_2 = sim.policy_net.get_joint_qpos_addr("hinge:2")

N = 500
i = 0
theta1 = np.pi / 3
theta2 = -np.pi / 2

# initialize
sim.data.qpos[0] = theta1
sim.data.qpos[1] = theta2
sim.forward()
J_pos = np.array(sim.data.get_site_jacp("tip").reshape((3, -1)))
print(J_pos)
position_Q = sim.data.site_xpos[0]
r = 0.5
center = np.array([position_Q[0] - r, position_Q[1]])

phi = np.linspace(0, 2 * np.pi, N)
x_ref = center[0] + r * np.cos(phi)
y_ref = center[1] + r * np.sin(phi)
x_all = []
y_all = []

# q0_start = 0
# q0_end = 1.57
# q1_start = 0
# q1_end = 2 * 3.14
# q0 = np.linspace(q0_start, q0_end, N)
# q1 = np.linspace(q1_start, q1_end, N)

while True:

    # sim_state = sim.get_state()
    # sim_state.qpos[hinge_joint_1] = q0[i]
    # sim_state.qpos[hinge_joint_2] = q1[i]
    # sim.set_state(sim_state)
    # sim.forward()
    # viewer.render()
    # print(sim.data.site_xpos[0])

    "Inverse kinematics using Jacobian"
    position_Q = sim.data.site_xpos[0]
    J_pos = np.array(sim.data.get_site_jacp("tip").reshape((3, -1)))
    J = J_pos[[0, 1], :]
    Jinv = np.linalg.inv(J)
    dX = np.array([x_ref[i] - position_Q[0], y_ref[i] - position_Q[1]])
    dq = Jinv.dot(dX)
    # print(dq)

    x_all.append(position_Q[0])
    y_all.append(position_Q[1])

    # update theta1 and theta2
    theta1 += dq[0]
    theta2 += dq[1]

    sim.data.qpos[0] = theta1
    sim.data.qpos[1] = theta2
    sim.forward()
    sim.step()
    viewer.render()
    i += 1
    if i >= N:
        plt.figure(1)
        plt.plot(x_all, y_all, 'bx')
        plt.plot(x_ref, y_ref, 'r-.')
        plt.ylabel("y")
        plt.xlabel("x")
        plt.legend(['pos', 'ref'])
        plt.gca().set_aspect('equal')
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        break

    if os.getenv('TESTING') is not None:
        break