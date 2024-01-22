import time

from RobotController import RobotController


robot = RobotController(render=True)
# robot.show_model_info()
# robot.display_current_values()
# print(len(robot.sim.data.ctrl))
# target = [1.2, 0.9, -1.5, 0.7, 0.5, 0.8, 0.5]
# target = [1.2, 0, 0, 0, 0, 0, 0]
target = [-0.137422, -0.0625251, 0.246773, -0.062914, -0.4083, 0.031419, -0.015677]
print(robot.dof)
robot.move_group_to_joint_target('LeftArm', target, plot=False, quiet=False)
robot.stay(1000)
# target = [1.5, -1.0, -0.5, 0.5, 1.0, -1]
# target = [-1, -0.5, 0, 0.8, -1, -1]
# robot.move_group_to_joint_target('LeftArm', target, plot=True, quiet=False)
# robot.move_group_to_joint_target('RightArm', target, plot=True, quiet=False)
# robot.stay(1000)
# robot.display_current_values()
# start_time = time.time()
# while time.time() - start_time < 5:
#     robot.sim.data.ctrl[0] = 1
#     robot.sim.step()
#     robot.render()