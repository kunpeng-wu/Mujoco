from collections import defaultdict
import os
from pathlib import Path
import mujoco_py as mp
import time
import numpy as np
from simple_pid import PID
from termcolor import colored
import ikpy.chain
import matplotlib.pyplot as plt
import copy

class RobotController:
    def __init__(self, render):
        # path = os.path.realpath(__file__)
        # path = str(Path(path).parent.parent.parent)
        self.model = mp.load_model_from_path('../model/robot_hand/body_new.xml')
        self.sim = mp.MjSim(self.model)
        self.is_render = render
        if self.is_render: self.viewer = mp.MjViewer(self.sim)
        self.create_lists()
        self.groups = defaultdict(list)
        self.groups["All"] = list(range(len(self.sim.data.ctrl)))
        self.create_group("LeftArm", list(range(7)))
        self.create_group("RightArm", list(range(7, 14)))
        self.actuated_joint_ids = np.array([i[2] for i in self.actuators])
        self.reached_target = False
        self.current_output = np.zeros(len(self.sim.data.ctrl))
        self.image_counter = 0

    def create_lists(self):
        self.controller_list = []

        # Values for training
        sample_time = 0.001
        # p_scale = 1
        p_scale = 2
        i_scale = 0.1
        d_scale = 0.1

        self.controller_list.append(PID(30 * p_scale, 0.0 * i_scale, 1.1 * d_scale,
                                        setpoint=0, output_limits=(-2, 2), sample_time=sample_time))  # Shoulder_1
        self.controller_list.append(PID(50 * p_scale, 0.0 * i_scale, 1.0 * d_scale,
                                        setpoint=0, output_limits=(-2, 2), sample_time=sample_time))  # Shoulder_2
        self.controller_list.append(PID(30 * p_scale, 0.0 * i_scale, 0.5 * d_scale,
                                        setpoint=0, output_limits=(-2, 2), sample_time=sample_time))  # Shoulder_3
        self.controller_list.append(PID(60 * p_scale, 0.0 * i_scale, 1.1 * d_scale,
                                        setpoint=0, output_limits=(-1, 1), sample_time=sample_time))  # Elbow
        self.controller_list.append(PID(50 * p_scale, 0.0 * i_scale, 1.1 * d_scale,
                                        setpoint=0, output_limits=(-1, 1), sample_time=sample_time))  # Wrist 1 Joint
        self.controller_list.append(PID(50 * p_scale, 0.0 * i_scale, 1.1 * d_scale,
                                        setpoint=0.0, output_limits=(-1, 1), sample_time=sample_time))  # Wrist 2 Joint
        self.controller_list.append(PID(50 * p_scale, 0.0 * i_scale, 1.1 * d_scale,
                                        setpoint=0.0, output_limits=(-1, 1), sample_time=sample_time))  # Wrist 3 Joint

        self.controller_list.append(PID(30 * p_scale, 0.0 * i_scale, 1.1 * d_scale,
                                        setpoint=0, output_limits=(-2, 2), sample_time=sample_time))  # Shoulder_1
        self.controller_list.append(PID(50 * p_scale, 0.0 * i_scale, 1.0 * d_scale,
                                        setpoint=0, output_limits=(-2, 2), sample_time=sample_time))  # Shoulder_2
        self.controller_list.append(PID(30 * p_scale, 0.0 * i_scale, 0.5 * d_scale,
                                        setpoint=0, output_limits=(-2, 2), sample_time=sample_time))  # Shoulder_3
        self.controller_list.append(PID(60 * p_scale, 0.0 * i_scale, 1.1 * d_scale,
                                        setpoint=0, output_limits=(-1, 1), sample_time=sample_time))  # Elbow
        self.controller_list.append(PID(50 * p_scale, 0.0 * i_scale, 1.1 * d_scale,
                                        setpoint=0, output_limits=(-1, 1), sample_time=sample_time))  # Wrist 1 Joint
        self.controller_list.append(PID(50 * p_scale, 0.0 * i_scale, 1.1 * d_scale,
                                        setpoint=0.0, output_limits=(-1, 1), sample_time=sample_time))  # Wrist 2 Joint
        self.controller_list.append(PID(50 * p_scale, 0.0 * i_scale, 1.1 * d_scale,
                                        setpoint=0.0, output_limits=(-1, 1), sample_time=sample_time))  # Wrist 3 Joint

        self.current_target_joint_values = np.array([
            self.controller_list[i].setpoint for i in range(len(self.sim.data.ctrl))
        ])

        self.current_output = [controller(0) for controller in self.controller_list]
        self.actuators = []
        for i in range(len(self.sim.data.ctrl)):
            item = [i, self.model.actuator_id2name(i)]
            item.append(self.model.actuator_trnid[i][0])
            item.append(self.model.joint_id2name(self.model.actuator_trnid[i][0]))
            item.append(self.controller_list[i])
            self.actuators.append(item)

    def create_group(self, group_name, idx_list):
        """
        Allows the user to create custom objects for controlling groups of joints.
        The method show_model_info can be used to get lists of joints and actuators.

        Args:
            group_name: String defining the désired name of the group.
            idx_list: List containing the IDs of the actuators that will belong to this group.
        """

        try:
            assert len(idx_list) <= len(self.sim.data.ctrl), "Too many joints specified!"
            assert (
                group_name not in self.groups.keys()
            ), "A group with name {} already exists!".format(group_name)
            assert np.max(idx_list) <= len(
                self.sim.data.ctrl
            ), "List contains invalid actuator ID (too high)"

            self.groups[group_name] = idx_list
            print(f"Created new control group '{group_name}':", self.groups[group_name])

        except Exception as e:
            print(e)
            print("Could not create a new group.")

    # def move_ee(self, ee_position, **kwargs):
    #     """
    #     Moves the robot arm so that the gripper center ends up at the requested XYZ-position,
    #     with a vertical gripper position.
    #
    #     Args:
    #         ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).
    #         plot: If True, a .png image of the arm joint trajectories will be saved to the local directory.
    #               This can be used for PID tuning in case of overshoot etc. The name of the file will be "Joint_angles_" + a number.
    #         marker: If True, a colored visual marker will be added into the scene to visualize the current
    #                 cartesian target.
    #     """
    #     # joint_angles = self.ik(ee_position)
    #     if joint_angles is not None:
    #         result = self.move_group_to_joint_target(group="Arm", target=joint_angles, **kwargs)
    #         # result = self.move_group_to_joint_target(group='Arm', target=joint_angles, tolerance=0.05, plot=plot, marker=marker, max_steps=max_steps, quiet=quiet, render=render)
    #     else:
    #         result = "No valid joint angles received, could not move EE to position."
    #         self.last_movement_steps = 0
    #     return result

    def move_group_to_joint_target(
        self,
        group="All",
        target=None,
        tolerance=0.03,
        max_steps=5000,
        plot=False,
        marker=False,
        quiet=False,
    ):
        """
        Moves the specified joint group to a joint target.

        Args:
            group: String specifying the group to move.
            target: List of target joint values for the group.
            tolerance: Threshold within which the error of each joint must be before the method finishes.
            max_steps: maximum number of steps to actuate before breaking
            plot: If True, a .png image of the group joint trajectories will be saved to the local directory.
                  This can be used for PID tuning in case of overshoot etc. The name of the file will be "Joint_angles_" + a number.
            marker: If True, a colored visual marker will be added into the scene to visualize the current
                    cartesian target.
        """

        try:
            assert group in self.groups.keys(), "No group with name {} exists!".format(group)
            if target is not None:
                assert len(target) == len(
                    self.groups[group]
                ), "Mismatching target dimensions for group {}!".format(group)
            ids = self.groups[group]

            steps = 1
            result = ""
            if plot:
                self.plot_list = defaultdict(list)
            self.reached_target = False
            deltas = np.zeros(len(self.sim.data.ctrl))

            if target is not None:
                for i, v in enumerate(ids):
                    self.current_target_joint_values[v] = target[i]
                    # print('Target joint value: {}: {}'.format(v, self.current_target_joint_values[v]))
            # print('target: ', self.current_target_joint_values)
            for j in range(ids[0], ids[-1] + 1):
                # Update the setpoints of the relevant controllers for the group
                self.actuators[j][4].setpoint = self.current_target_joint_values[j]
                # print('Setpoint {}: {}'.format(j, self.actuators[j][4].setpoint))

            while not self.reached_target:
                current_joint_values = self.sim.data.qpos[self.actuated_joint_ids]

                # We still want to actuate all motors towards their targets, otherwise the joints of non-controlled
                # groups will start to drift
                for j in range(ids[0], ids[-1] + 1):
                    self.current_output[j] = self.actuators[j][4](current_joint_values[j])
                    self.sim.data.ctrl[j] = self.current_output[j]
                for i in ids:
                    deltas[i] = abs(self.current_target_joint_values[i] - current_joint_values[i])
                # print(deltas)

                if steps % 1000 == 0 and target is not None and not quiet:
                    print(
                        "Moving group {} to joint target! Max. delta: {}, Joint: {}".format(
                            group, max(deltas), self.actuators[np.argmax(deltas)][3]
                        )
                    )

                if plot and steps % 2 == 0:
                    self.fill_plot_list(group, steps)
                # temp = self.sim.data.body_xpos[self.model.body_name2id("ee_link")] - [
                #     0,
                #     -0.005,
                #     0.16,
                # ]
                # if marker:
                #     self.add_marker(self.current_carthesian_target)
                #     self.add_marker(temp)

                if max(deltas) < tolerance:
                    if target is not None and not quiet:
                        print(f"Joint values for {group} within requested tolerance! ({steps} steps)")
                    result = "success"
                    self.reached_target = True
                    # break

                if steps > max_steps:
                    if not quiet:
                        print(f"Max number of steps reached: {max_steps}")
                        print("Deltas: ", deltas)
                    result = "max. steps reached: {}".format(max_steps)
                    break

                self.sim.step()
                if self.is_render:
                    self.viewer.render()
                steps += 1

            self.last_movement_steps = steps

            if plot:
                self.create_joint_angle_plot(group=group, tolerance=tolerance)

            return result

        except Exception as e:
            print(e)
            print("Could not move to requested joint target.")

    def stay(self, duration):
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:
            self.move_group_to_joint_target(max_steps=10, tolerance=0.001, plot=False, quiet=True)
            elapsed = (time.time() - starting_time)*1000

    def show_model_info(self):
        print("\nNumber of bodies: {}".format(self.model.nbody))
        for i in range(self.model.nbody):
            print("Body ID: {}, Body Name: {}".format(i, self.model.body_id2name(i)))

        print("\nNumber of joints: {}".format(self.model.njnt))
        for i in range(self.model.njnt):
            print(
                "Joint ID: {}, Joint Name: {}, Limits: {}".format(
                    i, self.model.joint_id2name(i), self.model.jnt_range[i]
                )
            )

        print("\nNumber of Actuators: {}".format(len(self.sim.data.ctrl)))
        for i in range(len(self.sim.data.ctrl)):
            print(
                "Actuator ID: {}, Actuator Name: {}, Controlled Joint: {}, Control Range: {}".format(
                    i,
                    self.model.actuator_id2name(i),
                    self.actuators[i][3],
                    self.model.actuator_ctrlrange[i],
                )
            )

    def fill_plot_list(self, group, step):
        """
        Creates a two-dimensional list of joint angles for plotting.

        Args:
            group: The group involved in the movement.
            step: The step of the trajectory the values correspond to.
        """

        for i in self.groups[group]:
            self.plot_list[self.actuators[i][3]].append(
                self.sim.data.qpos[self.actuated_joint_ids][i]
            )
        self.plot_list["Steps"].append(step)

    def create_joint_angle_plot(self, group, tolerance):
        """
        Saves the recorded joint values as a .png-file. The values for each joint of the group are
        put in a seperate subplot.

        Args:
            group: The group the stored values belong to.
            tolerance: The tolerance value that the joints were required to be in.
        """

        self.image_counter += 1
        keys = list(self.plot_list.keys())
        number_subplots = len(self.plot_list) - 1
        columns = 3
        rows = (number_subplots // columns) + (number_subplots % columns)

        position = range(1, number_subplots + 1)
        fig = plt.figure(1, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.05)

        for i in range(number_subplots):
            axis = fig.add_subplot(rows, columns, position[i])
            axis.plot(self.plot_list["Steps"], self.plot_list[keys[i]])
            axis.set_title(keys[i])
            axis.set_xlabel(keys[-1])
            axis.set_ylabel("Joint angle [rad]")
            axis.xaxis.set_label_coords(0.05, -0.1)
            axis.yaxis.set_label_coords(1.05, 0.5)
            axis.axhline(
                self.current_target_joint_values[self.groups[group][i]], color="g", linestyle="--"
            )
            axis.axhline(
                self.current_target_joint_values[self.groups[group][i]] + tolerance,
                color="r",
                linestyle="--",
            )
            axis.axhline(
                self.current_target_joint_values[self.groups[group][i]] - tolerance,
                color="r",
                linestyle="--",
            )
        plt.show()
        # filename = "Joint_values_{}.png".format(self.image_counter)
        # plt.savefig(filename)
        # print(f"Saved trajectory to {filename}")
        # plt.clf()

    def display_current_values(self):
        """
        Debug method, simply displays some relevant data at the time of the call.
        """
        print("\n################################################")
        print("CURRENT JOINT POSITIONS (ACTUATED)")
        print("################################################")
        for i in range(len(self.actuated_joint_ids)):
            print(
                "Current angle for joint {}: {}".format(
                    self.actuators[i][3], self.sim.data.qpos[self.actuated_joint_ids][i]
                )
            )

        # print("\n################################################")
        # print("CURRENT JOINT POSITIONS (ALL)")
        # print("################################################")
        # for i in range(len(self.model.jnt_qposadr)):
        #     # for i in range(self.model.njnt):
        #     name = self.model.joint_id2name(i)
        #     print("Current angle for joint {}: {}".format(name, self.sim.data.get_joint_qpos(name)))
        #     # print('Current angle for joint {}: {}'.format(self.model.joint_id2name(i), self.sim.data.qpos[i]))
        #
        # print("\n################################################")
        # print("CURRENT BODY POSITIONS")
        # print("################################################")
        # for i in range(self.model.nbody):
        #     print(
        #         "Current position for body {}: {}".format(
        #             self.model.body_id2name(i), self.sim.data.body_xpos[i]
        #         )
        #     )
        #
        # print("\n################################################")
        # print("CURRENT BODY ROTATION MATRIZES")
        # print("################################################")
        # for i in range(self.model.nbody):
        #     print(
        #         "Current rotation for body {}: {}".format(
        #             self.model.body_id2name(i), self.sim.data.body_xmat[i]
        #         )
        #     )
        #
        # print("\n################################################")
        # print("CURRENT BODY ROTATION QUATERNIONS (w,x,y,z)")
        # print("################################################")
        # for i in range(self.model.nbody):
        #     print(
        #         "Current rotation for body {}: {}".format(
        #             self.model.body_id2name(i), self.sim.data.body_xquat[i]
        #         )
        #     )
        # print("\n################################################")
        # print("CURRENT ACTUATOR CONTROLS")
        # print("################################################")
        # for i in range(len(self.sim.data.ctrl)):
        #     print(
        #         "Current activation of actuator {}: {}".format(
        #             self.actuators[i][1], self.sim.data.ctrl[i]
        #         )
        #     )

    def render(self):
        assert self.is_render, "Not set render=True at the beginning!"
        self.viewer.render()

    @property
    def dof(self):
        return len(self.sim.data.ctrl)

