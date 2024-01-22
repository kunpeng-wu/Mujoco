"""
Defines the base class of all grippers
"""
import numpy as np

from robot_hand.MujocoXMLModel import MujocoXMLModel

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import GRIPPER_COLLISION_COLOR, find_elements, string_to_array
from robosuite.utils.mjcf_utils import xml_path_completion


class GripperModel(MujocoXMLModel):
    """
    Base class for grippers

    Args:
        fname (str): Path to relevant xml file to create this gripper instance
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, fname, idn=0):
        super().__init__(fname, idn=idn)

        # Set variable to hold current action being outputted
        self.current_action = np.zeros(self.dof)

        # Grab gripper offset (string -> np.array -> elements [1, 2, 3, 0] (x, y, z, w))
        # This is the comopunded rotation with the base body and the eef body as well!
        base_quat = np.fromstring(self.worldbody[0].attrib.get("quat", "1 0 0 0"), dtype=np.float64, sep=" ")[[1, 2, 3, 0]]
        eef_element = find_elements(
            root=self.root, tags="body", attribs={"name": self.correct_naming("eef")}, return_first=True
        )
        eef_relative_quat = string_to_array(eef_element.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]
        self.rotation_offset = T.quat_multiply(eef_relative_quat, base_quat)

    def format_action(self, action):
        """
        Given (-1,1) abstract control as np-array
        returns the (-1,1) control signals
        for underlying actuators as 1-d np array
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------------------- #
    # Properties: In general, these are the name-adjusted versions from the private          #
    #             subclass implementations pulled from their respective raw xml files        #
    # -------------------------------------------------------------------------------------- #

    @property
    def naming_prefix(self):
        # return "gripper{}_".format(self.idn)
        return ""

    @property
    def speed(self):
        """
        How quickly the gripper opens / closes

        Returns:
            float: Speed of the gripper
        """
        return 0.0

    @property
    def dof(self):
        """
        Defines the number of DOF of the gripper

        Returns:
            int: gripper DOF
        """
        return len(self._actuators)

    @property
    def bottom_offset(self):
        return np.zeros(3)

    @property
    def top_offset(self):
        return np.zeros(3)

    @property
    def horizontal_radius(self):
        return 0

    @property
    def contact_geom_rgba(self):
        return GRIPPER_COLLISION_COLOR

    # -------------------------------------------------------------------------------------- #
    # All subclasses must implement the following properties                                 #
    # -------------------------------------------------------------------------------------- #

    @property
    def init_qpos(self):
        """
        Defines the default rest (open) qpos of the gripper

        Returns:
            np.array: Default init qpos of this gripper
        """
        raise NotImplementedError

    @property
    def _important_sites(self):
        """
        Sites used to aid visualization by human. (usually "grip_site" and "grip_cylinder")
        (and should be hidden from robots)

        Returns:
            dict:

                :`'grip_site'`: Name of grip actuation intersection location site
                :`'grip_cylinder'`: Name of grip actuation z-axis location site
                :`'ee'`: Name of end effector site
                :`'ee_x'`: Name of end effector site (x-axis)
                :`'ee_y'`: Name of end effector site (y-axis)
                :`'ee_z'`: Name of end effector site (z-axis)
        """
        return {
            "grip_site": "grip_site",
            "grip_cylinder": "grip_site_cylinder",
            "ee": "ee",
            "ee_x": "ee_x",
            "ee_y": "ee_y",
            "ee_z": "ee_z",
        }

    @property
    def _important_geoms(self):
        """
        Geoms corresponding to important components of the gripper (by default, left_finger, right_finger,
        left_fingerpad, right_fingerpad).
        Note that these are the raw string names directly pulled from a gripper's corresponding XML file,
        NOT the adjusted name with an auto-generated naming prefix

        Note that this should be a dict of lists.

        Returns:
            dict of list: Raw XML important geoms, where each set of geoms are grouped as a list and are
            organized by keyword string entries into a dict
        """
        return {
            "left_finger": [],
            "right_finger": [],
            "left_fingerpad": [],
            "right_fingerpad": [],
        }

    @property
    def _important_sensors(self):
        """
        Sensor names for each gripper (usually "force_ee" and "torque_ee")

        Returns:
            dict:

                :`'force_ee'`: Name of force eef sensor for this gripper
                :`'torque_ee'`: Name of torque eef sensor for this gripper
        """
        return {sensor: sensor for sensor in ["force_ee", "torque_ee"]}


class NullGripper(GripperModel):
    """
    Dummy Gripper class to represent no gripper

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/null_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None


class Robotiq85GripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/robotiq_gripper_85.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([-0.026, -0.267, -0.200, -0.026, -0.267, -0.200])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_outer_finger_collision",
                "left_inner_finger_collision",
                "left_fingertip_collision",
                "left_fingerpad_collision",
            ],
            "right_finger": [
                "right_outer_finger_collision",
                "right_inner_finger_collision",
                "right_fingertip_collision",
                "right_fingerpad_collision",
            ],
            "left_fingerpad": ["left_fingerpad_collision"],
            "right_fingerpad": ["right_fingerpad_collision"],
        }


class Robotiq85Gripper(Robotiq85GripperBase):
    """
    1-DoF variant of RobotiqGripperBase.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1


class HandLeft(GripperModel):
    def __init__(self, idn=0):
        super().__init__("/home/kavin/Documents/PycharmProjects/Robosuite/robosuite/models/assets/grippers/hand_left.xml", idn=idn)

    def format_action(self, action):
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def init_qpos(self):
        return np.zeros(16)

    @property
    def _important_geoms(self):
        return {}

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 12


if __name__ == '__main__':
    hand_left = HandLeft(0)
    print(hand_left.dof)
    print(hand_left.tendon)
    str = hand_left.get_xml()
