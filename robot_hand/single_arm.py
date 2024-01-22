import numpy as np
import copy
import os
from collections import OrderedDict


from robot_hand.controller_factory import controller_factory, load_controller_config
from robot_hand.MujocoXMLModel import MujocoXMLModel
from robot_hand.griper_factory import gripper_factory
from robot_hand.mount_factory import mount_factory

from robosuite.utils.mjcf_utils import ROBOT_COLLISION_COLOR, array_to_string, find_elements, string_to_array
from robosuite.utils.transform_utils import euler2mat, mat2quat
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.buffers import DeltaBuffer, RingBuffer
from robosuite.utils.observables import Observable, sensor

REGISTERED_ROBOTS = {}


def create_robot(name, *args, **kwargs):
    return REGISTERED_ROBOTS[name](*args, **kwargs)


class RobotModelMeta(type):
    def __new__(cls, name, bases, class_dict):
        cls = super().__new__(cls, name, bases, class_dict)
        REGISTERED_ROBOTS[cls.__name__] = cls
        return cls


class RobotModel(MujocoXMLModel, metaclass=RobotModelMeta):
    def set_base_xpos(self, pos):
        self._elements["root_body"].set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_ori(self, rot):
        """
        Rotates robot by rotation @rot from its original orientation.

        Args:
            rot (3-array): (r,p,y) euler angles specifying the orientation for the robot base
        """
        # xml quat assumes w,x,y,z so we need to convert to this format from outputted x,y,z,w format from fcn
        rot = mat2quat(euler2mat(rot))[[3, 0, 1, 2]]
        self._elements["root_body"].set("quat", array_to_string(rot))

    def set_joint_attribute(self, attrib, values, force=True):
        """
        Sets joint attributes, e.g.: friction loss, damping, etc.

        Args:
            attrib (str): Attribute to set for all joints
            values (n-array): Values to set for each joint
            force (bool): If True, will automatically override any pre-existing value. Otherwise, if a value already
                exists for this value, it will be skipped

        Raises:
            AssertionError: [Inconsistent dimension sizes]
        """
        assert values.size == len(self._elements["joints"]), (
            "Error setting joint attributes: "
            + "Values must be same size as joint dimension. Got {}, expected {}!".format(values.size, self.dof)
        )
        for i, joint in enumerate(self._elements["joints"]):
            if force or joint.get(attrib, None) is None:
                joint.set(attrib, array_to_string(np.array([values[i]])))

    def add_mount(self, mount):
        """
        Mounts @mount to arm.

        Throws error if robot already has a mount or if mount type is incorrect.

        Args:
            mount (MountModel): mount MJCF model

        Raises:
            ValueError: [mount already added]
        """
        if self.mount is not None:
            raise ValueError("Mount already added for this robot!")

        # First adjust mount's base position
        offset = self.base_offset - mount.top_offset
        mount._elements["root_body"].set("pos", array_to_string(offset))

        self.merge(mount, merge_body=self.root_body)

        self.mount = mount

        # Update cameras in this model
        self.cameras = self.get_element_names(self.worldbody, "camera")

    @property
    def naming_prefix(self):
        # return "robot{}_".format(self.idn)
        return ""
    @property
    def dof(self):
        """
        Defines the number of DOF of the robot

        Returns:
            int: robot DOF
        """
        return len(self._joints)

    @property
    def bottom_offset(self):
        """
        Returns vector from model root body to model bottom.
        By default, this is equivalent to this robot's mount's (bottom_offset - top_offset) + this robot's base offset

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        return (
            (self.mount.bottom_offset - self.mount.top_offset) + self._base_offset
            if self.mount is not None
            else self._base_offset
        )

    @property
    def horizontal_radius(self):
        """
        Returns maximum distance from model root body to any radial point of the model. This method takes into
        account the mount horizontal radius as well

        Returns:
            float: radius
        """
        return max(self._horizontal_radius, self.mount.horizontal_radius)

    @property
    def models(self):
        """
        Returns a list of all m(sub-)models owned by this robot model. By default, this includes the mount model,
        if specified

        Returns:
            list: models owned by this object
        """
        return [self.mount] if self.mount is not None else []

    @property
    def contact_geom_rgba(self):
        return ROBOT_COLLISION_COLOR

    @property
    def default_mount(self):
        """
        Defines the default mount type for this robot that gets added to root body (base)

        Returns:
            str: Default mount name to add to this robot
        """
        raise NotImplementedError

    @property
    def default_controller_config(self):
        """
        Defines the name of default manipulator config file in the controllers/config directory for this robot.

        Returns:
            str: filename of default manipulator config for this robot
        """
        raise NotImplementedError

    @property
    def init_qpos(self):
        """
        Defines the default rest qpos of this robot

        Returns:
            np.array: Default init qpos of this robot
        """
        raise NotImplementedError

    @property
    def base_xpos_offset(self):
        """
        Defines the dict of various (x,y,z) tuple offsets relative to specific arenas placed at (0,0,0)
        Assumes robot is facing forwards (in the +x direction) when determining offset. Should have entries for each
        arena case; i.e.: "bins", "empty", and "table")

        Returns:
            dict: Dict mapping arena names to robot offsets from the global origin (dict entries may also be lambdas
                for variable offsets)
        """
        raise NotImplementedError

    @property
    def top_offset(self):
        """
        Returns vector from model root body to model top.
        Useful for, e.g. placing models on a surface.
        Must be defined by subclass.

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        raise NotImplementedError

    @property
    def _horizontal_radius(self):
        """
        Returns maximum distance from model root body to any radial point of the model.

        Helps us put models programmatically without them flying away due to a huge initial contact force.
        Must be defined by subclass.

        Returns:
            float: radius
        """
        raise NotImplementedError

    @property
    def _important_sites(self):
        """
        Returns:
            dict: (Default is no important sites; i.e.: empty dict)
        """
        return {}

    @property
    def _important_geoms(self):
        """
        Returns:
             dict: (Default is no important geoms; i.e.: empty dict)
        """
        return {}

    @property
    def _important_sensors(self):
        """
        Returns:
            dict: (Default is no sensors; i.e.: empty dict)
        """
        return {}


class UR5e(RobotModel):
    def __init__(self, idn):
        super().__init__(xml_path_completion("robots/ur5e/robot.xml"), idn)
        # key: gripper name and value: gripper model
        self.grippers = OrderedDict()

        # Grab hand's offset from final robot link (string -> np.array -> elements [1, 2, 3, 0] (x, y, z, w))
        # Different case based on whether we're dealing with single or bimanual armed robot
        if self.arm_type == "single":
            hand_element = find_elements(
                root=self.root, tags="body", attribs={"name": self.eef_name}, return_first=True
            )
            self.hand_rotation_offset = string_to_array(hand_element.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]
        else:  # "bimanual" case
            self.hand_rotation_offset = {}
            for arm in ("right", "left"):
                hand_element = find_elements(
                    root=self.root, tags="body", attribs={"name": self.eef_name[arm]}, return_first=True
                )
                self.hand_rotation_offset[arm] = string_to_array(hand_element.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]

        # Get camera names for this robot
        self.cameras = self.get_element_names(self.worldbody, "camera")

    def add_gripper(self, gripper, arm_name=None):
        """
        Mounts @gripper to arm.

        Throws error if robot already has a gripper or gripper type is incorrect.

        Args:
            gripper (GripperModel): gripper MJCF model
            arm_name (str): name of arm mount -- defaults to self.eef_name if not specified

        Raises:
            ValueError: [Multiple grippers]
        """
        if arm_name is None:
            arm_name = self.eef_name
        if arm_name in self.grippers:
            raise ValueError("Attempts to add multiple grippers to one body")

        self.merge(gripper, merge_body=arm_name)

        self.grippers[arm_name] = gripper

        # Update cameras in this model
        self.cameras = self.get_element_names(self.worldbody, "camera")

    @property
    def eef_name(self):
        """
        Returns:
            str or dict of str: Prefix-adjusted eef name for this robot. If bimanual robot, returns {"left", "right"}
                keyword-mapped eef names
        """
        return self.correct_naming(self._eef_name)

    @property
    def models(self):
        """
        Returns a list of all m(sub-)models owned by this robot model. By default, this includes the gripper model,
        if specified

        Returns:
            list: models owned by this object
        """
        models = super().models
        return models + list(self.grippers.values())

    # -------------------------------------------------------------------------------------- #
    # -------------------------- Private Properties ---------------------------------------- #
    # -------------------------------------------------------------------------------------- #

    @property
    def _important_sites(self):
        """
        Returns:
            dict: (Default is no important sites; i.e.: empty dict)
        """
        return {}

    @property
    def _eef_name(self):
        """
        XML eef name for this robot to which grippers can be attached. Note that these should be the raw
        string names directly pulled from a robot's corresponding XML file, NOT the adjusted name with an
        auto-generated naming prefix

        Returns:
            str: Raw XML eef name for this robot (default is "right_hand")
        """
        return "right_hand"

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        # return "Robotiq85Gripper"
        return None
        # return "HandLeft"

    @property
    def default_controller_config(self):
        return "osc_pose"

    @property
    def init_qpos(self):
        return np.array([-0.470, -1.735, 2.480, -2.275, -1.590, -1.991])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


class Humanoid(RobotModel):
    def __init__(self, idn):
        super().__init__("/home/kavin/Documents/PycharmProjects/Robosuite/robosuite/models/assets/robots/humanoid/body.xml", idn)
        # key: gripper name and value: gripper model
        self.grippers = OrderedDict()

        # Grab hand's offset from final robot link (string -> np.array -> elements [1, 2, 3, 0] (x, y, z, w))
        # Different case based on whether we're dealing with single or bimanual armed robot
        if self.arm_type == "single":
            hand_element = find_elements(
                root=self.root, tags="body", attribs={"name": self.eef_name}, return_first=True
            )
            self.hand_rotation_offset = string_to_array(hand_element.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]  # (x, y, z, w)
        else:  # "bimanual" case
            self.hand_rotation_offset = {}
            for arm in ("right", "left"):
                hand_element = find_elements(
                    root=self.root, tags="body", attribs={"name": self.eef_name[arm]}, return_first=True
                )
                self.hand_rotation_offset[arm] = string_to_array(hand_element.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]

        # Get camera names for this robot
        self.cameras = self.get_element_names(self.worldbody, "camera")

    def add_gripper(self, gripper, arm_name=None):
        """
        Mounts @gripper to arm.

        Throws error if robot already has a gripper or gripper type is incorrect.

        Args:
            gripper (GripperModel): gripper MJCF model
            arm_name (str): name of arm mount -- defaults to self.eef_name if not specified

        Raises:
            ValueError: [Multiple grippers]
        """
        if arm_name is None:
            arm_name = self.eef_name
        if arm_name in self.grippers:
            raise ValueError("Attempts to add multiple grippers to one body")

        self.merge(gripper, merge_body=arm_name)

        self.grippers[arm_name] = gripper

        # Update cameras in this model
        self.cameras = self.get_element_names(self.worldbody, "camera")

    @property
    def eef_name(self):
        """
        Returns:
            str or dict of str: Prefix-adjusted eef name for this robot. If bimanual robot, returns {"left", "right"}
                keyword-mapped eef names
        """
        return self.correct_naming(self._eef_name)

    @property
    def models(self):
        """
        Returns a list of all m(sub-)models owned by this robot model. By default, this includes the gripper model,
        if specified

        Returns:
            list: models owned by this object
        """
        models = super().models
        return models + list(self.grippers.values())

    # -------------------------------------------------------------------------------------- #
    # -------------------------- Private Properties ---------------------------------------- #
    # -------------------------------------------------------------------------------------- #

    @property
    def _important_sites(self):
        """
        Returns:
            dict: (Default is no important sites; i.e.: empty dict)
        """
        return {}

    @property
    def _eef_name(self):
        """
        XML eef name for this robot to which grippers can be attached. Note that these should be the raw
        string names directly pulled from a robot's corresponding XML file, NOT the adjusted name with an
        auto-generated naming prefix

        Returns:
            str: Raw XML eef name for this robot (default is "right_hand")
        """
        return "left_hand"
        # return {"right": "right_hand", "left": "left_hand"}

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        # return "Robotiq85Gripper"
        return "HandLeft"
        # return None

    @property
    def default_controller_config(self):
        return "osc_pose"

    @property
    def init_qpos(self):
        return np.array([0., 0, 0, 0, 0, 0], dtype=np.float64)

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


class Robot(object):
    """
    Initializes a robot simulation object, as defined by a single corresponding robot XML

    Args:
        robot_type (str): Specification for specific robot arm to be instantiated within this env (e.g: "Panda")

        idn (int or str): Unique ID of this robot. Should be different from others

        initial_qpos (sequence of float): If set, determines the initial joint positions of the robot to be
            instantiated for the task

        initialization_noise (dict): Dict containing the initialization noise parameters. The expected keys and
            corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to "None" or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            :Note: Specifying None will automatically create the required dict with "magnitude" set to 0.0

        mount_type (str): type of mount, used to instantiate mount models from mount factory.
            Default is "default", which is the default mount associated with this robot's corresponding model.
            None results in no mount, and any other (valid) model overrides the default mount.

        control_freq (float): how many control signals to receive
            in every second. This sets the amount of simulation time
            that passes between every action input.
    """

    def __init__(
        self,
        robot_type: str,
        idn=0,
        initial_qpos=None,
        initialization_noise=None,
        mount_type="default",
        control_freq=20,
    ):
        # Set relevant attributes
        self.sim = None  # MjSim this robot is tied to
        self.name = robot_type  # Specific robot to instantiate
        self.idn = idn  # Unique ID of this robot
        self.robot_model = None  # object holding robot model-specific info
        self.control_freq = control_freq  # manipulator Hz
        self.mount_type = mount_type  # Type of mount to use

        # Scaling of Gaussian initial noise applied to robot joints
        self.initialization_noise = initialization_noise
        if self.initialization_noise is None:
            self.initialization_noise = {"magnitude": 0.0, "type": "gaussian"}  # no noise conditions
        elif self.initialization_noise == "default":
            self.initialization_noise = {"magnitude": 0.02, "type": "gaussian"}
        self.initialization_noise["magnitude"] = (
            self.initialization_noise["magnitude"] if self.initialization_noise["magnitude"] else 0.0
        )

        self.init_qpos = initial_qpos  # n-dim list / array of robot joints

        self.robot_joints = None  # xml joint names for robot
        self.base_pos = None  # Base position in world coordinates (x,y,z)
        self.base_ori = None  # Base rotation in world coordinates (x,y,z,w quat)
        self._ref_joint_indexes = None  # xml joint indexes for robot in mjsim
        self._ref_joint_pos_indexes = None  # xml joint position indexes in mjsim
        self._ref_joint_vel_indexes = None  # xml joint velocity indexes in mjsim
        self._ref_joint_actuator_indexes = None  # xml joint (torq) actuator indexes for robot in mjsim

        self.recent_qpos = None  # Current and last robot arm qpos
        self.recent_actions = None  # Current and last action applied
        self.recent_torques = None  # Current and last torques applied

    def _load_controller(self):
        """
        Loads manipulator to be used for dynamic trajectories.
        """
        raise NotImplementedError

    def load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        self.robot_model = create_robot(self.name, idn=self.idn)

        # Add mount if specified
        if self.mount_type == "default":
            self.robot_model.add_mount(mount=mount_factory(self.robot_model.default_mount, idn=self.idn))
        else:
            self.robot_model.add_mount(mount=mount_factory(self.mount_type, idn=self.idn))

        # Use default from robot model for initial joint positions if not specified
        if self.init_qpos is None:
            self.init_qpos = self.robot_model.init_qpos

    def reset_sim(self, sim: MjSim):
        """
        Replaces current sim with a new sim

        Args:
            sim (MjSim): New simulation being instantiated to replace the old one
        """
        self.sim = sim

    def reset(self, deterministic=False):
        """
        Sets initial pose of arm and grippers. Overrides robot joint configuration if we're using a
        deterministic reset (e.g.: hard reset from xml file)

        Args:
            deterministic (bool): If true, will not randomize initializations within the sim

        Raises:
            ValueError: [Invalid noise type]
        """
        init_qpos = np.array(self.init_qpos)
        if not deterministic:
            # Determine noise
            if self.initialization_noise["type"] == "gaussian":
                noise = np.random.randn(len(self.init_qpos)) * self.initialization_noise["magnitude"]
            elif self.initialization_noise["type"] == "uniform":
                noise = np.random.uniform(-1.0, 1.0, len(self.init_qpos)) * self.initialization_noise["magnitude"]
            else:
                raise ValueError("Error: Invalid noise type specified. Options are 'gaussian' or 'uniform'.")
            init_qpos += noise

        # Set initial position in sim
        self.sim.data.qpos[self._ref_joint_pos_indexes] = init_qpos

        # Load controllers
        self._load_controller()

        # Update base pos / ori references
        self.base_pos = self.sim.data.get_body_xpos(self.robot_model.root_body)
        self.base_ori = T.mat2quat(self.sim.data.get_body_xmat(self.robot_model.root_body).reshape((3, 3)))

        # Setup buffers to hold recent values
        self.recent_qpos = DeltaBuffer(dim=len(self.joint_indexes))
        self.recent_actions = DeltaBuffer(dim=self.action_dim)
        self.recent_torques = DeltaBuffer(dim=len(self.joint_indexes))

    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.
        """
        # indices for joints in qpos, qvel
        self.robot_joints = self.robot_model.joints
        self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints]
        self._ref_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints]

        # indices for joint indexes
        self._ref_joint_indexes = [self.sim.model.joint_name2id(joint) for joint in self.robot_model.joints]

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_joint_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator) for actuator in self.robot_model.actuators
        ]

    def setup_observables(self):
        """
        Sets up observables to be used for this robot

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        # Get prefix from robot model to avoid naming clashes for multiple robots and define observables modality
        pf = self.robot_model.naming_prefix
        pre_compute = f"{pf}joint_pos"
        modality = f"{pf}proprio"

        # proprioceptive features
        @sensor(modality=modality)
        def joint_pos(obs_cache):
            return np.array([self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes])

        @sensor(modality=modality)
        def joint_pos_cos(obs_cache):
            return np.cos(obs_cache[pre_compute]) if pre_compute in obs_cache else np.zeros(self.robot_model.dof)

        @sensor(modality=modality)
        def joint_pos_sin(obs_cache):
            return np.sin(obs_cache[pre_compute]) if pre_compute in obs_cache else np.zeros(self.robot_model.dof)

        @sensor(modality=modality)
        def joint_vel(obs_cache):
            return np.array([self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes])

        sensors = [joint_pos, joint_pos_cos, joint_pos_sin, joint_vel]
        names = ["joint_pos", "joint_pos_cos", "joint_pos_sin", "joint_vel"]
        # We don't want to include the direct joint pos sensor outputs
        actives = [False, True, True, True]

        # Create observables for this robot
        observables = OrderedDict()
        for name, s, active in zip(names, sensors, actives):
            obs_name = pf + name
            observables[obs_name] = Observable(
                name=obs_name,
                sensor=s,
                sampling_rate=self.control_freq,
                active=active,
            )

        return observables

    def control(self, action, policy_step=False):
        """
        Actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (np.array): The control to apply to the robot. The first @self.robot_model.dof dimensions should
                be the desired normalized joint velocities and if the robot has a gripper, the next @self.gripper.dof
                dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken
        """
        raise NotImplementedError

    def check_q_limits(self):
        """
        Check if this robot is either very close or at the joint limits

        Returns:
            bool: True if this arm is near its joint limits
        """
        tolerance = 0.1
        for (qidx, (q, q_limits)) in enumerate(
            zip(self.sim.data.qpos[self._ref_joint_pos_indexes], self.sim.model.jnt_range[self._ref_joint_indexes])
        ):
            if q_limits[0] != q_limits[1] and not (q_limits[0] + tolerance < q < q_limits[1] - tolerance):
                print("Joint limit reached in joint " + str(qidx))
                return True
        return False

    def visualize(self, vis_settings):
        """
        Do any necessary visualization for this robot

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "robots" keyword as well as any other robot-specific
                options specified.
        """
        self.robot_model.set_sites_visibility(sim=self.sim, visible=vis_settings["robots"])

    @property
    def action_limits(self):
        """
        Action lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        raise NotImplementedError

    @property
    def torque_limits(self):
        """
        Torque lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) torque values
                - (np.array) maximum (high) torque values
        """
        # Torque limit values pulled from relevant robot.xml file
        low = self.sim.model.actuator_ctrlrange[self._ref_joint_actuator_indexes, 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_joint_actuator_indexes, 1]

        return low, high

    @property
    def action_dim(self):
        """
        Action space dimension for this robot
        """
        return self.action_limits[0].shape[0]

    @property
    def dof(self):
        """
        Returns:
            int: the active DoF of the robot (Number of robot joints + active gripper DoF).
        """
        dof = self.robot_model.dof
        return dof

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.

        Args:
            name (str): Name of body in sim to grab pose

        Returns:
            np.array: (4,4) array corresponding to the pose of @name in the base frame
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos(self.robot_model.root_body)
        base_rot_in_world = self.sim.data.get_body_xmat(self.robot_model.root_body).reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        """
        Helper method to force robot joint positions to the passed values.

        Args:
            jpos (np.array): Joint positions to manually set the robot to
        """
        self.sim.data.qpos[self._ref_joint_pos_indexes] = jpos
        self.sim.forward()

    @property
    def js_energy(self):
        """
        Returns:
            np.array: the energy consumed by each joint between previous and current steps
        """
        # We assume in the motors torque is proportional to current (and voltage is constant)
        # In that case the amount of power scales proportional to the torque and the energy is the
        # time integral of that
        # Note that we use mean torque
        return np.abs((1.0 / self.control_freq) * self.recent_torques.average)

    @property
    def _joint_positions(self):
        """
        Returns:
            np.array: joint positions (in angles / radians)
        """
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """
        Returns:
            np.array: joint velocities (angular velocity)
        """
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    @property
    def joint_indexes(self):
        """
        Returns:
            list: mujoco internal indexes for the robot joints
        """
        return self._ref_joint_indexes

    def get_sensor_measurement(self, sensor_name):
        """
        Grabs relevant sensor data from the sim object

        Args:
            sensor_name (str): name of the sensor

        Returns:
            np.array: sensor values
        """
        sensor_idx = np.sum(self.sim.model.sensor_dim[: self.sim.model.sensor_name2id(sensor_name)])
        sensor_dim = self.sim.model.sensor_dim[self.sim.model.sensor_name2id(sensor_name)]
        return np.array(self.sim.data.sensordata[sensor_idx : sensor_idx + sensor_dim])


class SingleArm(Robot):
    """
    Initializes a single-armed robot simulation object.

    Args:
        robot_type (str): Specification for specific robot arm to be instantiated within this env (e.g: "Panda")

        idn (int or str): Unique ID of this robot. Should be different from others

        controller_config (dict): If set, contains relevant manipulator parameters for creating a custom manipulator.
            Else, uses the default manipulator for this specific task

        initial_qpos (sequence of float): If set, determines the initial joint positions of the robot to be
            instantiated for the task

        initialization_noise (dict): Dict containing the initialization noise parameters. The expected keys and
            corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to "None" or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            :Note: Specifying None will automatically create the required dict with "magnitude" set to 0.0

        mount_type (str): type of mount, used to instantiate mount models from mount factory.
            Default is "default", which is the default mount associated with this robot's corresponding model.
            None results in no mount, and any other (valid) model overrides the default mount.

        gripper_type (str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default gripper associated
            within the 'robot' specification. None removes the gripper, and any other (valid) model overrides the
            default gripper

        control_freq (float): how many control signals to receive
            in every second. This sets the amount of simulation time
            that passes between every action input.
    """

    def __init__(
        self,
        robot_type: str,
        idn=0,
        controller_config=None,
        initial_qpos=None,
        initialization_noise=None,
        mount_type="default",
        gripper_type="default",
        control_freq=20,
    ):

        self.controller = None
        self.controller_config = copy.deepcopy(controller_config)
        self.gripper_type = gripper_type
        self.has_gripper = self.gripper_type is not None

        self.gripper = None  # Gripper class
        self.gripper_joints = None  # xml joint names for gripper
        self._ref_gripper_joint_pos_indexes = None  # xml gripper joint position indexes in mjsim
        self._ref_gripper_joint_vel_indexes = None  # xml gripper joint velocity indexes in mjsim
        self._ref_joint_gripper_actuator_indexes = None  # xml gripper (pos) actuator indexes for robot in mjsim
        self.eef_rot_offset = None  # rotation offsets from final arm link to gripper (quat)
        self.eef_site_id = None  # xml element id for eef in mjsim
        self.eef_cylinder_id = None  # xml element id for eef cylinder in mjsim
        self.torques = None  # Current torques being applied

        self.recent_ee_forcetorques = None  # Current and last forces / torques sensed at eef
        self.recent_ee_pose = None  # Current and last eef pose (pos + ori (quat))
        self.recent_ee_vel = None  # Current and last eef velocity
        self.recent_ee_vel_buffer = None  # RingBuffer holding prior 10 values of velocity values
        self.recent_ee_acc = None  # Current and last eef acceleration

        super().__init__(
            robot_type=robot_type,
            idn=idn,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
            mount_type=mount_type,
            control_freq=control_freq,
        )

    def _load_controller(self):
        """
        Loads manipulator to be used for dynamic trajectories
        """
        # First, load the default manipulator if none is specified
        if not self.controller_config:
            # Need to update default for a single agent
            controller_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "controllers/config/{}.json".format(self.robot_model.default_controller_config),
            )
            self.controller_config = load_controller_config(custom_fpath=controller_path)

        # Assert that the manipulator config is a dict file:
        #             NOTE: "type" must be one of: {JOINT_POSITION, JOINT_TORQUE, JOINT_VELOCITY,
        #                                           OSC_POSITION, OSC_POSE, IK_POSE}
        assert (
            type(self.controller_config) == dict
        ), "Inputted manipulator config must be a dict! Instead, got type: {}".format(type(self.controller_config))

        # Add to the manipulator dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self.controller_config["robot_name"] = self.name
        self.controller_config["sim"] = self.sim
        self.controller_config["eef_name"] = self.gripper.important_sites["grip_site"]
        self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        self.controller_config["joint_indexes"] = {
            "joints": self.joint_indexes,
            "qpos": self._ref_joint_pos_indexes,
            "qvel": self._ref_joint_vel_indexes,
        }
        self.controller_config["actuator_range"] = self.torque_limits
        self.controller_config["policy_freq"] = self.control_freq
        self.controller_config["ndim"] = len(self.robot_joints)

        # Instantiate the relevant manipulator
        self.controller = controller_factory(self.controller_config["type"], self.controller_config)

    def load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        # First, run the superclass method to load the relevant model
        super().load_model()

        # Verify that the loaded model is of the correct type for this robot
        if self.robot_model.arm_type != "single":
            raise TypeError(
                "Error loading robot model: Incompatible arm type specified for this robot. "
                "Requested model arm type: {}, robot arm type: {}".format(self.robot_model.arm_type, type(self))
            )

        # Now, load the gripper if necessary
        if self.has_gripper:
            if self.gripper_type == "default":
                # Load the default gripper from the robot file
                self.gripper = gripper_factory(self.robot_model.default_gripper, idn=self.idn)
            else:
                # Load user-specified gripper
                self.gripper = gripper_factory(self.gripper_type, idn=self.idn)
        else:
            # Load null gripper
            self.gripper = gripper_factory(None, idn=self.idn)
        # Grab eef rotation offset
        self.eef_rot_offset = T.quat_multiply(self.robot_model.hand_rotation_offset, self.gripper.rotation_offset)
        # Add gripper to this robot model
        self.robot_model.add_gripper(self.gripper)

    def reset(self, deterministic=False):
        """
        Sets initial pose of arm and grippers. Overrides gripper joint configuration if we're using a
        deterministic reset (e.g.: hard reset from xml file)

        Args:
            deterministic (bool): If true, will not randomize initializations within the sim
        """
        # First, run the superclass method to reset the position and manipulator
        super().reset(deterministic)

        # Now, reset the gripper if necessary
        if self.has_gripper:
            if not deterministic:
                self.sim.data.qpos[self._ref_gripper_joint_pos_indexes] = self.gripper.init_qpos

            self.gripper.current_action = np.zeros(self.gripper.dof)

        # Update base pos / ori references in manipulator
        self.controller.update_base_pose(self.base_pos, self.base_ori)

        # # Setup buffers to hold recent values
        self.recent_ee_forcetorques = DeltaBuffer(dim=6)
        self.recent_ee_pose = DeltaBuffer(dim=7)
        self.recent_ee_vel = DeltaBuffer(dim=6)
        self.recent_ee_vel_buffer = RingBuffer(dim=6, length=10)
        self.recent_ee_acc = DeltaBuffer(dim=6)

    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.

        Note that this should get called during every reset from the environment
        """
        # First, run the superclass method to setup references for joint-related values / indexes
        super().setup_references()

        # Now, add references to gripper if necessary
        # indices for grippers in qpos, qvel
        if self.has_gripper:
            self.gripper_joints = list(self.gripper.joints)
            self._ref_gripper_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints]
            self._ref_gripper_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints]
            self._ref_joint_gripper_actuator_indexes = [
                self.sim.model.actuator_name2id(actuator) for actuator in self.gripper.actuators
            ]

        # IDs of sites for eef visualization
        self.eef_site_id = self.sim.model.site_name2id(self.gripper.important_sites["grip_site"])
        self.eef_cylinder_id = self.sim.model.site_name2id(self.gripper.important_sites["grip_cylinder"])

    def control(self, action, policy_step=False):
        """
        Actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (np.array): The control to apply to the robot. The first @self.robot_model.dof dimensions should be
                the desired normalized joint velocities and if the robot has a gripper, the next @self.gripper.dof
                dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken

        Raises:
            AssertionError: [Invalid action dimension]
        """

        # clip actions into valid range
        assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )

        gripper_action = None
        if self.has_gripper:
            gripper_action = action[self.controller.control_dim :]  # all indexes past manipulator dimension indexes
            arm_action = action[: self.controller.control_dim]
        else:
            arm_action = action

        # Update the manipulator goal if this is a new policy step
        if policy_step:
            self.controller.set_goal(arm_action)

        # Now run the manipulator for a step
        torques = self.controller.run_controller()

        # Clip the torques
        low, high = self.torque_limits
        self.torques = np.clip(torques, low, high)

        # Get gripper action, if applicable
        if self.has_gripper:
            self.grip_action(gripper=self.gripper, gripper_action=gripper_action)

        # Apply joint torque control
        self.sim.data.ctrl[self._ref_joint_actuator_indexes] = self.torques

        # If this is a policy step, also update buffers holding recent values of interest
        if policy_step:
            # Update proprioceptive values
            self.recent_qpos.push(self._joint_positions)
            self.recent_actions.push(action)
            self.recent_torques.push(self.torques)
            self.recent_ee_forcetorques.push(np.concatenate((self.ee_force, self.ee_torque)))
            self.recent_ee_pose.push(np.concatenate((self.controller.ee_pos, T.mat2quat(self.controller.ee_ori_mat))))
            self.recent_ee_vel.push(np.concatenate((self.controller.ee_pos_vel, self.controller.ee_ori_vel)))

            # Estimation of eef acceleration (averaged derivative of recent velocities)
            self.recent_ee_vel_buffer.push(np.concatenate((self.controller.ee_pos_vel, self.controller.ee_ori_vel)))
            diffs = np.vstack(
                [self.recent_ee_acc.current, self.control_freq * np.diff(self.recent_ee_vel_buffer.buf, axis=0)]
            )
            ee_acc = np.array([np.convolve(col, np.ones(10) / 10.0, mode="valid")[0] for col in diffs.transpose()])
            self.recent_ee_acc.push(ee_acc)

    def _visualize_grippers(self, visible):
        """
        Visualizes the gripper site(s) if applicable.

        Args:
            visible (bool): True if visualizing the gripper for this arm.
        """
        self.gripper.set_sites_visibility(sim=self.sim, visible=visible)

    def setup_observables(self):
        """
        Sets up observables to be used for this robot

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        # Get general robot observables first
        observables = super().setup_observables()

        # Get prefix from robot model to avoid naming clashes for multiple robots and define observables modality
        pf = self.robot_model.naming_prefix
        modality = f"{pf}proprio"

        # eef features
        @sensor(modality=modality)
        def eef_pos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.eef_site_id])

        @sensor(modality=modality)
        def eef_quat(obs_cache):
            return T.convert_quat(self.sim.data.get_body_xquat(self.robot_model.eef_name), to="xyzw")

        @sensor(modality=modality)
        def eef_vel_lin(obs_cache):
            return np.array(self.sim.data.get_body_xvelp(self.robot_model.eef_name))

        @sensor(modality=modality)
        def eef_vel_ang(obs_cache):
            return np.array(self.sim.data.get_body_xvelr(self.robot_model.eef_name))

        sensors = [eef_pos, eef_quat, eef_vel_lin, eef_vel_ang]
        names = [f"{pf}eef_pos", f"{pf}eef_quat", f"{pf}eef_vel_lin", f"{pf}eef_vel_ang"]
        # Exclude eef vel by default
        actives = [True, True, False, False]

        # add in gripper sensors if this robot has a gripper
        if self.has_gripper:

            @sensor(modality=modality)
            def gripper_qpos(obs_cache):
                return np.array([self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes])

            @sensor(modality=modality)
            def gripper_qvel(obs_cache):
                return np.array([self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes])

            sensors += [gripper_qpos, gripper_qvel]
            names += [f"{pf}gripper_qpos", f"{pf}gripper_qvel"]
            actives += [True, True]

        # Create observables for this robot
        for name, s, active in zip(names, sensors, actives):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
                active=active,
            )

        return observables

    @property
    def action_limits(self):
        """
        Action lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        # Action limits based on manipulator limits
        low, high = ([-1] * self.gripper.dof, [1] * self.gripper.dof) if self.has_gripper else ([], [])
        low_c, high_c = self.controller.control_limits
        low = np.concatenate([low_c, low])
        high = np.concatenate([high_c, high])

        return low, high

    @property
    def ee_ft_integral(self):
        """
        Returns:
            np.array: the integral over time of the applied ee force-torque
        """
        return np.abs((1.0 / self.control_freq) * self.recent_ee_forcetorques.average)

    @property
    def ee_force(self):
        """
        Returns:
            np.array: force applied at the force sensor at the robot arm's eef
        """
        return self.get_sensor_measurement(self.gripper.important_sensors["force_ee"])

    @property
    def ee_torque(self):
        """
        Returns torque applied at the torque sensor at the robot arm's eef
        """
        return self.get_sensor_measurement(self.gripper.important_sensors["torque_ee"])

    @property
    def _hand_pose(self):
        """
        Returns:
            np.array: (4,4) array corresponding to the eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name(self.robot_model.eef_name)

    @property
    def _hand_quat(self):
        """
        Returns:
            np.array: (x,y,z,w) eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._hand_orn)

    @property
    def _hand_total_velocity(self):
        """
        Returns:
            np.array: 6-array representing the total eef velocity (linear + angular) in the base frame
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp(self.robot_model.eef_name).reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes]

        Jr = self.sim.data.get_body_jacr(self.robot_model.eef_name).reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _hand_pos(self):
        """
        Returns:
            np.array: 3-array representing the position of eef in base frame of robot.
        """
        eef_pose_in_base = self._hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _hand_orn(self):
        """
        Returns:
            np.array: (3,3) array representing the orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _hand_vel(self):
        """
        Returns:
            np.array: (x,y,z) velocity of eef in base frame of robot.
        """
        return self._hand_total_velocity[:3]

    @property
    def _hand_ang_vel(self):
        """
        Returns:
            np.array: (ax,ay,az) angular velocity of eef in base frame of robot.
        """
        return self._hand_total_velocity[3:]

    def visualize(self, vis_settings):
        """
        Do any necessary visualization for this manipulator

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "robots" and "grippers" keyword as well as any other
                robot-specific options specified.
        """
        super().visualize(vis_settings=vis_settings)
        self._visualize_grippers(visible=vis_settings["grippers"])

    def grip_action(self, gripper, gripper_action):
        """
        Executes @gripper_action for specified @gripper

        Args:
            gripper (GripperModel): Gripper to execute action for
            gripper_action (float): Value between [-1,1] to send to gripper
        """
        actuator_idxs = [self.sim.model.actuator_name2id(actuator) for actuator in gripper.actuators]
        gripper_action_actual = gripper.format_action(gripper_action)
        # rescale normalized gripper action to control ranges
        ctrl_range = self.sim.model.actuator_ctrlrange[actuator_idxs]
        bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        applied_gripper_action = bias + weight * gripper_action_actual
        self.sim.data.ctrl[actuator_idxs] = applied_gripper_action

    @property
    def dof(self):
        """
        Returns:
            int: degrees of freedom of the robot (with grippers).
        """
        # Get the dof of the base robot model
        dof = super().dof
        for gripper in self.robot_model.grippers.values():
            dof += gripper.dof
        return dof


if __name__ == '__main__':
    arm = SingleArm('UR5e', 0)
    arm.load_model()
    # sim = MjSim(arm.robot_model.get_xml())  # model is not complete, so it raises error
    # arm.reset_sim(sim)
    # arm.setup_references()
    # # arm.reset()
    # # low, high = arm.action_limits
    # # print(low, high)
    print(arm._ref_joint_pos_indexes)
    print(arm._ref_joint_actuator_indexes)