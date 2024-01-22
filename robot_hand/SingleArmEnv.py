import os
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import cv2

import robot_hand.macros as macros
from robot_hand.single_arm import SingleArm
from robot_hand.Task import Task
from robot_hand.arena import TableArena
from robot_hand.controller_factory import load_controller_config, reset_controllers
from robot_hand.MujocoXMLModel import MujocoModel
from robot_hand.gripper_model import GripperModel
import robot_hand.sim_utils as SU
from robot_hand.opencv_renderer import OpenCVRenderer

from robosuite.utils.transform_utils import mat2quat
from robosuite.renderers.base import load_renderer_config
from robosuite.utils import SimulationError
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from robosuite.utils.observables import Observable, sensor
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler

REGISTERED_ENVS = {}

ROBOT_CLASS_MAPPING = {
    'Humanoid': SingleArm,
    'UR5e': SingleArm,
}


def make(env_name, *args, **kwargs):
    """
    Instantiates a robosuite environment.
    This method attempts to mirror the equivalent functionality of gym.make in a somewhat sloppy way.
    Args:
        env_name (str): Name of the robosuite environment to initialize
        *args: Additional arguments to pass to the specific environment class initializer
        **kwargs: Additional arguments to pass to the specific environment class initializer
    Returns:
        MujocoEnv: Desired robosuite environment
    Raises:
        Exception: [Invalid environment name]
    """
    if env_name not in REGISTERED_ENVS:
        raise Exception(
            "Environment {} not found. Make sure it is a registered environment among: {}".format(
                env_name, ", ".join(REGISTERED_ENVS)
            )
        )
    return REGISTERED_ENVS[env_name](*args, **kwargs)


class EnvMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        REGISTERED_ENVS[cls.__name__] = cls
        return cls


class MujocoEnv(metaclass=EnvMeta):
    """
    Initializes a Mujoco Environment.
    Args:
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.
        has_offscreen_renderer (bool): True if using off-screen rendering.
        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse
        render_collision_mesh (bool): True if rendering collision meshes
            in camera. False otherwise.
        render_visual_mesh (bool): True if rendering visual meshes
            in camera. False otherwise.
        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).
        control_freq (float): how many control signals to receive
            in every simulated second. This sets the amount of simulation time
            that passes between every action input.
        horizon (int): Every episode lasts for exactly @horizon timesteps.
        ignore_done (bool): True if never terminating the environment (ignore @horizon).
        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables
        renderer (str): string for the renderer to use
        renderer_config (dict): dictionary for the renderer configurations
    Raises:
        ValueError: [Invalid renderer selection]
    """

    def __init__(
        self,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        renderer="mujoco",
        renderer_config=None,
    ):
        # If you're using an onscreen renderer, you must be also using an offscreen renderer!
        if has_renderer and not has_offscreen_renderer:
            has_offscreen_renderer = True

        # Rendering-specific attributes
        self.has_renderer = has_renderer
        # offscreen renderer needed for on-screen rendering
        self.has_offscreen_renderer = has_renderer or has_offscreen_renderer
        self.render_camera = render_camera
        self.render_collision_mesh = render_collision_mesh
        self.render_visual_mesh = render_visual_mesh
        self.render_gpu_device_id = render_gpu_device_id
        self.viewer = None

        # Simulation-specific attributes
        self._observables = {}  # Maps observable names to observable objects
        self._obs_cache = {}  # Maps observable names to pre-/partially-computed observable values
        self.control_freq = control_freq
        self.horizon = horizon
        self.ignore_done = ignore_done
        self.hard_reset = hard_reset
        self._xml_processor = None  # Function to process model xml in _initialize_sim() call
        self.model = None
        self.cur_time = None
        self.model_timestep = None
        self.control_timestep = None
        self.deterministic_reset = False  # Whether to add randomized resetting of objects / robot joints

        self.renderer = renderer
        self.renderer_config = renderer_config

        # Load the model
        self._load_model()

        # Initialize the simulation
        self._initialize_sim()

        # initializes the rendering
        self.initialize_renderer()

        # Run all further internal (re-)initialization required
        self._reset_internal()

        # Load observables
        if hasattr(self.viewer, "_setup_observables"):
            self._observables = self.viewer._setup_observables()
        else:
            self._observables = self._setup_observables()

        # check if viewer has get observations method and set a flag for future use.
        self.viewer_get_obs = hasattr(self.viewer, "_get_observations")

    def initialize_renderer(self):
        self.renderer = self.renderer.lower()

        if self.renderer_config is None and self.renderer != "mujoco":
            self.renderer_config = load_renderer_config(self.renderer)

        if self.renderer == "mujoco" or self.renderer == "default":
            pass
        # elif self.renderer == "nvisii":
        #     from robosuite.renderers.nvisii.nvisii_renderer import NVISIIRenderer
        #
        #     self.viewer = NVISIIRenderer(env=self, **self.renderer_config)
        else:
            raise ValueError(
                f"{self.renderer} is not a valid renderer name. Valid options include default (native mujoco renderer), and nvisii"
            )

    def initialize_time(self, control_freq):
        """
        Initializes the time constants used for simulation.
        Args:
            control_freq (float): Hz rate to run control loop at within the simulation
        """
        self.cur_time = 0
        self.model_timestep = macros.SIMULATION_TIMESTEP
        if self.model_timestep <= 0:
            raise ValueError("Invalid simulation timestep defined!")
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError("Control frequency {} is invalid".format(control_freq))
        self.control_timestep = 1.0 / control_freq

    def set_xml_processor(self, processor):
        """
        Sets the processor function that xml string will be passed to inside _initialize_sim() calls.
        Args:
            processor (None or function): If set, processing method should take in a xml string and
                return no arguments.
        """
        self._xml_processor = processor

    def _load_model(self):
        """Loads an xml model, puts it in self.model"""
        pass

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        # Setup mappings from model to IDs
        self.model.generate_id_mappings(sim=self.sim)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment.
        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        return OrderedDict()

    def _initialize_sim(self, xml_string=None):
        """
        Creates a MjSim object and stores it in self.sim. If @xml_string is specified, the MjSim object will be created
        from the specified xml_string. Else, it will pull from self.model to instantiate the simulation
        Args:
            xml_string (str): If specified, creates MjSim object from this filepath
        """
        xml = xml_string if xml_string else self.model.get_xml()

        # process the xml before initializing sim
        if self._xml_processor is not None:
            xml = self._xml_processor(xml)

        # Create the simulation instance
        self.sim = MjSim.from_xml_string(xml)   # self.sim = MjSim()

        # run a single step to make sure changes have propagated through sim state
        self.sim.forward()

        # Setup sim time based on control frequency
        self.initialize_time(self.control_freq)

    def reset(self):
        """
        Resets simulation.
        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        # TODO(yukez): investigate black screen of death
        # Use hard reset if requested

        if self.hard_reset and not self.deterministic_reset:
            if self.renderer == "mujoco" or self.renderer == "default":
                self._destroy_viewer()
                self._destroy_sim()
            self._load_model()
            self._initialize_sim()
        # Else, we only reset the sim internally
        else:
            self.sim.reset()

        # Reset necessary robosuite-centric variables
        self._reset_internal()
        self.sim.forward()
        # Setup observables, reloading if
        self._obs_cache = {}
        if self.hard_reset:
            # If we're using hard reset, must re-update sensor object references
            if hasattr(self.viewer, "_setup_observables"):
                _observables = self.viewer._setup_observables()
            else:
                _observables = self._setup_observables()
            for obs_name, obs in _observables.items():
                self.modify_observable(observable_name=obs_name, attribute="sensor", modifier=obs._sensor)
        # Make sure that all sites are toggled OFF by default
        self.visualize(vis_settings={vis: False for vis in self._visualizations})   # {"env": False, "robot": False, "gripper": False}

        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.reset()

        observations = (
            self.viewer._get_observations(force_update=True)
            if self.viewer_get_obs
            else self._get_observations(force_update=True)
        )

        # Return new observations
        return observations

    def _reset_internal(self):
        """Resets simulation internal configurations."""

        # create visualization screen or renderer
        if self.has_renderer and self.viewer is None:
            self.viewer = OpenCVRenderer(self.sim)

            # Set the camera angle for viewing
            if self.render_camera is not None:
                camera_id = self.sim.model.camera_name2id(self.render_camera)
                self.viewer.set_camera(camera_id)

        if self.has_offscreen_renderer:
            if self.sim._render_context_offscreen is None:  # self.sim._render_context_offscreen = MjRenderContext()
                render_context = MjRenderContextOffscreen(self.sim, device_id=self.render_gpu_device_id)
            self.sim._render_context_offscreen.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0   # 0 cannot see collision geom
            self.sim._render_context_offscreen.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0      # 1 can see visual geom

        # additional housekeeping
        self.sim_state_initial = self.sim.get_state()
        self._setup_references()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

        # Empty observation cache and reset all observables
        self._obs_cache = {}
        for observable in self._observables.values():
            observable.reset()

    def _update_observables(self, force=False):
        """
        Updates all observables in this environment
        Args:
            force (bool): If True, will force all the observables to update their internal values to the newest
                value. This is useful if, e.g., you want to grab observations when directly setting simulation states
                without actually stepping the simulation.
        """
        for observable in self._observables.values():
            observable.update(timestep=self.model_timestep, obs_cache=self._obs_cache, force=force)

    def _get_observations(self, force_update=False):
        """
        Grabs observations from the environment.
        Args:
            force_update (bool): If True, will force all the observables to update their internal values to the newest
                value. This is useful if, e.g., you want to grab observations when directly setting simulation states
                without actually stepping the simulation.
        Returns:
            OrderedDict: OrderedDict containing observations [(name_string, np.array), ...]
        """
        observations = OrderedDict()
        obs_by_modality = OrderedDict()

        # Force an update if requested
        if force_update:
            self._update_observables(force=True)

        # Loop through all observables and grab their current observation
        for obs_name, observable in self._observables.items():
            if observable.is_enabled() and observable.is_active():
                obs = observable.obs
                observations[obs_name] = obs
                modality = observable.modality + "-state"
                if modality not in obs_by_modality:
                    obs_by_modality[modality] = []
                # Make sure all observations are numpy arrays so we can concatenate them
                array_obs = [obs] if type(obs) in {int, float} or not obs.shape else obs
                obs_by_modality[modality].append(np.array(array_obs))

        # Add in modality observations
        for modality, obs in obs_by_modality.items():
            # To save memory, we only concatenate the image observations if explicitly requested
            if modality == "image-state" and not macros.CONCATENATE_IMAGES:
                continue
            observations[modality] = np.concatenate(obs, axis=-1)

        return observations

    def step(self, action):
        """
        Takes a step in simulation with control command @action.
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            4-tuple:
                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        Raises:
            ValueError: [Steps past episode termination]
        """
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal manipulator will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the manipulator,
        # or an actual policy update
        policy_step = True

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(int(self.control_timestep / self.model_timestep)):   # 0.05 / 0.002 = 25
            self.sim.forward()
            self._pre_action(action, policy_step)
            self.sim.step()
            self._update_observables()
            policy_step = False

        # Note: this is done all at once to avoid floating point inaccuracies
        self.cur_time += self.control_timestep

        reward, done, info = self._post_action(action)

        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.update()

        observations = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        return observations, reward, done, info

    def _pre_action(self, action, policy_step=False):
        """
        Do any preprocessing before taking an action.
        Args:
            action (np.array): Action to execute within the environment
            policy_step (bool): Whether this current loop is an actual policy step or internal sim update step
        """
        self.sim.data.ctrl[:] = action

    def _post_action(self, action):
        """
        Do any housekeeping after taking an action.
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method
        """
        reward = self.reward(action)

        # done if number of elapsed timesteps is greater than horizon
        self.done = (self.timestep >= self.horizon) and not self.ignore_done

        return reward, self.done, {}

    def reward(self, action):
        """
        Reward should be a function of state and action
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            float: Reward from environment
        """
        raise NotImplementedError

    def render(self):
        """
        Renders to an on-screen window.
        """
        self.viewer.render()

    def get_pixel_obs(self):
        """
        Gets the pixel observations for the environment from the specified renderer
        """
        self.viewer.get_pixel_obs()

    def close_renderer(self):
        """
        Closes the renderer
        """
        self.viewer.close()

    def observation_spec(self):
        """
        Returns an observation as observation specification.
        An alternative design is to return an OrderedDict where the keys
        are the observation names and the values are the shapes of observations.
        We leave this alternative implementation commented out, as we find the
        current design is easier to use in practice.
        Returns:
            OrderedDict: Observations from the environment
        """
        observation = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        return observation

    def clear_objects(self, object_names):
        """
        Clears objects with the name @object_names out of the task space. This is useful
        for supporting task modes with single types of objects, as in
        @self.single_object_mode without changing the model definition.
        Args:
            object_names (str or list of str): Name of object(s) to remove from the task workspace
        """
        object_names = {object_names} if type(object_names) is str else set(object_names)
        for obj in self.model.mujoco_objects:
            if obj.name in object_names:
                self.sim.data.set_joint_qpos(obj.joints[0], np.array((10, 10, 10, 1, 0, 0, 0)))

    def visualize(self, vis_settings):
        """
        Do any needed visualization here
        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "env" keyword as well as any other relevant
                options specified.
        """
        # Set visuals for environment objects
        for obj in self.model.mujoco_objects:
            obj.set_sites_visibility(sim=self.sim, visible=vis_settings["env"])

    def set_camera_pos_quat(self, camera_pos, camera_quat):
        if self.renderer in ["nvisii"]:
            self.viewer.set_camera_pos_quat(camera_pos, camera_quat)
        else:
            raise AttributeError("setting camera position and quat requires renderer to be NVISII.")

    # def edit_model_xml(self, xml_str):
    #     """
    #     This function edits the model xml with custom changes, including resolving relative paths,
    #     applying changes retroactively to existing demonstration files, and other custom scripts.
    #     Environment subclasses should modify this function to add environment-specific xml editing features.
    #     Args:
    #         xml_str (str): Mujoco sim demonstration XML file as string
    #     Returns:
    #         str: Edited xml file as string
    #     """
    #
    #     path = os.path.split(robosuite.__file__)[0]
    #     path_split = path.split("/")
    #
    #     # replace mesh and texture file paths
    #     tree = ET.fromstring(xml_str)
    #     root = tree
    #     asset = root.find("asset")
    #     meshes = asset.findall("mesh")
    #     textures = asset.findall("texture")
    #     all_elements = meshes + textures
    #
    #     for elem in all_elements:
    #         old_path = elem.get("file")
    #         if old_path is None:
    #             continue
    #         old_path_split = old_path.split("/")
    #         ind = max(loc for loc, val in enumerate(old_path_split) if val == "robosuite")  # last occurrence index
    #         new_path_split = path_split + old_path_split[ind + 1 :]
    #         new_path = "/".join(new_path_split)
    #         elem.set("file", new_path)
    #
    #     return ET.tostring(root, encoding="utf8").decode("utf8")

    def reset_from_xml_string(self, xml_string):
        """
        Reloads the environment from an XML description of the environment.
        Args:
            xml_string (str): Filepath to the xml file that will be loaded directly into the sim
        """

        # if there is an active viewer window, destroy it
        if self.renderer != "nvisii":
            self.close()

        # Since we are reloading from an xml_string, we are deterministically resetting
        self.deterministic_reset = True

        # initialize sim from xml
        self._initialize_sim(xml_string=xml_string)

        # Now reset as normal
        self.reset()

        # Turn off deterministic reset
        self.deterministic_reset = False

    def check_contact(self, geoms_1, geoms_2=None):
        """
        Finds contact between two geom groups.
        Args:
            geoms_1 (str or list of str or MujocoModel): an individual geom name or list of geom names or a model. If
                a MujocoModel is specified, the geoms checked will be its contact_geoms
            geoms_2 (str or list of str or MujocoModel or None): another individual geom name or list of geom names.
                If a MujocoModel is specified, the geoms checked will be its contact_geoms. If None, will check
                any collision with @geoms_1 to any other geom in the environment
        Returns:
            bool: True if any geom in @geoms_1 is in contact with any geom in @geoms_2.
        """
        return SU.check_contact(sim=self.sim, geoms_1=geoms_1, geoms_2=geoms_2)

    def get_contacts(self, model):
        """
        Checks for any contacts with @model (as defined by @model's contact_geoms) and returns the set of
        geom names currently in contact with that model (excluding the geoms that are part of the model itself).
        Args:
            model (MujocoModel): Model to check contacts for.
        Returns:
            set: Unique geoms that are actively in contact with this model.
        Raises:
            AssertionError: [Invalid input type]
        """
        return SU.get_contacts(sim=self.sim, model=model)

    def add_observable(self, observable):
        """
        Adds an observable to this environment.
        Args:
            observable (Observable): Observable instance.
        """
        assert observable.name not in self._observables, (
            "Observable name {} is already associated with an existing observable! Use modify_observable(...) "
            "to modify a pre-existing observable.".format(observable.name)
        )
        self._observables[observable.name] = observable

    def modify_observable(self, observable_name, attribute, modifier):
        """
        Modifies observable with associated name @observable_name, replacing the given @attribute with @modifier.
        Args:
             observable_name (str): Observable to modify
             attribute (str): Observable attribute to modify.
                Options are {`'sensor'`, `'corrupter'`,`'filter'`,  `'delayer'`, `'sampling_rate'`,
                `'enabled'`, `'active'`}
             modifier (any): New function / value to replace with for observable. If a function, new signature should
                match the function being replaced.
        """
        # Find the observable
        assert observable_name in self._observables, "No valid observable with name {} found. Options are: {}".format(
            observable_name, self.observation_names
        )
        obs = self._observables[observable_name]
        # replace attribute accordingly
        if attribute == "sensor":
            obs.set_sensor(modifier)
        elif attribute == "corrupter":
            obs.set_corrupter(modifier)
        elif attribute == "filter":
            obs.set_filter(modifier)
        elif attribute == "delayer":
            obs.set_delayer(modifier)
        elif attribute == "sampling_rate":
            obs.set_sampling_rate(modifier)
        elif attribute == "enabled":
            obs.set_enabled(modifier)
        elif attribute == "active":
            obs.set_active(modifier)
        else:
            # Invalid attribute specified
            raise ValueError(
                "Invalid observable attribute specified. Requested: {}, valid options are {}".format(
                    attribute, {"sensor", "corrupter", "filter", "delayer", "sampling_rate", "enabled", "active"}
                )
            )

    def _check_success(self):
        """
        Checks if the task has been completed. Should be implemented by subclasses
        Returns:
            bool: True if the task has been completed
        """
        raise NotImplementedError

    def _destroy_viewer(self):
        """
        Destroys the current mujoco renderer instance if it exists
        """
        # if there is an active viewer window, destroy it
        if self.viewer is not None:
            self.viewer.close()  # change this to viewer.finish()?
            self.viewer = None

    def _destroy_sim(self):
        """
        Destroys the current MjSim instance if it exists
        """
        if self.sim is not None:
            self.sim.free()
            self.sim = None

    def close(self):
        """Do any cleanup necessary here."""
        self._destroy_viewer()
        self._destroy_sim()

    @property
    def observation_modalities(self):
        """
        Modalities for this environment's observations
        Returns:
            set: All observation modalities
        """
        return set([observable.modality for observable in self._observables.values()])

    @property
    def observation_names(self):
        """
        Grabs all names for this environment's observables
        Returns:
            set: All observation names
        """
        return set(self._observables.keys())

    @property
    def enabled_observables(self):
        """
        Grabs all names of enabled observables for this environment. An observable is considered enabled if its values
        are being continually computed / updated at each simulation timestep.
        Returns:
            set: All enabled observation names
        """
        return set([name for name, observable in self._observables.items() if observable.is_enabled()])

    @property
    def active_observables(self):
        """
        Grabs all names of active observables for this environment. An observable is considered active if its value is
        being returned in the observation dict from _get_observations() call or from the step() call (assuming this
        observable is enabled).
        Returns:
            set: All active observation names
        """
        return set([name for name, observable in self._observables.items() if observable.is_active()])

    @property
    def _visualizations(self):
        """
        Visualization keywords for this environment
        Returns:
            set: All components that can be individually visualized for this environment
        """
        return {"env"}

    @property
    def action_spec(self):
        """
        Action specification should be implemented in subclasses.
        Action space is represented by a tuple of (low, high), which are two numpy
        vectors that specify the min/max action limits per dimension.
        """
        raise NotImplementedError

    @property
    def action_dim(self):
        """
        Size of the action space
        Returns:
            int: Action space dimension
        """
        raise NotImplementedError


class RobotEnv(MujocoEnv):
    """
    Initializes a robot environment in Mujoco.

    Args:
        robots: Specification for specific robot(s) to be instantiated within this env

        env_configuration (str): Specifies how to position the robot(s) within the environment. Default is "default",
            which should be interpreted accordingly by any subclasses.

        controller_configs (str or list of dict): If set, contains relevant manipulator parameters for creating a
            custom manipulator. Else, uses the default manipulator for this specific task. Should either be single
            dict if same manipulator is to be used for all robots or else it should be a list of the same length as
            "robots" param

        mount_types (None or str or list of str): type of mount, used to instantiate mount models from mount factory.
            Default is "default", which is the default mount associated with the robot(s) the 'robots' specification.
            None results in no mount, and any other (valid) model overrides the default mount. Should either be
            single str if same mount type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

        robot_configs (list of dict): Per-robot configurations set from any subclass initializers.

    Raises:
        ValueError: [Camera obs require offscreen renderer]
        ValueError: [Camera name must be specified to use camera obs]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        mount_types="default",
        controller_configs=None,
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,    # False: you cannot see the collision mesh
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        robot_configs=None,
        renderer="mujoco",
        renderer_config=None,
    ):
        # First, verify that correct number of robots are being inputted
        self.env_configuration = env_configuration
        self._check_robot_configuration(robots)

        # Robot
        robots = list(robots) if type(robots) is list or type(robots) is tuple else [robots]    # ['UR5e']
        self.num_robots = len(robots)
        self.robot_names = robots
        self.robots = self._input2list(None, self.num_robots)
        self._action_dim = None

        # Mount
        mount_types = self._input2list(mount_types, self.num_robots)    # ['default']

        # Controller
        controller_configs = self._input2list(controller_configs, self.num_robots)

        # Initialization Noise
        initialization_noise = self._input2list(initialization_noise, self.num_robots)

        # Observations -- Ground truth = object_obs, Image data = camera_obs
        self.use_camera_obs = use_camera_obs

        # Camera / Rendering Settings
        self.has_offscreen_renderer = has_offscreen_renderer
        self.camera_names = (
            list(camera_names) if type(camera_names) is list or type(camera_names) is tuple else [camera_names]
        )
        self.num_cameras = len(self.camera_names)

        self.camera_heights = self._input2list(camera_heights, self.num_cameras)
        self.camera_widths = self._input2list(camera_widths, self.num_cameras)
        self.camera_depths = self._input2list(camera_depths, self.num_cameras)
        self.camera_segmentations = self._input2list(camera_segmentations, self.num_cameras)
        # We need to parse camera segmentations more carefully since it may be a nested list
        seg_is_nested = False
        for i, camera_s in enumerate(self.camera_segmentations):
            if isinstance(camera_s, list) or isinstance(camera_s, tuple):
                seg_is_nested = True
                break
        camera_segs = deepcopy(self.camera_segmentations)
        for i, camera_s in enumerate(self.camera_segmentations):
            if camera_s is not None:
                self.camera_segmentations[i] = self._input2list(camera_s, 1) if seg_is_nested else deepcopy(camera_segs)

        # sanity checks for camera rendering
        if self.use_camera_obs and not self.has_offscreen_renderer:
            raise ValueError("Error: Camera observations require an offscreen renderer!")
        if self.use_camera_obs and self.camera_names is None:
            raise ValueError("Must specify at least one camera name when using camera obs")

        # Robot configurations -- update from subclass configs
        if robot_configs is None:
            robot_configs = [{} for _ in range(self.num_robots)]
        self.robot_configs = [
            dict(
                **{
                    "controller_config": controller_configs[idx],
                    "mount_type": mount_types[idx],
                    "initialization_noise": initialization_noise[idx],
                    "control_freq": control_freq,
                },
                **robot_config,
            )
            for idx, robot_config in enumerate(robot_configs)
        ]

        # Run superclass init
        super().__init__(
            has_renderer=has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def visualize(self, vis_settings):
        """
        In addition to super call, visualizes robots.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "robots" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)
        # Loop over robots to visualize them independently
        for robot in self.robots:
            robot.visualize(vis_settings=vis_settings)

    @property
    def _visualizations(self):
        """
        Visualization keywords for this environment

        Returns:
            set: All components that can be individually visualized for this environment
        """
        vis_set = super()._visualizations
        vis_set.add("robots")
        return vis_set

    @property
    def action_spec(self):
        """
        Action space (low, high) for this environment

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        low, high = [], []
        for robot in self.robots:
            lo, hi = robot.action_limits
            low, high = np.concatenate([low, lo]), np.concatenate([high, hi])
        return low, high

    @property
    def action_dim(self):
        """
        Size of the action space

        Returns:
            int: Action space dimension
        """
        return self._action_dim

    @staticmethod
    def _input2list(inp, length):
        """
        Helper function that converts an input that is either a single value or a list into a list

        Args:
            inp (None or str or list): Input value to be converted to list
            length (int): Length of list to broadcast input to

        Returns:
            list: input @inp converted into a list of length @length
        """
        # convert to list if necessary
        return list(inp) if type(inp) is list or type(inp) is tuple else [inp for _ in range(length)]

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Load robots
        self._load_robots()

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Setup robot-specific references as well (note: requires resetting of sim for robot first)
        for robot in self.robots:
            robot.reset_sim(self.sim)
            robot.setup_references()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Loops through all robots and grabs their corresponding
        observables to add to the procedurally generated dict of observables

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        # Loop through all robots and grab their observables, adding it to the proprioception modality
        for robot in self.robots:
            robot_obs = robot.setup_observables()
            observables.update(robot_obs)

        # Loop through cameras and update the observations if using camera obs
        if self.use_camera_obs:
            # Create sensor information
            sensors = []
            names = []
            for (cam_name, cam_w, cam_h, cam_d, cam_segs) in zip(
                self.camera_names,
                self.camera_widths,
                self.camera_heights,
                self.camera_depths,
                self.camera_segmentations,
            ):

                # Add cameras associated to our arrays
                cam_sensors, cam_sensor_names = self._create_camera_sensors(
                    cam_name, cam_w=cam_w, cam_h=cam_h, cam_d=cam_d, cam_segs=cam_segs, modality="image"
                )
                sensors += cam_sensors
                names += cam_sensor_names

            # If any the camera segmentations are not None, then we shrink all the sites as a hacky way to
            # prevent them from being rendered in the segmentation mask
            if not all(seg is None for seg in self.camera_segmentations):
                self.sim.model.site_size[:, :] = 1.0e-8

            # Create observables for these cameras
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _create_camera_sensors(self, cam_name, cam_w, cam_h, cam_d, cam_segs, modality="image"):
        """
        Helper function to create sensors for a given camera. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.
        Args:
            cam_name (str): Name of camera to create sensors for
            cam_w (int): Width of camera
            cam_h (int): Height of camera
            cam_d (bool): Whether to create a depth sensor as well
            cam_segs (None or list): Type of segmentation(s) to use, where each entry can be the following:
                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            modality (str): Modality to assign to all sensors
        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given camera
                names (list): array of corresponding observable names
        """
        # Make sure we get correct convention
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        # Create sensor information
        sensors = []
        names = []

        # Add camera observables to the dict
        rgb_sensor_name = f"{cam_name}_image"
        depth_sensor_name = f"{cam_name}_depth"
        segmentation_sensor_name = f"{cam_name}_segmentation"

        @sensor(modality=modality)
        def camera_rgb(obs_cache):
            img = self.sim.render(
                camera_name=cam_name,
                width=cam_w,
                height=cam_h,
                depth=cam_d,
            )
            if cam_d:
                rgb, depth = img
                obs_cache[depth_sensor_name] = np.expand_dims(depth[::convention], axis=-1)
                return rgb[::convention]
            else:
                return img[::convention]

        sensors.append(camera_rgb)
        names.append(rgb_sensor_name)

        if cam_d:

            @sensor(modality=modality)
            def camera_depth(obs_cache):
                return obs_cache[depth_sensor_name] if depth_sensor_name in obs_cache else np.zeros((cam_h, cam_w, 1))

            sensors.append(camera_depth)
            names.append(depth_sensor_name)

        if cam_segs is not None:
            # Define mapping we'll use for segmentation
            for cam_s in cam_segs:
                seg_sensor, seg_sensor_name = self._create_segementation_sensor(
                    cam_name=cam_name,
                    cam_w=cam_w,
                    cam_h=cam_h,
                    cam_s=cam_s,
                    seg_name_root=segmentation_sensor_name,
                    modality=modality,
                )

                sensors.append(seg_sensor)
                names.append(seg_sensor_name)

        return sensors, names

    def _create_segementation_sensor(self, cam_name, cam_w, cam_h, cam_s, seg_name_root, modality="image"):
        """
        Helper function to create sensors for a given camera. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            cam_name (str): Name of camera to create sensors for
            cam_w (int): Width of camera
            cam_h (int): Height of camera
            cam_s (None or list): Type of segmentation to use, should be the following:
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level
            seg_name_root (str): Sensor name root to assign to this sensor

            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                camera_segmentation (function): Generated sensor function for this segmentation sensor
                name (str): Corresponding sensor name
        """
        # Make sure we get correct convention
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        if cam_s == "instance":
            name2id = {inst: i for i, inst in enumerate(list(self.model.instances_to_ids.keys()))}
            mapping = {idn: name2id[inst] for idn, inst in self.model.geom_ids_to_instances.items()}
        elif cam_s == "class":
            name2id = {cls: i for i, cls in enumerate(list(self.model.classes_to_ids.keys()))}
            mapping = {idn: name2id[cls] for idn, cls in self.model.geom_ids_to_classes.items()}
        else:  # element
            # No additional mapping needed
            mapping = None

        @sensor(modality=modality)
        def camera_segmentation(obs_cache):
            seg = self.sim.render(
                camera_name=cam_name,
                width=cam_w,
                height=cam_h,
                depth=False,
                segmentation=True,
            )
            seg = np.expand_dims(seg[::convention, :, 1], axis=-1)
            # Map raw IDs to grouped IDs if we're using instance or class-level segmentation
            if mapping is not None:
                seg = (
                    np.fromiter(map(lambda x: mapping.get(x, -1), seg.flatten()), dtype=np.int32).reshape(
                        cam_h, cam_w, 1
                    )
                    + 1
                )
            return seg

        name = f"{seg_name_root}_{cam_s}"

        return camera_segmentation, name

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # Run superclass reset functionality
        super()._reset_internal()

        # Reset controllers
        reset_controllers()

        # Reset action dim
        self._action_dim = 0

        # Reset robot and update action space dimension along the way
        for robot in self.robots:
            robot.reset(deterministic=self.deterministic_reset)
            self._action_dim += robot.action_dim

        # Update cameras if appropriate
        if self.use_camera_obs:
            temp_names = []
            for cam_name in self.camera_names:
                if "all-" in cam_name:
                    # We need to add all robot-specific camera names that include the key after the tag "all-"
                    start_idx = len(temp_names) - 1
                    key = cam_name.replace("all-", "")
                    for robot in self.robots:
                        for robot_cam_name in robot.robot_model.cameras:
                            if key in robot_cam_name:
                                temp_names.append(robot_cam_name)
                    # We also need to broadcast the corresponding values from each camera dimensions as well
                    end_idx = len(temp_names) - 1
                    self.camera_widths = (
                        self.camera_widths[:start_idx]
                        + [self.camera_widths[start_idx]] * (end_idx - start_idx)
                        + self.camera_widths[(start_idx + 1) :]
                    )
                    self.camera_heights = (
                        self.camera_heights[:start_idx]
                        + [self.camera_heights[start_idx]] * (end_idx - start_idx)
                        + self.camera_heights[(start_idx + 1) :]
                    )
                    self.camera_depths = (
                        self.camera_depths[:start_idx]
                        + [self.camera_depths[start_idx]] * (end_idx - start_idx)
                        + self.camera_depths[(start_idx + 1) :]
                    )
                else:
                    # We simply add this camera to the temp_names
                    temp_names.append(cam_name)
            # Lastly, replace camera names with the updated ones
            self.camera_names = temp_names

    def _pre_action(self, action, policy_step=False):
        """
        Overrides the superclass method to control the robot(s) within this environment using their respective
        controllers using the passed actions and gripper control.

        Args:
            action (np.array): The control to apply to the robot(s). Note that this should be a flat 1D array that
                encompasses all actions to be distributed to each robot if there are multiple. For each section of the
                action space assigned to a single robot, the first @self.robots[i].manipulator.control_dim dimensions
                should be the desired manipulator actions and if the robot has a gripper, the next
                @self.robots[i].gripper.dof dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken

        Raises:
            AssertionError: [Invalid action dimension]
        """
        # Verify that the action is the correct dimension
        assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )

        # Update robot joints based on manipulator actions
        cutoff = 0
        for idx, robot in enumerate(self.robots):
            robot_action = action[cutoff : cutoff + robot.action_dim]
            robot.control(robot_action, policy_step=policy_step)
            cutoff += robot.action_dim

    def _load_robots(self):
        """
        Instantiates robots and stores them within the self.robots attribute
        """
        # Loop through robots and instantiate Robot object for each
        for idx, (name, config) in enumerate(zip(self.robot_names, self.robot_configs)):
            # Create the robot instance
            self.robots[idx] = ROBOT_CLASS_MAPPING[name](robot_type=name, idn=idx, **config)
            # Now, load the robot models
            self.robots[idx].load_model()

    def reward(self, action):
        """
        Runs superclass method by default
        """
        return super().reward(action)

    def _check_success(self):
        """
        Runs superclass method by default
        """
        return super()._check_success()

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure inputted robots and the corresponding requested task/configuration combo is legal.
        Should be implemented in every specific task module

        Args:
            robots (str or list of str): Inputted requested robots at the task-level environment
        """
        raise NotImplementedError


class SingleArmEnv(RobotEnv):
    """
    A manipulation environment intended for a single robot arm.
    """

    def _load_model(self):
        """
        Verifies correct robot model is loaded
        """
        super()._load_model()

        # Verify the correct robot has been loaded
        assert isinstance(
            self.robots[0], SingleArm
        ), "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        # super()._check_robot_configuration(robots)
        # Make sure all inputted robots are a manipulation robot
        if type(robots) is str:
            robots = [robots]
        for robot in robots:
            assert issubclass(
                ROBOT_CLASS_MAPPING[robot], SingleArm
            ), "Only manipulator robots supported for manipulation environment!"

        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"

    @property
    def _eef_xpos(self):
        """
        Grabs End Effector position

        Returns:
            np.array: End effector(x,y,z)
        """
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])

    @property
    def _eef_xmat(self):
        """
        End Effector orientation as a rotation matrix
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (3,3) End Effector orientation matrix
        """
        pf = self.robots[0].gripper.naming_prefix

        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "right_grip_site")]).reshape(3, 3)
        else:
            return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "grip_site")]).reshape(3, 3)

    @property
    def _eef_xquat(self):
        """
        End Effector orientation as a (x,y,z,w) quaternion
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (x,y,z,w) End Effector quaternion
        """
        return mat2quat(self._eef_xmat)

    @property
    def _visualizations(self):
        """
        Visualization keywords for this environment

        Returns:
            set: All components that can be individually visualized for this environment
        """
        vis_set = super()._visualizations
        vis_set.add("grippers")
        return vis_set

    #TODO: ManipulatorEnv
    def _check_grasp(self, gripper, object_geoms):
        """
        Checks whether the specified gripper as defined by @gripper is grasping the specified object in the environment.

        By default, this will return True if at least one geom in both the "left_fingerpad" and "right_fingerpad" geom
        groups are in contact with any geom specified by @object_geoms. Custom gripper geom groups can be
        specified with @gripper as well.

        Args:
            gripper (GripperModel or str or list of str or list of list of str): If a MujocoModel, this is specific
            gripper to check for grasping (as defined by "left_fingerpad" and "right_fingerpad" geom groups). Otherwise,
                this sets custom gripper geom groups which together define a grasp. This can be a string
                (one group of single gripper geom), a list of string (multiple groups of single gripper geoms) or a
                list of list of string (multiple groups of multiple gripper geoms). At least one geom from each group
                must be in contact with any geom in @object_geoms for this method to return True.
            object_geoms (str or list of str or MujocoModel): If a MujocoModel is inputted, will check for any
                collisions with the model's contact_geoms. Otherwise, this should be specific geom name(s) composing
                the object to check for contact.

        Returns:
            bool: True if the gripper is grasping the given object
        """
        # Convert object, gripper geoms into standardized form
        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            # Parse each element in the gripper_geoms list accordingly
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            if not self.check_contact(g_group, o_geoms):
                return False
        return True

    def _gripper_to_target(self, gripper, target, target_type="body", return_distance=False):
        """
        Calculates the (x,y,z) Cartesian distance (target_pos - gripper_pos) from the specified @gripper to the
        specified @target. If @return_distance is set, will return the Euclidean (scalar) distance instead.

        Args:
            gripper (MujocoModel): Gripper model to update grip site rgb
            target (MujocoModel or str): Either a site / geom / body name, or a model that serves as the target.
                If a model is given, then the root body will be used as the target.
            target_type (str): One of {"body", "geom", or "site"}, corresponding to the type of element @target
                refers to.
            return_distance (bool): If set, will return Euclidean distance instead of Cartesian distance

        Returns:
            np.array or float: (Cartesian or Euclidean) distance from gripper to target
        """
        # Get gripper and target positions
        gripper_pos = self.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        # If target is MujocoModel, grab the correct body as the target and find the target position
        if isinstance(target, MujocoModel):
            target_pos = self.sim.data.get_body_xpos(target.root_body)
        elif target_type == "body":
            target_pos = self.sim.data.get_body_xpos(target)
        elif target_type == "site":
            target_pos = self.sim.data.get_site_xpos(target)
        else:
            target_pos = self.sim.data.get_geom_xpos(target)
        # Calculate distance
        diff = target_pos - gripper_pos
        # Return appropriate value
        return np.linalg.norm(diff) if return_distance else diff

    def _visualize_gripper_to_target(self, gripper, target, target_type="body"):
        """
        Colors the grip visualization site proportional to the Euclidean distance to the specified @target.
        Colors go from red --> green as the gripper gets closer.

        Args:
            gripper (MujocoModel): Gripper model to update grip site rgb
            target (MujocoModel or str): Either a site / geom / body name, or a model that serves as the target.
                If a model is given, then the root body will be used as the target.
            target_type (str): One of {"body", "geom", or "site"}, corresponding to the type of element @target
                refers to.
        """
        # Get gripper and target positions
        gripper_pos = self.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        # If target is MujocoModel, grab the correct body as the target and find the target position
        if isinstance(target, MujocoModel):
            target_pos = self.sim.data.get_body_xpos(target.root_body)
        elif target_type == "body":
            target_pos = self.sim.data.get_body_xpos(target)
        elif target_type == "site":
            target_pos = self.sim.data.get_site_xpos(target)
        else:
            target_pos = self.sim.data.get_geom_xpos(target)
        # color the gripper site appropriately based on (squared) distance to target
        dist = np.sum(np.square((target_pos - gripper_pos)))
        max_dist = 0.1
        scaled = (1.0 - min(dist / max_dist, 1.0)) ** 15
        rgba = np.zeros(3)
        rgba[0] = 1 - scaled
        rgba[1] = scaled
        self.sim.model.site_rgba[self.sim.model.site_name2id(gripper.important_sites["grip_site"])][:3] = rgba


class Lift(SingleArmEnv):

    def __init__(self, robots,
                 env_configuration=None,
                 controller_configs=None,
                 has_renderer=False,
                 has_offscreen_renderer=True, ):
        # TODO: settings for table top
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction = (1.0, 5e-3, 1e-4)
        self.table_offset = np.array((0, 0, 0.8))
        # reward configuration
        self.reward_scale = 1.0
        self.reward_shaping = False

        # whether to use ground-truth object states
        self.use_object_obs = True

        # object placement initializer
        self.placement_initializer = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table-top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        # tex_attrib = {
        #     "type": "cube",
        # }
        # mat_attrib = {
        #     "texrepeat": "1 1",
        #     "specular": "0.4",
        #     "shininess": "0.1",
        # }
        # redwood = CustomMaterial(
        #     texture="WoodRed",
        #     tex_name="redwood",
        #     mat_name="redwood_mat",
        #     tex_attrib=tex_attrib,
        #     mat_attrib=mat_attrib,
        # )
        # self.cube = BoxObject(
        #     name="cube",
        #     size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
        #     size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        # )

        # Create placement initializer
        # if self.placement_initializer is not None:
        #     self.placement_initializer.reset()
        #     self.placement_initializer.add_objects(self.cube)
        # else:
        #     self.placement_initializer = UniformRandomSampler(
        #         name="ObjectSampler",
        #         mujoco_objects=self.cube,
        #         x_range=[-0.03, 0.03],
        #         y_range=[-0.03, 0.03],
        #         rotation=None,
        #         ensure_object_boundary_in_range=False,
        #         ensure_valid_placement=True,
        #         reference_pos=self.table_offset,
        #         z_offset=0.01,
        #     )

        # task includes arena, robot, and objects of interest
        self.model = Task(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=None,
            # mujoco_objects=self.cube,
        )
        # xml = self.model.get_xml()

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        # if vis_settings["grippers"]:
        #     self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        # if self._check_success():
        #     reward = 2.25
        #
        # # use a shaping reward
        # elif self.reward_shaping:
        #
        #     # reaching reward
        #     cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        #     gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        #     dist = np.linalg.norm(gripper_site_pos - cube_pos)
        #     reaching_reward = 1 - np.tanh(10.0 * dist)
        #     reward += reaching_reward
        #
        #     # grasping reward
        #     if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
        #         reward += 0.25
        #
        # # Scale reward if requested
        # if self.reward_scale is not None:
        #     reward *= self.reward_scale / 2.25

        return reward


if __name__ == '__main__':
    print(REGISTERED_ENVS)
    options = {}
    options["env_name"] = 'Lift'
    # options["robots"] = 'UR5e'
    options["robots"] = 'Humanoid'
    controller_name = 'OSC_POSE'
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    env = make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=True,
    )

    # print(type(env))
    # print(env.robots[0].robot_joints)
    # print(env.robots[0]._ref_joint_pos_indexes)
    # env.robots[0].reset()
    # print(env.robots[0].action_dim)
    # print(env.action_spec)
    env.reset()
    print(env.sim.model.camera_names)
    print(env.viewer.camera_name)
    print(env.robots[0].controller_config["type"])
    print(env.robots[0].eef_site_id)
    env.viewer.set_camera(0)

    # for obj in env.model.mujoco_robots:
    #     print(type(obj))

    str = env.sim.model.get_xml()
    im = env.sim.render(camera_name=env.viewer.camera_name, height=env.viewer.height, width=env.viewer.width)[..., ::-1]

    # write frame to window
    # im = np.flip(im, axis=0)
    # cv2.imshow("offscreen render", im)
    # cv2.waitKey(0)

    low, high = env.action_spec
    start_time = time.time()

    # import mujoco.viewer
    # m = env.sim.model._model
    # d = mujoco.MjData(m)
    # print(env.sim.model.body_names)
    # print(env.sim.model.joint_names)
    # print(env.sim.model.actuator_names)
    # print(env.sim.model.camera_names)
    # for joint_name in env.sim.model.joint_names:
    #     # print(env.sim.model.get_joint_qpos_addr(joint_name))
    #     print(env.sim.data.get_joint_qpos(joint_name))
    # print(env.sim.data.get_body_xpos('base'))
    # with mujoco.viewer.launch_passive(m, d) as viewer:
    #     # Close the viewer automatically after 30 wall-seconds.
    #     start = time.time()
    #     while viewer.is_running() and time.time() - start < 10:
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

    # print(env.robots[0]._joint_positions)

    action = np.zeros(18)
    while time.time() - start_time < 3:
        if 1 < time.time() - start_time:
            action[2] = 0.4
            action[6:] = 1
        else:
            action[2] = 0
            action[6:] = -1
        obs, reward, done, _ = env.step(action)
        env.render()
    # action = np.zeros(18)
    # action[3] = 0.4
    # for i in range(400):
    #     if i > 200:
    #         action[3] = -0.4
    #     obs, reward, done, _ = env.step(action)
    #     env.render()
    env.close()
