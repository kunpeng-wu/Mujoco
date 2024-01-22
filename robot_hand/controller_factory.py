import json
import os
from copy import deepcopy
from robot_hand.linear_interpolator import LinearInterpolator
from robot_hand.osc import OperationalSpaceController


# Global var for linking pybullet server to multiple ik manipulator instances if necessary
pybullet_server = None

def reset_controllers():
    """
    Global function for doing one-time clears and restarting of any global manipulator-related
    specifics before re-initializing each individual manipulator again
    """
    global pybullet_server
    # Disconnect and reconnect to pybullet server if it exists
    if pybullet_server is not None:
        pybullet_server.disconnect()
        pybullet_server.connect()


def get_pybullet_server():
    """
    Getter to return reference to pybullet server module variable

    Returns:
        PyBulletServer: Server instance running PyBullet
    """
    global pybullet_server
    return pybullet_server

def controller_factory(name, params):
    interpolator = None
    if params["interpolation"] == "linear":
        interpolator = LinearInterpolator(
            ndim=params["ndim"],
            controller_freq=(1 / params["sim"].model.opt.timestep),
            policy_freq=params["policy_freq"],
            ramp_ratio=params["ramp_ratio"],
        )
    if name == "OSC_POSE":
        ori_interpolator = None
        if interpolator is not None:
            interpolator.set_states(dim=3)  # EE control uses dim 3 for pos and ori each
            ori_interpolator = deepcopy(interpolator)
            ori_interpolator.set_states(ori="euler")
        params["control_ori"] = True
        return OperationalSpaceController(interpolator_pos=interpolator, interpolator_ori=ori_interpolator, **params)


def load_controller_config(custom_fpath=None, default_controller=None):
    if custom_fpath is None:
        custom_fpath = os.path.join(
            os.path.dirname(__file__), "{}.json".format(default_controller.lower())
        )
    try:
        with open(custom_fpath) as f:
            controller_config = json.load(f)
    except FileNotFoundError:
        print("Error opening manipulator filepath at: {}. " "Please check filepath and try again.".format(custom_fpath))

    return controller_config
