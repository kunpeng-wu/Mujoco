import numpy as np

from robot_hand.MujocoXMLModel import MujocoXML
import robot_hand.macros as macros
from robosuite.utils.mjcf_utils import (
    ENVIRONMENT_COLLISION_COLOR,
    array_to_string,
    find_elements,
    new_body,
    new_element,
    new_geom,
    new_joint,
    recolor_collision_geoms,
    string_to_array,
    xml_path_completion,
    convert_to_string
)




class MujocoWorldBase(MujocoXML):
    """Base class to inherit all mujoco worlds from."""

    def __init__(self):
        super().__init__(xml_path_completion("base.xml"))
        # Modify the simulation timestep to be the requested value
        options = find_elements(root=self.root, tags="option", attribs=None, return_first=True)
        options.set("timestep", convert_to_string(macros.SIMULATION_TIMESTEP))


class Arena(MujocoXML):
    """Base arena class."""

    def __init__(self, fname):
        super().__init__(fname)
        # Get references to floor and bottom
        self.bottom_pos = np.zeros(3)
        self.floor = self.worldbody.find("./geom[@name='floor']")

        # Run any necessary post-processing on the model
        self._postprocess_arena()

        # Recolor all geoms
        recolor_collision_geoms(
            root=self.worldbody,
            rgba=ENVIRONMENT_COLLISION_COLOR,
            exclude=lambda e: True if e.get("name", None) == "floor" else False,
        )

    def set_origin(self, offset):
        """
        Applies a constant offset to all objects.

        Args:
            offset (3-tuple): (x,y,z) offset to apply to all nodes in this XML
        """
        offset = np.array(offset)
        for node in self.worldbody.findall("./*[@pos]"):
            cur_pos = string_to_array(node.get("pos"))
            new_pos = cur_pos + offset
            node.set("pos", array_to_string(new_pos))

    def set_camera(self, camera_name, pos, quat, camera_attribs=None):
        """
        Sets a camera with @camera_name. If the camera already exists, then this overwrites its pos and quat values.

        Args:
            camera_name (str): Camera name to search for / create
            pos (3-array): (x,y,z) coordinates of camera in world frame
            quat (4-array): (w,x,y,z) quaternion of camera in world frame
            camera_attribs (dict): If specified, should be additional keyword-mapped attributes for this camera.
                See http://www.mujoco.org/book/XMLreference.html#camera for exact attribute specifications.
        """
        # Determine if camera already exists
        camera = find_elements(root=self.worldbody, tags="camera", attribs={"name": camera_name}, return_first=True)

        # Compose attributes
        if camera_attribs is None:
            camera_attribs = {}
        camera_attribs["pos"] = array_to_string(pos)
        camera_attribs["quat"] = array_to_string(quat)

        if camera is None:
            # If camera doesn't exist, then add a new camera with the specified attributes
            self.worldbody.append(new_element(tag="camera", name=camera_name, **camera_attribs))
        else:
            # Otherwise, we edit all specified attributes in that camera
            for attrib, value in camera_attribs.items():
                camera.set(attrib, value)

    def _postprocess_arena(self):
        """
        Runs any necessary post-processing on the imported Arena model
        """
        pass


class TableArena(Arena):
    """
    Workspace that contains an empty table.


    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        has_legs (bool): whether the table has legs or not
        xml (str): xml file to load arena
    """

    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        has_legs=True,
        xml="arenas/table_arena.xml",
    ):
        super().__init__(xml_path_completion(xml))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction
        self.table_offset = table_offset
        self.center_pos = self.bottom_pos + np.array([0, 0, -self.table_half_size[2]]) + self.table_offset

        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")

        self.has_legs = has_legs
        self.table_legs_visual = [
            self.table_body.find("./geom[@name='table_leg1_visual']"),
            self.table_body.find("./geom[@name='table_leg2_visual']"),
            self.table_body.find("./geom[@name='table_leg3_visual']"),
            self.table_body.find("./geom[@name='table_leg4_visual']"),
        ]

        self.configure_location()

    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))

        self.table_top.set("pos", array_to_string(np.array([0, 0, self.table_half_size[2]])))

        # If we're not using legs, set their size to 0
        if not self.has_legs:
            for leg in self.table_legs_visual:
                leg.set("rgba", array_to_string([1, 0, 0, 0]))
                leg.set("size", array_to_string([0.0001, 0.0001]))
        else:
            # Otherwise, set leg locations appropriately
            delta_x = [0.1, -0.1, -0.1, 0.1]
            delta_y = [0.1, 0.1, -0.1, -0.1]
            for leg, dx, dy in zip(self.table_legs_visual, delta_x, delta_y):
                # If x-length of table is less than a certain length, place leg in the middle between ends
                # Otherwise we place it near the edge
                x = 0
                if self.table_half_size[0] > abs(dx * 2.0):
                    x += np.sign(dx) * self.table_half_size[0] - dx
                # Repeat the same process for y
                y = 0
                if self.table_half_size[1] > abs(dy * 2.0):
                    y += np.sign(dy) * self.table_half_size[1] - dy
                # Get z value
                z = (self.table_offset[2] - self.table_half_size[2]) / 2.0
                # Set leg position
                leg.set("pos", array_to_string([x, y, -z]))
                # Set leg size
                leg.set("size", array_to_string([0.025, z]))

    @property
    def table_top_abs(self):
        """
        Grabs the absolute position of table top

        Returns:
            np.array: (x,y,z) table position
        """
        return string_to_array(self.floor.get("pos")) + self.table_offset