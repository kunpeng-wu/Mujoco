import mujoco as mj
from mujoco.glfw import glfw


class Controller:
    def __init__(self, signs):
        self._signs = signs
        self._is_done = False

    def set_sign(self, sign):
        self._is_done = False
        self._ctrl_transition_iter = iter(self._signs[sign])

    def get_next_control(self):
        if self._is_done:
            return None
        next_ctrl = self._get_next_control()
        if next_ctrl is None:
            self._is_done = True
        return next_ctrl

    def _get_next_control(self):
        return next(self._ctrl_transition_iter, None)

    @property
    def is_done(self):
        return self._is_done


# Generates a control trajectory of N steps between start control and end control
def generate_control_trajectory(start_ctrl, end_ctrl, n_steps):
    trajectory = []
    for i in range(n_steps + 1):
        ctrl = start_ctrl + i*(end_ctrl - start_ctrl)/n_steps
        trajectory.append(ctrl)
    return trajectory


class GLFWSimulator:
    def __init__(self, xml_file, controller=None, trajectory_steps=20):
        self._model = mj.MjModel.from_xml_path(filename=xml_file)
        self._controller = controller
        self._trajectory_steps = trajectory_steps

        self._data = mj.MjData(self._model)
        self._camera = mj.MjvCamera()
        self._options = mj.MjvOption()

        self._window = None
        self._scene = None
        self._context = None

        self._mouse_button_left = None
        self._mouse_button_middle = None
        self._mouse_button_right = None
        self._mouse_x_last = 0
        self._mouse_y_last = 0
        self._terminate_simulation = False

        self._sign = ""
        self._trajectory_iter = iter([])
        self._transition_history = []

        self._init_simulation()
        self._init_controller()
        mj.set_mjcb_control(self._controller_fn)

    def _init_simulation(self):
        self._init_world()
        self._init_callbacks()
        self._init_camera()

    def _init_world(self):
        glfw.init()
        self._window = glfw.create_window(1200, 900, 'Simulation', None, None)
        glfw.make_context_current(window=self._window)
        glfw.swap_interval(interval=1)

        mj.mjv_defaultCamera(cam=self._camera)
        # self._options.geomgroup[0] = 0
        mj.mjv_defaultOption(opt=self._options)
        self._scene = mj.MjvScene(self._model, maxgeom=10000)
        self._context = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150.value)

    def _init_callbacks(self):
        glfw.set_key_callback(window=self._window, cbfun=self._keyboard_cb)
        glfw.set_mouse_button_callback(window=self._window, cbfun=self._mouse_button_cb)
        glfw.set_cursor_pos_callback(window=self._window, cbfun=self._mouse_move_cb)
        glfw.set_scroll_callback(window=self._window, cbfun=self._mouse_scroll_cb)

    def _init_camera(self):
        self._camera.azimuth = -180
        self._camera.elevation = -20
        self._camera.distance = 1.6
        self._camera.lookat = [0.37, 0, 0.9]

    def _init_controller(self):
        self._sign = 'test'
        self._controller.set_sign(sign=self._sign)

    def _keyboard_cb(self, window, key, scancode, act, mods):
        # Handles keyboard button events to interact with simulator
        if act == glfw.PRESS:
            if key == glfw.KEY_BACKSPACE:
                mj.mj_resetData(self._model, self._data)
                mj.mj_forward(self._model, self._data)
            elif key == glfw.KEY_ESCAPE:
                self._terminate_simulation = True
            elif key == glfw.KEY_0:
                self._sign = 'test'
                self._controller.set_sign(self._sign)
            elif key == glfw.KEY_1:
                self._sign = 'action'
                self._controller.set_sign(self._sign)
            elif key == glfw.KEY_2:
                self._sign = 'grasp'
                self._controller.set_sign(self._sign)
            elif key == glfw.KEY_3:
                self._sign = 'lift'
                self._controller.set_sign(self._sign)

    def _mouse_button_cb(self, window, button, act, mods):
        # Handles mouse-click events to move/rotate camera
        self._mouse_button_left = glfw.get_mouse_button(window=window, button=glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._mouse_button_middle = glfw.get_mouse_button(window=window, button=glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self._mouse_button_right = glfw.get_mouse_button(window=window, button=glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        glfw.get_cursor_pos(window)

    def _mouse_move_cb(self, window, xpos, ypos):
        # Handles mouse-move callbacks to navigate camera
        dx = xpos - self._mouse_x_last
        dy = ypos - self._mouse_y_last
        self._mouse_x_last = xpos
        self._mouse_y_last = ypos

        if not (self._mouse_button_left or self._mouse_button_middle or self._mouse_button_right):
            return

        width, height = glfw.get_window_size(window=window)
        press_left_shift = glfw.get_key(window=window, key=glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        press_right_shift = glfw.get_key(window=window, key=glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = press_left_shift or press_right_shift

        if self._mouse_button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self._mouse_button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            assert self._mouse_button_middle

            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(
            m=self._model,
            action=action,
            reldx=dx / height,
            reldy=dy / height,
            scn=self._scene,
            cam=self._camera
        )

    def _mouse_scroll_cb(self, window, xoffset, yoffset):
        # Zooms in/out with the camera inside the simulation world
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self._model, action, 0.0, -0.05 * yoffset, self._scene, self._camera)

    def _controller_fn(self, model, data):
        if self._controller.is_done:
            return
        next_ctrl = next(self._trajectory_iter, None)

        if next_ctrl is None:
            start_ctrl = data.ctrl
            end_ctrl = self._controller.get_next_control()
            if end_ctrl is None:
                print('Sign transitions completed')
            else:
                print(f'New control transition is set from {start_ctrl} to {end_ctrl}')
                control_trajectory = generate_control_trajectory(start_ctrl, end_ctrl, self._trajectory_steps)
                self._trajectory_iter = iter(control_trajectory)
                print('New trajectory is computed')
        else:
            data.ctrl = next_ctrl

    def run(self):
        while not glfw.window_should_close(window=self._window) and not self._terminate_simulation:
            time_prev = self._data.time
            while self._data.time - time_prev < 1.0/60.0:
                mj.mj_step(self._model, self._data)

            viewport_width, viewport_height = glfw.get_framebuffer_size(self._window)
            viewport = mj.MjrRect(left=0, bottom=0, width=viewport_width, height=viewport_height)

            mj.mjv_updateScene(self._model, self._data, self._options, None, self._camera, mj.mjtCatBit.mjCAT_ALL.value, self._scene)
            mj.mjr_render(viewport=viewport, scn=self._scene, con=self._context)
            glfw.swap_buffers(window=self._window)
            glfw.poll_events()
        glfw.terminate()


if __name__ == '__main__':
    signs = {}
    # signs['test'] = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # signs['action'] = [[0.1, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    signs['test'] = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    signs['action'] = [[0, -0.2, 0, 0, 0, 0, 0.18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    signs['grasp'] = [[0, -0.2, 0, 0, 0, 0, 0.18, 0, 0, 0, 0, 0, 0, 0, 1.57, 1.57, 0.3, 0.8, 1, 1, 1, 1, 1, 1, 1, 1]]
    signs['lift'] = [[0.3, 0, 0, 0, 0, 0, 0.18, 0, 0, 0, 0, 0, 0, 0, 1.57, 1.57, 0.3, 0.8, 1, 1, 1, 1, 1, 1, 1, 1]]


    controller = Controller(signs)
    # sim = GLFWSimulator("mujoco_menagerie/universal_robots_ur5e/scene.xml")
    # sim = GLFWSimulator("../model/mujoco_menagerie/shadow_hand/scene_left.xml", controller)
    sim = GLFWSimulator("/home/kavin/Documents/PycharmProjects/Mujoco/model/robot_hand/body_new.xml", controller)
    # sim = GLFWSimulator("/home/kavin/Documents/PycharmProjects/Mujoco/robot_hand/demo/robot_device_control.xml", controller)
    sim.run()
