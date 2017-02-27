from enum import Enum
import glfw
import OpenGL.GL as gl
import threading
import numpy as np
from PIL import Image
import queue

from env.config import get_config
from env.env import DrawData, Line
from env.buttons import get_buttons


def key_callback(window, key, scan_code, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        get_buttons().on_press_esc()

    if key == glfw.KEY_UP and action == glfw.PRESS:
        get_buttons().on_press_up()
    if key == glfw.KEY_UP and action == glfw.RELEASE:
        get_buttons().on_release_up()

    if key == glfw.KEY_DOWN and action == glfw.PRESS:
        get_buttons().on_press_down()
    if key == glfw.KEY_DOWN and action == glfw.RELEASE:
        get_buttons().on_release_down()

class EnvInfo:

    def __init__(self, env, fbo, texture):
        self._env = env
        self._fbo = fbo
        self._texture = texture

    @property
    def env(self):
        return self._env

    @property
    def fbo(self):
        return self._fbo

    @property
    def texture(self):
        return self._texture


class CommandType(Enum):
    START_ENV = 0
    DRAW = 1
    RENDER = 2
    STOP_ENV = 3
    STOP = 4
    GRAB_DATA = 5
    REGISTER_KEY_CALLBACK = 6


class Command:
    def __init__(self, command_type: CommandType, payload):
        self._command_type = command_type
        self._payload = payload

    @property
    def command_type(self) -> CommandType:
        return self._command_type

    @property
    def payload(self):
        return self._payload


def window_close_callback(window):
    glfw.set_window_should_close(window, False)


class UiThread:
    def __init__(self):
        thread_func = lambda: self.gui_thread()
        self._t = threading.Thread(target=(thread_func))
        self._started = False
        # Task queue
        self._q = queue.Queue()
        # Window to render into
        self._window = None
        # Flag which indicate if window hidden
        self._hidden = True
        # Buffer to grab image
        self._buffer = (gl.GLubyte * (3 * get_config().window_px_width * get_config().window_px_height))(0)
        # Registered environments
        self._envs = {}

    def start_env(self, env):
        self._q.put(Command(CommandType.START_ENV, env))

    def stop_env(self, env):
        self._q.put(Command(CommandType.STOP_ENV, env))

    def register_key_callback(self, callback):
        self._q.put(Command(CommandType.REGISTER_KEY_CALLBACK, callback))

    def _on_register_key_callback(self, callback):
        glfw.set_key_callback(self._window, cbfun=callback)

    def gui_thread(self):
        # Initialize glwf
        if not glfw.init():
            raise "Glfw initialization error"

        # Create window with hints
        # glfw.window_hint(glfw.DECORATED, False)
        glfw.window_hint(glfw.FOCUSED, True)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.VISIBLE, False)

        self._window = glfw.create_window(get_config().window_px_width, get_config().window_px_height, "Bar game", None,
                                          None)
        if not self._window:
            glfw.terminate()
            raise "Can not create window"

        # Disable close button
        glfw.set_window_close_callback(self._window, window_close_callback)

        # Make the window's context current
        glfw.make_context_current(self._window)

        # setup viewport
        gl.glViewport(0, 0, get_config().window_px_width, get_config().window_px_height)

        # Setup projection matrix
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, 1, 0, 1, 1, -1)
        # Setup modelview matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        # Loop until the user closes the window
        while not glfw.window_should_close(self._window):
            # Poll for and process events
            glfw.poll_events()

            # Render here, e.g. using pyOpenGL
            if self._process_queue():
                break

        glfw.terminate()

    def render(self, env):
        self._q.put(Command(CommandType.RENDER, env))

    def draw(self, dd: DrawData):
        self._q.put(Command(CommandType.DRAW, dd))

    def grab_data(self, env):
        self._q.put(Command(CommandType.GRAB_DATA, env))

    def start(self):
        if not self._started:
            self._started = True
            self._t.start()

        self.register_key_callback(key_callback)

    def stop(self):
        self._q.put(Command(CommandType.STOP, None))
        self._t.join()

    def _process_queue(self) -> bool:
        try:
            c = self._q.get(timeout=0.001)
            if c.command_type == CommandType.START_ENV:
                self._on_start_env(c.payload)
            elif c.command_type == CommandType.DRAW:
                self._on_draw(c.payload)
            elif c.command_type == CommandType.RENDER:
                self._on_render(c.payload)
            elif c.command_type == CommandType.GRAB_DATA:
                self._on_grab_data(c.payload)
            elif c.command_type == CommandType.STOP_ENV:
                self._on_stop_env(c.payload)
            elif c.command_type == CommandType.REGISTER_KEY_CALLBACK:
                self._on_register_key_callback(c.payload)
            elif c.command_type == CommandType.STOP:
                return True
        except queue.Empty:
            pass
        return False

    def _on_stop_env(self, env):
        env_info = self._envs[env.id]
        # Cleanup
        gl.glDeleteTextures([env_info.texture])
        gl.glDeleteFramebuffers([env_info.fbo])

        self._envs.pop(env.id)

    def _on_start_env(self, env):
        # create framebuffer object to render into
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        # create texture
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, get_config().window_px_width, get_config().window_px_height, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        # bind texture to FBO
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, texture, 0)

        # Check FBO status
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise "FBO error"

        # Bind default FBO
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        env_info = EnvInfo(env, fbo=fbo, texture=texture)
        self._envs[env.id] = env_info

    def _on_render(self, env):
        env_info = self._envs[env.id]

        if self._hidden:
            glfw.show_window(self._window)
            self._hidden = False

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, env_info.texture)

        # http://stackoverflow.com/questions/27712437/opengl-how-do-i-affect-the-lighting-on-a-textured-plane
        gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_REPLACE)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0)
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex3f(0.0, 1.0, 0)
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex3f(1.0, 1.0, 0)
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex3f(1.0, 0.0, 0)
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)

        # Swap front and back buffers
        glfw.swap_buffers(self._window)

    def _on_draw(self, dd: DrawData):
        env_info = self._envs[dd.env.id]
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, env_info.fbo)
        gl.glBindTexture(gl.GL_TEXTURE_2D, env_info.texture)

        # Clear screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glColor3f(0.600, 0.196, 0.800)

        quads = dd.quads
        count = quads.shape[0]

        for i in range(count):
            # Square rendering
            gl.glBegin(gl.GL_QUADS)
            gl.glColor3f(0.600, 0.196, 0.800)
            x_l = quads[i][0]
            y_l = quads[i][1]
            x_r = quads[i][2]
            y_h = quads[i][3]

            gl.glVertex3f(x_l, y_h, 0)
            gl.glVertex3f(x_r, y_h, 0)
            gl.glVertex3f(x_r, y_l, 0)
            gl.glVertex3f(x_l, y_l, 0)
            gl.glEnd()

        # draw line
        gl.glLineWidth(3.0)

        if dd.line is not None:
            gl.glBegin(gl.GL_LINES)
            if dd.line.pl_positive:
                gl.glColor3f(0.196, 0.804, 0.196)
            else:
                gl.glColor3f(0.502, 0.000, 0.000)
            gl.glVertex3f(dd.line.x0, dd.line.y0, 0.0)
            gl.glVertex3f(dd.line.x1, dd.line.y1, 0.0)
            gl.glEnd()

    def _on_grab_data(self, env):
        env_info = self._envs[env.id]

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, env_info.fbo)
        gl.glReadPixels(0, 0, get_config().window_px_width, get_config().window_px_height, gl.GL_RGB,
                        gl.GL_UNSIGNED_BYTE, self._buffer)
        arr = np.array(self._buffer, dtype=np.uint8).reshape(
            (get_config().window_px_height, get_config().window_px_width, 3))
        arr = arr[::-1, :, :]
        # debug check
        # image = Image.fromarray(arr)
        # image.save('state.png')
        env.post_data(arr.reshape((-1,get_config().window_px_height, get_config().window_px_width, 3)))

_renderer = UiThread()


def get_ui_thread() -> UiThread:
    return _renderer