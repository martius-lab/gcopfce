import os
import imageio


class VideoRecorder(object):
    """ A simple class to record trajectories in environments."""
    def __init__(self, root_dir, height=96, width=96, camera_id=None, fps=30, enabled=False):
        self.save_dir = root_dir
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.enabled = enabled
        self.frames = []

    def init(self):
        self.frames = []

    def record(self, env):
        if self.enabled:
            frame = env.render(mode="rgb_array")
            self.frames.append(frame)

    def save(self, trial_n):
        if self.enabled:
            path = os.path.join(self.save_dir, f'trial_{trial_n}.mp4')
            imageio.mimsave(path, self.frames, fps=self.fps)
