import numpy as np
import logging

logger = logging.getLogger(__name__)

class WorldAnchor:
    def __init__(self, decay_rate=0.95, max_hold_frames=90):
        self.position = None
        self.box = None
        self.velocity = np.array([0.0, 0.0])
        self.is_locked = False
        self.lost_frames = 0
        self.decay_rate = decay_rate
        self.max_hold_frames = max_hold_frames
        self._snap_position = None
        self._snap_box = None

    def snap(self, position, box=None):
        self._snap_position = np.array(position, dtype=float)
        self.position = np.array(position, dtype=float)
        self.box = tuple(box) if box else None
        self._snap_box = tuple(box) if box else None
        self.velocity = np.array([0.0, 0.0])
        self.is_locked = False
        self.lost_frames = 0

    def update(self, current_position=None, current_box=None):
        if current_position is not None and current_box is not None:
            new_pos = np.array(current_position, dtype=float)
            self.velocity = (new_pos - self.position) * 0.3 + self.velocity * 0.7
            self.position = new_pos
            self.box = tuple(current_box)
            self.lost_frames = 0
            self.is_locked = False
            return self.position
        self.lost_frames += 1
        if self.lost_frames > self.max_hold_frames:
            self.is_locked = True
        if self.position is not None:
            self.position = self.position + self.velocity * 0.5
            self.velocity *= self.decay_rate
        return self.position

    def reset(self):
        self.position = None
        self.box = None
        self.velocity = np.array([0.0, 0.0])
        self.is_locked = False
        self.lost_frames = 0
        self._snap_position = None
        self._snap_box = None

    def get_position(self):
        if self.position is not None:
            return (int(self.position[0]), int(self.position[1]))
        return None

    @property
    def is_holding(self):
        return self.position is not None and self.lost_frames <= self.max_hold_frames
