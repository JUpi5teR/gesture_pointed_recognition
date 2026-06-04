import numpy as np
import time
import logging
from utils import point_in_box

logger = logging.getLogger(__name__)

class ButtonTriggerZone:
    def __init__(self, scale=1.5, debounce_ms=200):
        self.scale = scale
        self.debounce_s = debounce_ms / 1000.0
        self._last_trigger_time = 0.0
        self._triggered = False
        self._center = None
        self._original_box = None
        self._zone_box = None

    def set_target(self, center, original_box=None):
        self._center = np.array(center, dtype=float)
        self._original_box = original_box
        if original_box:
            x1, y1, x2, y2 = original_box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            half_w = (x2 - x1) * self.scale / 2.0
            half_h = (y2 - y1) * self.scale / 2.0
            self._zone_box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        else:
            r = 40 * self.scale
            c = self._center
            self._zone_box = (c[0] - r, c[1] - r, c[0] + r, c[1] + r)

    def contains(self, point):
        if self._zone_box is None:
            return False
        return point_in_box(point, self._zone_box)

    def check_trigger(self, finger_tip, nod_occurred):
        if not nod_occurred:
            self._triggered = False
            return False
        if finger_tip is None or self._zone_box is None:
            return False
        now = time.time()
        if now - self._last_trigger_time < self.debounce_s:
            return False
        if self.contains(finger_tip):
            self._last_trigger_time = now
            self._triggered = True
            logger.info("Button triggered via nod + finger in zone")
            return True
        self._triggered = False
        return False

    def get_zone_box(self):
        return self._zone_box

    def reset(self):
        self._last_trigger_time = 0.0
        self._triggered = False
        self._center = None
        self._original_box = None
        self._zone_box = None
