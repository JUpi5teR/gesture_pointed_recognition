import cv2
import time
import numpy as np
import logging
from enum import Enum
from .login_manager import LoginManager
from .gpt_analyzer import GPTImageAnalyzer
from .world_anchor import WorldAnchor
from .ar_panel import ARPanel

logger = logging.getLogger(__name__)

class ARState(Enum):
    OFF = "OFF"
    AR_VISIBLE = "AR_VISIBLE"
    AR_CONTROL = "AR_CONTROL"
    LOST = "LOST"

class ARManager:
    def __init__(self, api_key=""):
        self.state = ARState.OFF
        self.login = LoginManager()
        self.gpt = GPTImageAnalyzer(api_key=api_key)
        self.anchor = WorldAnchor()
        self.panel = ARPanel()
        self._cached_result = None
        self._cached_label = None
        self._panel_initialized = False
        self._thumbs_up_latch = False
        self._prev_scroll_y = None
        if api_key:
            self.login.login(api_key)

    def on_target_bound(self, frame, box, label):
        self.anchor.snap(self._get_box_center(box), box)
        self._bound_box = tuple(box) if box is not None else None
        self._bound_label = label
        self._thumbs_up_latch = False
        self._panel_initialized = False
        logger.info("AR: target bound (%s) - thumbs up to scan", label)

    def on_target_lost(self):
        self.state = ARState.LOST
        self.anchor.update()
        logger.info("AR: LOST")

    def on_target_visible(self, position, box):
        self.anchor.update(position, box)
        if self.panel.is_visible:
            self.panel.update_position(position)
        if self.state == ARState.LOST:
            self.state = ARState.AR_VISIBLE

    def try_open_panel(self, hands, current_frame=None):
        if self.state not in (ARState.OFF, ARState.AR_VISIBLE, ARState.LOST):
            return False
        is_up = self._is_thumbs_up(hands)
        if not is_up:
            self._thumbs_up_latch = False
            return False
        if self._thumbs_up_latch:
            return False
        self._thumbs_up_latch = True
        if self.state == ARState.AR_VISIBLE:
            self.state = ARState.AR_CONTROL
            self._prev_scroll_y = None
            logger.info("AR: AR_CONTROL entered via thumbs up")
            return True
        if not self._panel_initialized and current_frame is not None and self._bound_box is not None:
            pos = self.anchor.get_position() or (300, 200)
            label = self._bound_label or "object"
            image_bytes = self.gpt.crop_from_frame(current_frame, self._bound_box)
            if image_bytes:
                logger.info("AR: sending cropped image to GPT for %s ...", label)
                self._cached_result = self.gpt.analyze_image(image_bytes)
            else:
                self._cached_result = "[Crop failed]"
            self._cached_label = label
            self.panel.show(self._cached_result, pos)
            self.state = ARState.AR_VISIBLE
            self._panel_initialized = True
            logger.info("AR: panel opened - GPT result ready")
            return True
        return False

    def update_scroll(self, hands, frame_shape=None):
        if self.state != ARState.AR_CONTROL:
            return
        if not hands or len(hands) == 0:
            self._prev_scroll_y = None
            return
        pts = hands[0].get("pts", [])
        if len(pts) < 12:
            self._prev_scroll_y = None
            return
        tip_idx = np.array(pts[8][:2], dtype=float)
        tip_mid = np.array(pts[12][:2], dtype=float)
        avg_y = (tip_idx[1] + tip_mid[1]) / 2.0
        if self._prev_scroll_y is not None:
            dy = self._prev_scroll_y - avg_y
            if abs(dy) > 5:
                direction = 1 if dy > 0 else -1
                self.panel.scroll(direction)
        self._prev_scroll_y = avg_y

    def on_target_unlock(self):
        self.state = ARState.OFF
        self.panel.hide()
        self.anchor.reset()
        self._panel_initialized = False
        self._cached_result = None
        self._cached_label = None
        self._thumbs_up_latch = False
        self._bound_box = None
        self._bound_label = None
        logger.info("AR: OFF")

    def draw(self, frame):
        if self.state in (ARState.AR_VISIBLE, ARState.AR_CONTROL):
            frame = self.panel.draw(frame)
            mode = "AR_CONTROL" if self.state == ARState.AR_CONTROL else "AR_VISIBLE"
            cv2.putText(frame, mode, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 2)
            return frame
        if self.state == ARState.LOST:
            pos = self.anchor.get_position()
            if pos:
                cv2.putText(frame, "Target Lost - Panel Held", (pos[0] - 60, pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 200), 2)
            return frame
        return frame

    def _is_thumbs_up(self, hands):
        if not hands or len(hands) == 0:
            return False
        return hands[0].get("code", 0) == 1

    def _get_box_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
