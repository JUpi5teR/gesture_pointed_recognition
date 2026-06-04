import cv2
import time
import numpy as np
import logging
import os
from enum import Enum
from pathlib import Path
from .login_manager import LoginManager
from .gpt_analyzer import GPTImageAnalyzer
from .world_anchor import WorldAnchor
from .ar_panel import ARPanel

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache"

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
        self._bound_box = None
        self._bound_label = None
        self._bound_dim = None
        self._image_cached = False
        self._cache_path = None
        self._cached_result = None
        self._panel_initialized = False
        self._thumbs_up_latch = False
        self._prev_scroll_y = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if api_key:
            self.login.login(api_key)
        # Sync GPT key and provider from saved login session
        if self.login.is_logged_in and self.login.api_key:
            self.gpt.api_key = self.login.api_key
            self.gpt.base_url = self.login.base_url
            self.gpt.model = self.login.model_name
            self.gpt.provider = self.login.provider

    def on_target_bound(self, frame, box, label):
        self.anchor.snap(self._get_box_center(box), box)
        self._bound_box = tuple(box) if box is not None else None
        w = (box[2] - box[0]) if box else 100
        h = (box[3] - box[1]) if box else 100
        self._bound_dim = (int(w), int(h))
        self._bound_label = label
        self._image_cached = False
        self._cache_path = None
        self._cached_result = None
        self._panel_initialized = False
        self._thumbs_up_latch = False
        self.state = ARState.OFF
        logger.info("AR: target bound (%s), box=%dx%d - thumbs up #1 to capture", label, int(w), int(h))

    def _get_crop_rect(self, frame, target_pt):
        if self._bound_dim is None or target_pt is None:
            return None
        cx, cy = target_pt
        bw, bh = self._bound_dim
        h, w = frame.shape[:2]
        x1 = int(max(0, cx - bw // 2))
        y1 = int(max(0, cy - bh // 2))
        x2 = int(min(w, cx + bw // 2))
        y2 = int(min(h, cy + bh // 2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

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

        # Already showing panel -> enter AR_CONTROL
        if self.state == ARState.AR_VISIBLE:
            self.state = ARState.AR_CONTROL
            self._prev_scroll_y = None
            logger.info("AR: AR_CONTROL entered via thumbs up")
            return True

        # First thumbs up: capture screenshot to cache
        if not self._image_cached and current_frame is not None:
            pos = self.anchor.get_position()
            rect = self._get_crop_rect(current_frame, pos)
            if rect is not None:
                x1, y1, x2, y2 = rect
                crop = current_frame[y1:y2, x1:x2].copy()
                stamp = int(time.time())
                fname = f"capture_{stamp}.jpg"
                self._cache_path = str(CACHE_DIR / fname)
                cv2.imwrite(self._cache_path, crop)
                self._image_cached = True
                logger.info("AR: screenshot cached -> %s", fname)
            return True

        # Second thumbs up: send cached image to GPT, open panel
        if self._image_cached and not self._panel_initialized and self._cache_path is not None:
            pos = self.anchor.get_position() or (300, 200)
            label = self._bound_label or "object"
            try:
                with open(self._cache_path, "rb") as f:
                    image_bytes = f.read()
                logger.info("AR: sending cached image to GPT for %s ...", label)
                self._cached_result = self.gpt.analyze_image(image_bytes)
            except Exception as e:
                self._cached_result = f"[Load failed: {e}]"
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
        self._bound_box = None
        self._bound_label = None
        self._bound_dim = None
        self._image_cached = False
        self._cache_path = None
        self._cached_result = None
        self._panel_initialized = False
        self._thumbs_up_latch = False
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
        if self.state == ARState.OFF:
            pos = self.anchor.get_position()
            if pos and self._bound_dim is not None:
                bw, bh = self._bound_dim
                x1 = int(pos[0] - bw // 2)
                y1 = int(pos[1] - bh // 2)
                x2 = int(pos[0] + bw // 2)
                y2 = int(pos[1] + bh // 2)
                color = (0, 255, 100) if self._image_cached else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = "Captured - thumbs up for GPT" if self._image_cached else "Thumbs up to capture"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return frame

    def _is_thumbs_up(self, hands):
        if not hands or len(hands) == 0:
            return False
        return hands[0].get("code", 0) == 1

    def _get_box_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
