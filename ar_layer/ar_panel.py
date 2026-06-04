import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ARPanel:
    def __init__(self, panel_width=300, panel_height=200, scroll_speed=20):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.scroll_speed = scroll_speed
        self.content = ""
        self._scroll_offset = 0
        self._max_scroll = 0
        self._position = None
        self._visible = False
        self._lines = []

    def show(self, content, position):
        self.content = content
        self._position = position
        self._visible = True
        self._scroll_offset = 0
        self._lines = self._wrap_text(content, self.panel_width - 20)
        text_h = len(self._lines) * 18
        self._max_scroll = max(0, text_h - self.panel_height + 30)

    def hide(self):
        self._visible = False
        self._scroll_offset = 0

    def update_position(self, position):
        if position is not None:
            self._position = position

    def scroll(self, direction):
        if not self._visible:
            return
        self._scroll_offset += direction * self.scroll_speed
        self._scroll_offset = max(0, min(self._scroll_offset, self._max_scroll))

    def _wrap_text(self, text, max_width_px):
        import textwrap
        lines = []
        for paragraph in text.split("\n"):
            wrapped = textwrap.wrap(paragraph, width=40)
            lines.extend(wrapped if wrapped else [""])
        return lines if lines else [""]

    def draw(self, frame):
        if not self._visible or self._position is None:
            return frame
        fx, fy = int(self._position[0]), int(self._position[1])
        h, w = frame.shape[:2]
        px = fx + 30
        py = fy - self.panel_height // 2
        px = max(5, min(px, w - self.panel_width - 5))
        py = max(5, min(py, h - self.panel_height - 5))
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + self.panel_width, py + self.panel_height), (30, 30, 50), -1)
        cv2.rectangle(overlay, (px, py), (px + self.panel_width, py + self.panel_height), (100, 100, 180), 2)
        alpha = 0.92
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.putText(frame, "AR Info", (px + 8, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
        cv2.line(frame, (px + 8, py + 26), (px + self.panel_width - 8, py + 26), (80, 80, 120), 1)
        line_y = py + 46 - self._scroll_offset
        for line in self._lines:
            if py + 30 < line_y < py + self.panel_height - 5:
                cv2.putText(frame, line, (px + 10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 240), 1)
            line_y += 18
        if self._max_scroll > 0:
            bar_h = max(20, int(self.panel_height * self.panel_height / (self._max_scroll + self.panel_height)))
            bar_y = py + int(self._scroll_offset / max(1, self._max_scroll) * (self.panel_height - bar_h))
            cv2.rectangle(frame, (px + self.panel_width - 6, bar_y), (px + self.panel_width - 2, bar_y + bar_h), (150, 150, 200), -1)
        return frame

    @property
    def is_visible(self):
        return self._visible
