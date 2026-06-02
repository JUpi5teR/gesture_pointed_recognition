import numpy as np
import collections
import math

class FaceGestureRecognizer:
    """
    Detects head shake (left-right) based on face landmarks.
    Uses nose tip, shoulder positions to compute neck vector and analyze motion.
    """

    def __init__(self, history_len=15, shake_threshold=0.25):
        self.history_len = history_len
        self.shake_threshold = shake_threshold

        self.history = collections.deque(maxlen=history_len)
        self.shake_detected = False
        self.last_x_direction = 0

    def detect(self, face_pts):
        """
        Detect face gestures from landmarks.
        Returns: 'shake_detected' or None
        """
        if face_pts is None or len(face_pts) < 468:
            return None

        # Extract key points: nose (1), left shoulder (11), right shoulder (12)
        nose = np.array(face_pts[1], dtype=float)
        left_shoulder = np.array(face_pts[11], dtype=float)
        right_shoulder = np.array(face_pts[12], dtype=float)

        # Compute shoulder center
        shoulder_center = (left_shoulder + right_shoulder) / 2.0

        # Compute neck vector (from shoulder center to nose)
        neck_vec = nose - shoulder_center

        # Store in history
        self.history.append({
            'neck_x': neck_vec[0],
            'neck_y': neck_vec[1],
            'nose': nose.copy()
        })

        # Reset flag each frame
        self.shake_detected = False

        # Detect shake (horizontal motion)
        self._detect_shake()

        if self.shake_detected:
            return 'shake_detected'
        return None

    def _detect_shake(self):
        """Detect head shake (left-right horizontal motion)."""
        if len(self.history) < 5:
            return

        # Compute horizontal (x) displacement of nose over recent frames
        recent_x = [h['nose'][0] for h in list(self.history)[-5:]]

        if len(recent_x) >= 5:
            x_range = max(recent_x) - min(recent_x)

            # Check for alternating left-right motion
            x_diffs = [recent_x[i+1] - recent_x[i] for i in range(len(recent_x)-1)]

            # Detect direction changes (sign flips)
            direction_changes = 0
            for i in range(len(x_diffs)-1):
                if x_diffs[i] * x_diffs[i+1] < 0:  # Sign change
                    direction_changes += 1

            # Shake detected if: large range + direction changes + threshold
            if x_range > self.shake_threshold and direction_changes >= 2:
                self.shake_detected = True

