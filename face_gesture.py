import numpy as np
import collections
import math

class FaceGestureRecognizer:
    """
    Detects head nod (up-down) and head shake (left-right) based on face landmarks.
    Uses nose tip, shoulder positions to compute neck vector and analyze angles.
    """

    def __init__(self, history_len=15, nod_threshold=0.12, shake_threshold=0.25):
        self.history_len = history_len
        self.nod_threshold = nod_threshold
        self.shake_threshold = shake_threshold

        self.history = collections.deque(maxlen=history_len)
        self.nod_count = 0
        self.last_nod_frame = -100
        self.nod_detected = False
        self.shake_detected = False
        self.last_x_direction = 0

    def detect(self, face_pts):
        """
        Detect face gestures from landmarks.
        Returns: ('nod_detected', 'shake_detected', or None)
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

        # Compute torso horizontal line (from left to right shoulder)
        torso_vec = right_shoulder - left_shoulder

        # Compute angle between neck and torso vectors
        angle = self._compute_angle(neck_vec, torso_vec)

        # Store in history
        self.history.append({
            'angle': angle,
            'neck_x': neck_vec[0],
            'neck_y': neck_vec[1],
            'nose': nose.copy()
        })

        # Reset flags each frame
        self.nod_detected = False
        self.shake_detected = False

        # Detect nod (vertical motion)
        self._detect_nod()

        # Detect shake (horizontal motion)
        self._detect_shake()

        if self.nod_detected:
            return 'nod_detected'
        elif self.shake_detected:
            return 'shake_detected'
        return None

    def _compute_angle(self, v1, v2):
        """Compute angle between two 2D vectors in radians."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return math.acos(cos_angle)

    def _detect_nod(self):
        """Detect two sequential nod motions (downward head tilt)."""
        if len(self.history) < 5:
            return

        # Look for large angle changes (nod = temporary increase in angle)
        current_angle = self.history[-1]['angle']
        prev_angle = self.history[-2]['angle'] if len(self.history) > 1 else current_angle

        # Check if we're seeing a downward motion (chin going down, angle increases)
        angle_change = current_angle - prev_angle

        # Detect first nod
        if angle_change > self.nod_threshold:
            if self.nod_count == 0:
                self.nod_count = 1
                self.last_nod_frame = len(self.history)

        # Detect second nod within time window
        elif angle_change > self.nod_threshold and self.nod_count == 1:
            if len(self.history) - self.last_nod_frame < 20:  # Within 20 frames
                self.nod_count = 2
                self.nod_detected = True
                self.nod_count = 0  # Reset for next sequence

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
