import math
import numpy as np


class HeadGestureRecognizer:
    """
    Skeleton-based head gesture recognition using Pose landmarks.
    Uses nose tip (0), left shoulder (11), right shoulder (12) to build
    a neck vector from shoulder-center to nose, then computes Pitch and Yaw.

    In 2D image projection, pure forward head tilt (pitch) manifests as
    the nose tip moving closer to the shoulder-center level (shorter neck vector),
    while lateral head turn (yaw) manifests as the nose deviating sideways.

    Pitch metric: angle of neck vector from the "up-perpendicular" of the shoulder axis.
      - 0deg = head upright (neck aligned with up-perpendicular)
      - Larger = head tilted forward/down
      - Also uses neck-length / shoulder-width ratio as a supplementary metric
        for purely front-facing cases where the neck vector doesn't rotate in 2D.

    Yaw metric: atan2(lateral, forward) of the neck vector in shoulder-relative coords.
      - Negative = left, Positive = right

    Thresholds:
      Pitch > 18deg => head down; Pitch < 8deg => head up  -> nod count
      Yaw < -12deg => left; Yaw > 12deg => right  -> shake count

    State flags nod_down / shake_left prevent double-counting.
    """

    def __init__(self):
        self.nod_count = 0
        self.shake_count = 0
        self.nod_down = False
        self.shake_left = False

        # Thresholds (degrees)
        self.nod_down_thresh = 18.0
        self.nod_up_thresh = 8.0
        self.shake_left_thresh = -12.0
        self.shake_right_thresh = 12.0

        # Calibration: store the initial "upright" neck ratio for adaptive pitch
        self.calibrated = False
        self.calibrated_neck_shoulder_ratio = None
        self.calibration_frames = 0
        self.calibration_needed = 15  # frames to calibrate

    def compute_pitch_yaw(self, pose_pts):
        """
        Compute Pitch and Yaw from pose skeleton landmarks.
        pose_pts: list of (x, y, z) from PoseDetector.
        Returns (pitch_deg, yaw_deg) or (None, None).
        """
        if pose_pts is None or len(pose_pts) < 13:
            return None, None

        nose = np.array(pose_pts[0][:2], dtype=float)
        l_shoulder = np.array(pose_pts[11][:2], dtype=float)
        r_shoulder = np.array(pose_pts[12][:2], dtype=float)

        shoulder_center = (l_shoulder + r_shoulder) / 2.0
        neck_vec = nose - shoulder_center
        neck_norm = np.linalg.norm(neck_vec)

        shoulder_axis = r_shoulder - l_shoulder
        shoulder_width = np.linalg.norm(shoulder_axis)

        if neck_norm < 1e-3 or shoulder_width < 1e-3:
            return None, None

        shoulder_unit = shoulder_axis / shoulder_width

        # Up-perpendicular to shoulder axis (pointing toward nose, i.e., upward in image)
        perp_unit = np.array([-shoulder_unit[1], shoulder_unit[0]])
        if perp_unit[1] > 0:
            perp_unit = -perp_unit  # Ensure pointing upward (negative y)

        # Decompose neck vector
        lateral_component = np.dot(neck_vec, shoulder_unit)
        forward_component = np.dot(neck_vec, perp_unit)  # positive = toward nose/upward

        # --- Pitch computation ---
        # Method 1: Angle between neck_vec and the up-perpendicular in the sagittal plane.
        # This detects lateral tilting of the neck vector.
        neck_lateral = np.dot(neck_vec, shoulder_unit)
        neck_sagittal = neck_vec - neck_lateral * shoulder_unit
        neck_sagittal_norm = np.linalg.norm(neck_sagittal)

        up_dir = np.array([0.0, -1.0])
        up_lateral = np.dot(up_dir, shoulder_unit)
        up_sagittal = up_dir - up_lateral * shoulder_unit
        up_sagittal_norm = np.linalg.norm(up_sagittal)

        if neck_sagittal_norm > 1e-6 and up_sagittal_norm > 1e-6:
            angle_from_up = math.degrees(math.acos(
                np.clip(np.dot(neck_sagittal / neck_sagittal_norm, up_sagittal / up_sagittal_norm), -1.0, 1.0)
            ))
        else:
            angle_from_up = 90.0

        # Method 2: Ratio-based pitch (for front-facing cases where neck doesn't rotate in 2D).
        # When head tilts down, the nose approaches shoulder level -> neck/shoulder ratio decreases.
        # A typical upright person has neck_length ~ 0.8-1.5 * shoulder_width.
        # A tilted-down person has a shorter ratio.
        neck_shoulder_ratio = neck_norm / shoulder_width

        # Combine: pitch is the angle-based metric, but enhanced by ratio when
        # the angle method gives near-zero (front-facing case).
        # When the person is front-facing and head tilts down:
        #   - angle_from_up stays ~0 (neck still points up)
        #   - but neck_shoulder_ratio decreases significantly
        #
        # We use the ratio to compute an equivalent pitch angle:
        #   ratio_pitch = max(0, (1 - ratio/reference_ratio)) * 90
        # where reference_ratio is the calibrated upright ratio.
        #
        # If not calibrated, use a default reference of 0.9 (typical upright ratio).

        ref_ratio = self.calibrated_neck_shoulder_ratio if self.calibrated else 0.9
        if ref_ratio < 0.2:
            ref_ratio = 0.9

        # The equivalent pitch from ratio change:
        # When upright: ratio = ref_ratio -> ratio_pitch = 0
        # When tilted 45deg: ratio ~ ref_ratio * cos(45deg) -> ratio_pitch ~ 45
        # When head down (90deg): ratio ~ 0 -> ratio_pitch = 90
        if neck_shoulder_ratio >= ref_ratio:
            ratio_pitch = 0.0
        else:
            # ratio = ref_ratio * cos(pitch) approximately
            cos_equiv = neck_shoulder_ratio / ref_ratio
            cos_equiv = min(1.0, cos_equiv)
            ratio_pitch = math.degrees(math.acos(cos_equiv))

        # Final pitch: use the MAXIMUM of angle-based and ratio-based metrics.
        # Angle-based works when person has body rotation.
        # Ratio-based works for front-facing pitch.
        pitch_deg = max(angle_from_up, ratio_pitch)

        # --- Yaw computation ---
        if abs(forward_component) < 1e-3:
            if abs(lateral_component) > 1e-3:
                yaw_deg = 90.0 * (1.0 if lateral_component > 0 else -1.0)
            else:
                yaw_deg = 0.0
        else:
            yaw_rad = math.atan2(lateral_component, forward_component)
            yaw_deg = math.degrees(yaw_rad)

        return pitch_deg, yaw_deg

    def _calibrate(self, pose_pts):
        """Auto-calibrate the upright neck-shoulder ratio during initial frames."""
        if self.calibrated:
            return
        if pose_pts is None or len(pose_pts) < 13:
            return
        nose = np.array(pose_pts[0][:2], dtype=float)
        l_shoulder = np.array(pose_pts[11][:2], dtype=float)
        r_shoulder = np.array(pose_pts[12][:2], dtype=float)
        shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
        neck_length = np.linalg.norm(nose - (l_shoulder + r_shoulder) / 2.0)
        if shoulder_width < 1e-3:
            return
        ratio = neck_length / shoulder_width
        # Only calibrate if ratio seems reasonable (person is roughly upright)
        if ratio > 0.3:
            if self.calibrated_neck_shoulder_ratio is None:
                self.calibrated_neck_shoulder_ratio = ratio
            else:
                # Running average
                self.calibrated_neck_shoulder_ratio = (
                    0.7 * self.calibrated_neck_shoulder_ratio + 0.3 * ratio
                )
            self.calibration_frames += 1
            if self.calibration_frames >= self.calibration_needed:
                self.calibrated = True

    def update(self, pose_pts):
        """
        Update gesture recognition with new pose landmarks.
        Returns dict with pitch, yaw, nod_count, shake_count, nod_down, shake_left.
        """
        # Auto-calibrate during first frames
        self._calibrate(pose_pts)

        pitch, yaw = self.compute_pitch_yaw(pose_pts)

        if pitch is None or yaw is None:
            return {
                'pitch': None, 'yaw': None,
                'nod_count': self.nod_count, 'shake_count': self.shake_count,
                'nod_down': self.nod_down, 'shake_left': self.shake_left
            }

        # --- Nod detection (Pitch-based) ---
        if pitch > self.nod_down_thresh:
            self.nod_down = True
        if self.nod_down and pitch < self.nod_up_thresh:
            self.nod_count += 1
            self.nod_down = False

        # --- Shake detection (Yaw-based) ---
        if yaw < self.shake_left_thresh:
            self.shake_left = True
        if self.shake_left and yaw > self.shake_right_thresh:
            self.shake_count += 1
            self.shake_left = False

        return {
            'pitch': pitch, 'yaw': yaw,
            'nod_count': self.nod_count, 'shake_count': self.shake_count,
            'nod_down': self.nod_down, 'shake_left': self.shake_left
        }

    def reset_shake(self):
        self.shake_count = 0
        self.shake_left = False

    def reset_nod(self):
        self.nod_count = 0
        self.nod_down = False

    def reset(self):
        self.nod_count = 0
        self.shake_count = 0
        self.nod_down = False
        self.shake_left = False
        self.calibrated = False
        self.calibrated_neck_shoulder_ratio = None
        self.calibration_frames = 0
