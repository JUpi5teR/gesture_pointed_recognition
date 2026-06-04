import math
import numpy as np
from config import (PITCH_NOD_DOWN, PITCH_NOD_UP, YAW_SHAKE_LEFT, YAW_SHAKE_RIGHT,
                    CALIBRATION_FRAMES)


class HeadGestureRecognizer:
    """
    Skeleton-based head gesture recognition using Pose landmarks.
    Key points: nose(0), Lshoulder(11), Rshoulder(12).
    Build neck vector (shoulder_center -> nose) to compute Pitch/Yaw.
    State flags nod_down / shake_left prevent double-counting.
    """

    def __init__(self, nod_down_thresh=None, nod_up_thresh=None,
                 shake_left_thresh=None, shake_right_thresh=None):
        self.nod_count = 0
        self.shake_count = 0
        self.nod_down = False
        self.shake_left = False

        self.nod_down_thresh = nod_down_thresh if nod_down_thresh is not None else PITCH_NOD_DOWN
        self.nod_up_thresh = nod_up_thresh if nod_up_thresh is not None else PITCH_NOD_UP
        self.shake_left_thresh = shake_left_thresh if shake_left_thresh is not None else YAW_SHAKE_LEFT
        self.shake_right_thresh = shake_right_thresh if shake_right_thresh is not None else YAW_SHAKE_RIGHT

        # Calibration
        self.calibrated = False
        self.calibrated_neck_shoulder_ratio = None
        self.calibration_frames = 0
        self.calibration_needed = CALIBRATION_FRAMES

        # Confirm-state helpers (for popup nod/shake, separate from main counters)
        self._confirm_nod_down = False
        self._confirm_shake_left = False

    def compute_pitch_yaw(self, pose_pts):
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
        perp_unit = np.array([-shoulder_unit[1], shoulder_unit[0]])
        if perp_unit[1] > 0:
            perp_unit = -perp_unit
        lateral_component = np.dot(neck_vec, shoulder_unit)
        forward_component = np.dot(neck_vec, perp_unit)

        # Pitch: max of angle-from-up and ratio-based
        neck_lateral = np.dot(neck_vec, shoulder_unit)
        neck_sagittal = neck_vec - neck_lateral * shoulder_unit
        neck_sagittal_norm = np.linalg.norm(neck_sagittal)
        up_dir = np.array([0.0, -1.0])
        up_lateral = np.dot(up_dir, shoulder_unit)
        up_sagittal = up_dir - up_lateral * shoulder_unit
        up_sagittal_norm = np.linalg.norm(up_sagittal)
        if neck_sagittal_norm > 1e-6 and up_sagittal_norm > 1e-6:
            angle_from_up = math.degrees(math.acos(
                np.clip(np.dot(neck_sagittal/neck_sagittal_norm, up_sagittal/up_sagittal_norm), -1.0, 1.0)))
        else:
            angle_from_up = 90.0
        neck_shoulder_ratio = neck_norm / shoulder_width
        ref_ratio = self.calibrated_neck_shoulder_ratio if self.calibrated else 0.9
        if ref_ratio < 0.2: ref_ratio = 0.9
        if neck_shoulder_ratio >= ref_ratio:
            ratio_pitch = 0.0
        else:
            ratio_pitch = math.degrees(math.acos(min(1.0, neck_shoulder_ratio / ref_ratio)))
        pitch_deg = max(angle_from_up, ratio_pitch)

        # Yaw
        if abs(forward_component) < 1e-3:
            yaw_deg = 90.0*(1 if lateral_component>0 else -1) if abs(lateral_component)>1e-3 else 0.0
        else:
            yaw_deg = math.degrees(math.atan2(lateral_component, forward_component))
        return pitch_deg, yaw_deg

    def _calibrate(self, pose_pts):
        if self.calibrated: return
        if pose_pts is None or len(pose_pts) < 13: return
        nose = np.array(pose_pts[0][:2], dtype=float)
        l_shoulder = np.array(pose_pts[11][:2], dtype=float)
        r_shoulder = np.array(pose_pts[12][:2], dtype=float)
        sw = np.linalg.norm(r_shoulder - l_shoulder)
        nl = np.linalg.norm(nose - (l_shoulder + r_shoulder) / 2.0)
        if sw < 1e-3: return
        ratio = nl / sw
        if ratio > 0.3:
            if self.calibrated_neck_shoulder_ratio is None:
                self.calibrated_neck_shoulder_ratio = ratio
            else:
                self.calibrated_neck_shoulder_ratio = 0.7*self.calibrated_neck_shoulder_ratio + 0.3*ratio
            self.calibration_frames += 1
            if self.calibration_frames >= self.calibration_needed:
                self.calibrated = True

    def update(self, pose_pts):
        """Update main gesture recognition. Returns dict with pitch/yaw/counts."""
        self._calibrate(pose_pts)
        pitch, yaw = self.compute_pitch_yaw(pose_pts)
        if pitch is None or yaw is None:
            return {'pitch': None, 'yaw': None, 'nod_count': self.nod_count,
                    'shake_count': self.shake_count, 'nod_down': self.nod_down, 'shake_left': self.shake_left}
        if pitch > self.nod_down_thresh: self.nod_down = True
        if self.nod_down and pitch < self.nod_up_thresh:
            self.nod_count += 1; self.nod_down = False
        if yaw < self.shake_left_thresh: self.shake_left = True
        if self.shake_left and yaw > self.shake_right_thresh:
            self.shake_count += 1; self.shake_left = False
        return {'pitch': pitch, 'yaw': yaw, 'nod_count': self.nod_count,
                'shake_count': self.shake_count, 'nod_down': self.nod_down, 'shake_left': self.shake_left}

    def check_nod_confirm(self, pose_pts):
        """Check for a single nod (down->up) as confirmation. Returns True on complete cycle."""
        pitch, _ = self.compute_pitch_yaw(pose_pts)
        if pitch is None: return False
        if pitch > self.nod_down_thresh: self._confirm_nod_down = True
        if self._confirm_nod_down and pitch < self.nod_up_thresh:
            self._confirm_nod_down = False; return True
        return False

    def check_shake_deny(self, pose_pts):
        """Check for a single shake (left->right) as denial. Returns True on complete cycle."""
        _, yaw = self.compute_pitch_yaw(pose_pts)
        if yaw is None: return False
        if yaw < self.shake_left_thresh: self._confirm_shake_left = True
        if self._confirm_shake_left and yaw > self.shake_right_thresh:
            self._confirm_shake_left = False; return True
        return False

    def reset_confirm_state(self):
        self._confirm_nod_down = False; self._confirm_shake_left = False

    def reset_shake(self):
        self.shake_count = 0; self.shake_left = False

    def reset_nod(self):
        self.nod_count = 0; self.nod_down = False

    def reset(self):
        self.nod_count = 0; self.shake_count = 0
        self.nod_down = False; self.shake_left = False
        self.calibrated = False; self.calibrated_neck_shoulder_ratio = None; self.calibration_frames = 0
        self._confirm_nod_down = False; self._confirm_shake_left = False
