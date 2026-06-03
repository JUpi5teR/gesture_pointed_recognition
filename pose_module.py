import cv2
import numpy as np
import os
import logging
import mediapipe as mp

try:
    from mediapipe.tasks.python import vision as mp_vision
    _HAS_TASKS = True
except Exception:
    _HAS_TASKS = False

logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_SCRIPT_DIR, 'models')

# All possible model filenames (in priority order)
_MODEL_CANDIDATES = [
    'pose_landmarker.task',
    'pose_landmarker_heavy.task',
    'pose_landmarker_full.task',
    'pose_landmarker_lite.task',
]


class PoseDetector:
    """
    Pose landmark detector using MediaPipe PoseLandmarker (Tasks API).
    Returns list of (x, y, z) tuples for 33 pose landmarks.
    Key indices: 0=nose, 11=left shoulder, 12=right shoulder.
    """

    def __init__(self, model_path=None, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        if not _HAS_TASKS:
            raise ImportError('mediapipe tasks API not available')

        if model_path is None:
            # Auto-find model file
            model_path = None
            for name in _MODEL_CANDIDATES:
                candidate = os.path.join(_MODEL_DIR, name)
                if os.path.isfile(candidate):
                    model_path = candidate
                    break

            if model_path is None:
                raise FileNotFoundError(
                    f"No pose_landmarker model found in {_MODEL_DIR}\n"
                    f"Searched for: {', '.join(_MODEL_CANDIDATES)}\n"
                    f"Please download from:\n"
                    f"  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task\n"
                    f"Save to: {os.path.join(_MODEL_DIR, 'pose_landmarker_heavy.task')}"
                )

        logger.info('Loading pose model: %s', model_path)
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = mp_vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        logger.info('PoseDetector ready.')

    def detect(self, image):
        """
        Detect pose landmarks in image.
        Returns list of (x, y, z) tuples for 33 landmarks, or None if no pose detected.
        """
        from utils import pad_to_square, map_normalized_landmark_to_original
        from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat

        arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padded, pad_left, pad_top, pad_w, pad_h, orig_w, orig_h = pad_to_square(arr)

        try:
            mp_image = MpImage(ImageFormat.SRGB, padded, padded.shape[1], padded.shape[0])
        except TypeError:
            mp_image = MpImage(ImageFormat.SRGB, padded)

        self._timestamp_ms += 33
        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)

        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0]
        lm_iter = lm.landmark if hasattr(lm, 'landmark') else lm

        pts = []
        for p in lm_iter:
            x, y = map_normalized_landmark_to_original(
                p.x, p.y, pad_left, pad_top, pad_w, pad_h, orig_w, orig_h
            )
            z = getattr(p, 'z', 0.0)
            pts.append((x, y, z))
        return pts
