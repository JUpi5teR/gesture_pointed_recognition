import cv2
import numpy as np

class BackgroundModel:
    """
    Single Gaussian background model using MOG2 variant.
    Extracts foreground for target object.
    """

    def __init__(self, roi_pad=30, learning_rate=0.01):
        """
        Initialize background model.
        roi_pad: padding around object box for ROI processing
        learning_rate: how quickly to adapt background model
        """
        self.roi_pad = roi_pad
        self.learning_rate = learning_rate

        # MOG2 with simplified single Gaussian behavior
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            varThreshold=16
        )

        self.initialized = False
        self.frame_count = 0
        self.warmup_frames = 5  # Frames to warm up the model

    def initialize(self, frame):
        """Initialize background model with frame."""
        self.frame_count = 0
        self.initialized = True

        # Warm up: apply to several frames without returning mask
        h, w = frame.shape[:2]
        for _ in range(self.warmup_frames):
            self.bg_subtractor.apply(frame, learningRate=0.5)

    def apply(self, frame, object_box=None):
        """
        Apply background subtraction.
        object_box: (x1, y1, x2, y2) - if provided, extract foreground only in ROI
        Returns: foreground_mask, roi_frame
        """
        if not self.initialized:
            self.initialize(frame)

        self.frame_count += 1

        # Apply background subtraction
        mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Extract ROI if object box provided
        roi_frame = None
        roi_mask = None

        if object_box is not None:
            x1, y1, x2, y2 = object_box
            h, w = frame.shape[:2]

            # Apply padding
            x1_roi = max(0, x1 - self.roi_pad)
            y1_roi = max(0, y1 - self.roi_pad)
            x2_roi = min(w, x2 + self.roi_pad)
            y2_roi = min(h, y2 + self.roi_pad)

            roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi].copy()
            roi_mask = mask[y1_roi:y2_roi, x1_roi:x2_roi].copy()

            # Apply mask to get foreground
            if roi_frame.shape[:2] != roi_mask.shape[:2]:
                roi_mask = cv2.resize(roi_mask, (roi_frame.shape[1], roi_frame.shape[0]))

            foreground = cv2.bitwise_and(roi_frame, roi_frame, mask=roi_mask)

            return mask, foreground, (x1_roi, y1_roi, x2_roi, y2_roi)

        return mask, None, None

    def draw_foreground_box(self, frame, object_box, color=(0, 255, 0), thickness=3):
        """Draw highlighted box around object."""
        if object_box is None:
            return frame

        x1, y1, x2, y2 = object_box
        h, w = frame.shape[:2]

        # Clamp to frame bounds
        x1 = max(0, min(w-1, x1))
        y1 = max(0, min(h-1, y1))
        x2 = max(0, min(w-1, x2))
        y2 = max(0, min(h-1, y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame

    def reset(self):
        """Reset background model."""
        self.initialized = False
        self.frame_count = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            varThreshold=16
        )
