import cv2
import numpy as np
import mediapipe as mp

# Compatibility: use legacy mp.solutions if available, otherwise try MediaPipe Tasks (new API)
USE_LEGACY = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')

if USE_LEGACY:
    mp_hands = mp.solutions.hands

    class HandDetector:
        def __init__(self):
            self.model = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

        def detect(self, image):
            # image: BGR
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = self.model.process(rgb)
            hands = []
            if not res.multi_hand_landmarks:
                return hands
            h, w = image.shape[:2]
            for hand_landmarks in res.multi_hand_landmarks:
                pts = [(int(lm.x*w), int(lm.y*h), lm.z) for lm in hand_landmarks.landmark]
                fingers = self._finger_states(pts)
                index_dir = self._index_direction(pts)
                hands.append({"pts": pts, "fingers": fingers, "index_dir": index_dir})
            return hands

        def _finger_states(self, pts):
            tips = [4, 8, 12, 16, 20]
            pips = [2, 6, 10, 14, 18]
            states = {}
            for name, t, p in zip(["thumb","index","middle","ring","pinky"], tips, pips):
                states[name] = pts[t][1] < pts[p][1]
            return states

        def _index_direction(self, pts):
            base = np.array(pts[5][:2], dtype=float)
            tip = np.array(pts[8][:2], dtype=float)
            vec = tip - base
            norm = np.linalg.norm(vec) + 1e-6
            return (vec / norm).tolist()

else:
    # New MediaPipe Tasks API
    try:
        from mediapipe.tasks.python import vision as mp_vision
    except Exception:
        raise ImportError('mediapipe installed but tasks API not available; please install a compatible mediapipe or contact developer')

    class HandDetector:
        def __init__(self):
            # Create HandLandmarker using default packaged model if available
            try:
                base_options = mp.tasks.BaseOptions(model_asset_path=r'models\\hand_landmarker.task')
                options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, running_mode=mp_vision.RunningMode.IMAGE)
                self.detector = mp_vision.HandLandmarker.create_from_options(options)
            except Exception:
                # Fall back to creating without explicit model path; may still fail
                try:
                    options = mp_vision.HandLandmarkerOptions(num_hands=2, running_mode=mp_vision.RunningMode.IMAGE)
                    self.detector = mp_vision.HandLandmarker.create_from_options(options)
                except Exception as e:
                    raise RuntimeError('Failed to create HandLandmarker with MediaPipe Tasks API: %s' % e)

        def detect(self, image):
            # image: BGR numpy array
            from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat
            arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = MpImage(ImageFormat.SRGB, arr)
            result = self.detector.detect(mp_image)
            hands = []
            if not result.hand_landmarks:
                return hands
            h, w = image.shape[:2]
            for hand_landmarks in result.hand_landmarks:
                pts = []
                # handle both proto and list
                if hasattr(hand_landmarks, 'landmark'):
                    lm_iter = hand_landmarks.landmark
                else:
                    lm_iter = hand_landmarks
                for lm in lm_iter:
                    # landmarks in normalized coordinates
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    z = getattr(lm, 'z', 0.0)
                    pts.append((x, y, z))
                fingers = self._finger_states(pts)
                index_dir = self._index_direction(pts)
                hands.append({"pts": pts, "fingers": fingers, "index_dir": index_dir})
            return hands

        def _finger_states(self, pts):
            tips = [4, 8, 12, 16, 20]
            pips = [2, 6, 10, 14, 18]
            states = {}
            for name, t, p in zip(["thumb","index","middle","ring","pinky"], tips, pips):
                states[name] = pts[t][1] < pts[p][1]
            return states

        def _index_direction(self, pts):
            base = np.array(pts[5][:2], dtype=float)
            tip = np.array(pts[8][:2], dtype=float)
            vec = tip - base
            norm = np.linalg.norm(vec) + 1e-6
            return (vec / norm).tolist()
