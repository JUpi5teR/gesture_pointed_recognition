import cv2
import numpy as np
import mediapipe as mp

# Compatibility for FaceMesh
USE_LEGACY = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh')

if USE_LEGACY:
    mp_face = mp.solutions.face_mesh

    class FaceExpression:
        def __init__(self):
            self.model = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                          refine_landmarks=True, min_detection_confidence=0.5)

        def detect(self, image):
            # image: BGR
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = self.model.process(rgb)
            if not res.multi_face_landmarks:
                return None, None
            lm = res.multi_face_landmarks[0]
            h, w = image.shape[:2]
            pts = [(int(p.x*w), int(p.y*h)) for p in lm.landmark]
            expr = self._heuristic_expression(pts)
            return expr, pts

        def _heuristic_expression(self, pts):
            left_cheek = pts[234]
            right_cheek = pts[454]
            face_width = np.linalg.norm(np.array(left_cheek)-np.array(right_cheek)) + 1e-6
            mouth_top = np.array(pts[13])
            mouth_bottom = np.array(pts[14])
            mouth_open = np.linalg.norm(mouth_top-mouth_bottom)/face_width
            left_corner = np.array(pts[61])
            right_corner = np.array(pts[291])
            smile = np.linalg.norm(left_corner-right_corner)/face_width
            if mouth_open > 0.05:
                return 'mouth_open'
            if smile > 0.36:
                return 'smile'
            return 'neutral'

else:
    try:
        from mediapipe.tasks.python import vision as mp_vision
    except Exception:
        raise ImportError('mediapipe installed but tasks API not available; please install a compatible mediapipe or contact developer')

    class FaceExpression:
        def __init__(self):
            try:
                base_options = mp.tasks.BaseOptions(model_asset_path=r'models\\face_landmarker.task')
                options = mp_vision.FaceLandmarkerOptions(base_options=base_options, running_mode=mp_vision.RunningMode.IMAGE)
                self.detector = mp_vision.FaceLandmarker.create_from_options(options)
            except Exception:
                try:
                    options = mp_vision.FaceLandmarkerOptions(running_mode=mp_vision.RunningMode.IMAGE)
                    self.detector = mp_vision.FaceLandmarker.create_from_options(options)
                except Exception as e:
                    raise RuntimeError('Failed to create FaceLandmarker with MediaPipe Tasks API: %s' % e)

        def detect(self, image):
            # Construct MediaPipe Image from numpy array with square padding for stable projection
            from utils import pad_to_square, map_normalized_landmark_to_original
            from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat
            arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            padded, pad_left, pad_top, pad_w, pad_h, orig_w, orig_h = pad_to_square(arr)
            mp_image = MpImage(ImageFormat.SRGB, padded)
            res = self.detector.detect(mp_image)
            if not res.face_landmarks:
                return None, None
            lm = res.face_landmarks[0]
            # Support both proto object with .landmark and raw list of landmarks
            if hasattr(lm, 'landmark'):
                landmark_iter = lm.landmark
            else:
                landmark_iter = lm
            pts = []
            for p in landmark_iter:
                x,y = map_normalized_landmark_to_original(p.x, p.y, pad_left, pad_top, pad_w, pad_h, orig_w, orig_h)
                pts.append((x,y))
            expr = self._heuristic_expression(pts)
            return expr, pts

        def _heuristic_expression(self, pts):
            left_cheek = pts[234]
            right_cheek = pts[454]
            face_width = np.linalg.norm(np.array(left_cheek)-np.array(right_cheek)) + 1e-6
            mouth_top = np.array(pts[13])
            mouth_bottom = np.array(pts[14])
            mouth_open = np.linalg.norm(mouth_top-mouth_bottom)/face_width
            left_corner = np.array(pts[61])
            right_corner = np.array(pts[291])
            smile = np.linalg.norm(left_corner-right_corner)/face_width
            if mouth_open > 0.05:
                return 'mouth_open'
            if smile > 0.36:
                return 'smile'
            return 'neutral'
