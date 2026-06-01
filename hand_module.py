import cv2
import numpy as np
import collections
import time
import mediapipe as mp

# Compatibility: use legacy mp.solutions if available, otherwise use MediaPipe Tasks API
USE_LEGACY = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')

if USE_LEGACY:
    mp_hands = mp.solutions.hands

    class HandDetector:
        def __init__(self):
            self.model = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

        def detect(self, image):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = self.model.process(rgb)
            hands = []
            if not res.multi_hand_landmarks:
                return hands
            h, w = image.shape[:2]
            for hand_landmarks in res.multi_hand_landmarks:
                pts = [(int(lm.x*w), int(lm.y*h), getattr(lm,'z',0.0)) for lm in hand_landmarks.landmark]
                hands.append({'pts': pts})
            return hands

else:
    try:
        from mediapipe.tasks.python import vision as mp_vision
    except Exception:
        raise ImportError('mediapipe tasks API not available')

    class HandDetector:
        def __init__(self):
            base_options = mp.tasks.BaseOptions(model_asset_path=r'models\\hand_landmarker.task')
            options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, running_mode=mp_vision.RunningMode.IMAGE)
            self.detector = mp_vision.HandLandmarker.create_from_options(options)

        def detect(self, image):
            from utils import pad_to_square, map_normalized_landmark_to_original
            from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat
            arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            padded, pad_left, pad_top, pad_w, pad_h, orig_w, orig_h = pad_to_square(arr)
            mp_image = MpImage(ImageFormat.SRGB, padded)
            result = self.detector.detect(mp_image)
            hands = []
            if not result.hand_landmarks:
                return hands
            for hand_landmarks in result.hand_landmarks:
                if hasattr(hand_landmarks, 'landmark'):
                    lm_iter = hand_landmarks.landmark
                else:
                    lm_iter = hand_landmarks
                pts = []
                for lm in lm_iter:
                    x, y = map_normalized_landmark_to_original(lm.x, lm.y, pad_left, pad_top, pad_w, pad_h, orig_w, orig_h)
                    z = getattr(lm,'z',0.0)
                    pts.append((x, y, z))
                hands.append({'pts': pts})
            return hands


class GestureRecognizer:
    """Provides gesture encoding, geometric features, smoothing, ArUco detection, HSV mask and convexity defects."""
    def __init__(self, smooth_len=7):
        self.detector = HandDetector()
        self.history = collections.deque(maxlen=smooth_len)
        # aruco compat: different OpenCV builds expose different APIs
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            # some builds expose class directly
            try:
                self.aruco_params = cv2.aruco.DetectorParameters()
            except Exception:
                self.aruco_params = None
        self.trackers = {}  # id -> tracker

    def analyze(self, frame):
        hands_raw = self.detector.detect(frame)
        hands = []
        for hr in hands_raw:
            pts = hr['pts']
            fingers = self._finger_states(pts)
            code = self._encode_fingers(fingers)
            geom = self._geometry(pts)
            gesture = self._map_gesture(code, geom, pts)
            hands.append({'pts': pts, 'fingers': fingers, 'code': code, 'geom': geom, 'gesture': gesture})
        smoothed = self._smooth(hands)
        ar = self._detect_aruco(frame)
        return {'hands': smoothed, 'aruco': ar}

    def _finger_states(self, pts):
        tips = [4,8,12,16,20]
        pips = [2,6,10,14,18]
        states = {}
        for name,t,p in zip(['thumb','index','middle','ring','pinky'], tips, pips):
            states[name] = pts[t][1] < pts[p][1]
        return states

    def _encode_fingers(self, states):
        order = ['thumb','index','middle','ring','pinky']
        code = 0
        for i,name in enumerate(order):
            if states.get(name, False):
                code |= (1<<i)
        return code

    def _geometry(self, pts):
        # normalize by distance between wrist(0) and index_mcp(5)
        wrist = np.array(pts[0][:2], dtype=float)
        index_base = np.array(pts[5][:2], dtype=float)
        norm = np.linalg.norm(index_base - wrist) + 1e-6
        dists = {i: (np.linalg.norm(np.array(pts[i][:2]) - wrist)/norm) for i in range(len(pts))}
        # direction vector of index finger
        dirv = (np.array(pts[8][:2], dtype=float) - index_base)
        dirv = dirv / (np.linalg.norm(dirv)+1e-6)
        return {'dists': dists, 'dir': dirv.tolist()}

    def _map_gesture(self, code, geom, pts):
        # basic mappings and pinch detection
        if code == 0:
            return 'FIST'
        if code == 31:
            return 'PALM'
        if code == 2:
            return 'POINT_INDEX'
        if code == 6:
            return 'V_SIGN'
        # pinch: thumb tip(4) close to index tip(8)
        tip_thumb = np.array(pts[4][:2], dtype=float)
        tip_index = np.array(pts[8][:2], dtype=float)
        distance = np.linalg.norm(tip_thumb - tip_index) / (np.linalg.norm(np.array(pts[5][:2])-np.array(pts[0][:2]))+1e-6)
        if distance < 0.2:
            return 'PINCH'
        return 'UNKNOWN'

    def _smooth(self, hands):
        if not hands:
            self.history.append(None)
            return []
        g = hands[0]['gesture']
        self.history.append(g)
        # majority
        counts = {}
        for it in self.history:
            counts[it] = counts.get(it,0)+1
        best = max(counts.items(), key=lambda x:x[1])[0]
        hands[0]['gesture_smoothed'] = best
        return hands

    def _detect_aruco(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        out = []
        if ids is None:
            return out
        for c,i in zip(corners, ids.flatten()):
            rect = cv2.boundingRect(c[0].astype(np.int32))
            out.append({'id': int(i), 'rect': rect})
            # tracker creation (CSRT)
            if int(i) not in self.trackers:
                Tr = getattr(cv2, 'TrackerCSRT_create', None) or getattr(cv2.legacy, 'TrackerCSRT_create', None)
                if Tr is not None:
                    tr = Tr()
                    tr.init(frame, tuple(rect))
                    self.trackers[int(i)] = tr
        # update trackers
        for tid in list(self.trackers.keys()):
            tr = self.trackers[tid]
            ok, bbox = tr.update(frame)
            if not ok:
                del self.trackers[tid]
        return out

    def hsv_color_mask(self, frame, roi=None, lower=(0,30,60), upper=(20,255,255)):
        if roi is not None:
            x,y,w,h = roi
            crop = frame[y:y+h, x:x+w]
        else:
            crop = frame
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        # morphological clean
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def convexity_defects_fingers(self, mask):
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c, returnPoints=False)
        if hull is None or len(hull) < 3:
            return 0
        defects = cv2.convexityDefects(c, hull)
        if defects is None:
            return 0
        count = 0
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            if d > 1000:
                count += 1
        return count

# helper
def analyze_frame(frame, smooth_len=7):
    gr = GestureRecognizer(smooth_len=smooth_len)
    return gr.analyze(frame)
