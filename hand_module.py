import cv2
import numpy as np
import collections
import time
import math
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
                hands.append({'pts': pts, 'landmarks': hand_landmarks})
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
            try:
                mp_image = MpImage(ImageFormat.SRGB, padded, padded.shape[1], padded.shape[0])
            except TypeError:
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
        self.prev_dir_vec = None
        self.prev_tip_pos = None
        self.keypoint_cache = collections.deque(maxlen=8)
        self.current_landmarks = None

    def analyze(self, frame):
        hands_raw = self.detector.detect(frame)
        hands = []
        for hr in hands_raw:
            pts = hr['pts']
            # Store landmarks for distance calculations
            self.current_landmarks = hr.get('landmarks')
            fingers = self._finger_states(pts)
            code = self._encode_fingers(fingers)
            geom = self._geometry(pts)
            gesture = self._map_gesture(code, geom, pts)
            hands.append({'pts': pts, 'fingers': fingers, 'code': code, 'geom': geom, 'gesture': gesture})
        smoothed = self._smooth(hands)
        ar = self._detect_aruco(frame)
        return {'hands': smoothed, 'aruco': ar}

    def _get_signed_dist(self, point_indices):
        """
        Calculate signed euclidean distance between two landmark points.
        point_indices: [idx1, idx2]
        Returns distance * sign based on y-axis difference.
        """
        if not self.current_landmarks:
            return 0
        lm = self.current_landmarks.landmark
        sign = -1
        if lm[point_indices[0]].y < lm[point_indices[1]].y:
            sign = 1
        dx = lm[point_indices[0]].x - lm[point_indices[1]].x
        dy = lm[point_indices[0]].y - lm[point_indices[1]].y
        dist = math.sqrt(dx*dx + dy*dy)
        return dist * sign

    def _get_dist(self, point_indices):
        """Calculate euclidean distance between two landmark points."""
        if not self.current_landmarks:
            return 0
        lm = self.current_landmarks.landmark
        dx = lm[point_indices[0]].x - lm[point_indices[1]].x
        dy = lm[point_indices[0]].y - lm[point_indices[1]].y
        return math.sqrt(dx*dx + dy*dy)

    def _get_dz(self, point_indices):
        """Calculate absolute z-axis difference between two landmark points."""
        if not self.current_landmarks:
            return 0
        lm = self.current_landmarks.landmark
        return abs(lm[point_indices[0]].z - lm[point_indices[1]].z)

    def _finger_states(self, pts):
        """
        Determine finger states based on distance ratios between keypoints.
        Uses skeletal landmark ratios similar to Gesture_Controller approach.
        Returns: dict with finger names as keys and open/closed states as boolean values.
        """
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]
        mcps = [5, 9, 13, 17]

        states = {}
        names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        # For thumb (tip 4, pip 2)
        tip_thumb = np.array(pts[4][:2], dtype=float)
        pip_thumb = np.array(pts[2][:2], dtype=float)
        wrist = np.array(pts[0][:2], dtype=float)

        thumb_dist_tip_pip = np.linalg.norm(tip_thumb - pip_thumb)
        thumb_dist_pip_wrist = np.linalg.norm(pip_thumb - wrist)

        try:
            thumb_ratio = thumb_dist_tip_pip / (thumb_dist_pip_wrist + 1e-6)
            states['thumb'] = thumb_ratio > 0.5
        except:
            states['thumb'] = False

        # For other four fingers use landmark distance ratios
        for i, (tip_idx, mcp_idx, pip_idx, name) in enumerate(
            zip([8, 12, 16, 20], [5, 9, 13, 17], [6, 10, 14, 18], names[1:])
        ):
            try:
                # Calculate signed distances like in Gesture_Controller
                dist_tip_mcp = self._get_signed_dist([tip_idx, mcp_idx])
                dist_mcp_pip = self._get_signed_dist([mcp_idx, pip_idx])

                ratio = round(dist_tip_mcp / (dist_mcp_pip + 1e-6), 1)
                states[name] = ratio > 0.5
            except:
                states[name] = False

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

    def detect_pointing_direction_keypoint(self, hands, depth_img, intrinsics, use_pca=False):
        """
        Detect pointing direction using MediaPipe keypoints (8=tip, 5=MCP or 6=PIP).
        Replaces contour-based method with direct keypoint vector calculation.
        Returns (tip_3d, dir_3d) in 3D space (meters), or (None, None) if detection fails.
        """
        if not hands or depth_img is None or intrinsics is None:
            return None, None

        from utils import sample_depth, pixel_to_point

        h = hands[0]
        pts = h['pts']

        tip_px = pts[8][:2]
        mcp_px = pts[5][:2]

        tip_d = sample_depth(depth_img, tip_px[0], tip_px[1], win=6)
        mcp_d = sample_depth(depth_img, mcp_px[0], mcp_px[1], win=6)

        if tip_d == 0 or mcp_d == 0:
            return None, None

        tip_3d = pixel_to_point(tip_d, tip_px[0], tip_px[1], intrinsics)
        mcp_3d = pixel_to_point(mcp_d, mcp_px[0], mcp_px[1], intrinsics)

        if tip_3d is None or mcp_3d is None:
            return None, None

        dir_3d = np.array([tip_3d[0]-mcp_3d[0], tip_3d[1]-mcp_3d[1], tip_3d[2]-mcp_3d[2]], dtype=float)
        dir_norm = np.linalg.norm(dir_3d)
        if dir_norm < 1e-6:
            return None, None
        dir_3d = dir_3d / dir_norm

        # Cache keypoints for PCA if enabled
        if use_pca:
            pip_px = pts[6][:2]
            pip_d = sample_depth(depth_img, pip_px[0], pip_px[1], win=6)
            if pip_d > 0:
                pip_3d = pixel_to_point(pip_d, pip_px[0], pip_px[1], intrinsics)
                if pip_3d:
                    self.keypoint_cache.append((tip_3d, pip_3d, mcp_3d))

        tip_pos_smoothed = tip_3d
        dir_vec_smoothed = tuple(dir_3d)

        # Apply exponential smoothing if previous state exists
        if self.prev_tip_pos is not None:
            alpha = 0.3
            tip_pos_smoothed = tuple(
                0.7 * self.prev_tip_pos[i] + 0.3 * tip_3d[i] for i in range(3)
            )

        if self.prev_dir_vec is not None:
            alpha = 0.3
            smoothed_dir = np.array(self.prev_dir_vec) * 0.7 + np.array(dir_3d) * 0.3
            smoothed_dir = smoothed_dir / (np.linalg.norm(smoothed_dir) + 1e-9)
            dir_vec_smoothed = tuple(smoothed_dir)

        self.prev_tip_pos = tip_pos_smoothed
        self.prev_dir_vec = dir_vec_smoothed

        return tip_pos_smoothed, dir_vec_smoothed

    def fit_3d_line_pca(self):
        """
        Fit a 3D line through cached keypoints using PCA.
        Returns (origin, direction) tuple or (None, None) if insufficient data.
        """
        if len(self.keypoint_cache) < 5:
            return None, None

        all_pts = []
        for tip, pip, mcp in self.keypoint_cache:
            all_pts.append(tip)
            all_pts.append(pip)
            all_pts.append(mcp)

        pts_array = np.array(all_pts, dtype=float)
        centroid = pts_array.mean(axis=0)

        cov = np.cov(pts_array.T)
        w, v = np.linalg.eigh(cov)
        idx = np.argmax(w)
        direction = v[:, idx]
        direction = direction / (np.linalg.norm(direction) + 1e-9)

        return tuple(centroid), tuple(direction)

    def detect_pointing_direction(self, frame, roi=None, D=7, Step=70, alpha0=30.0, beta0=20.0, theta0=30.0):
        """
        Detect pointing finger tip and direction using contour-based algorithm.
        Returns (tip_pt (x,y), dir_vec (dx,dy)) in image pixel coordinates or (None,None).
        """
        # get mask
        if roi is not None:
            x,y,w,h = roi
            crop = frame[y:y+h, x:x+w]
            mask = self.hsv_color_mask(frame, roi=roi)
            off_x, off_y = x, y
        else:
            mask = self.hsv_color_mask(frame)
            crop = frame
            off_x, off_y = 0,0
        # find largest contour
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None, None
        c = max(cnts, key=cv2.contourArea)
        if c.shape[0] < 10:
            return None, None
        # contour points as Nx2
        contour = c.reshape(-1,2)
        # 1) DP-like sampling with Step and D
        sampled = self._contour_sample_dp(contour, Step=Step, D=D)
        if len(sampled) < 6:
            return None, None
        # 2) smoothing by segment adaptive quadratic Bezier (fast)
        smooth = self._smooth_contour_bezier(sampled, seg_steps=5)
        # 3) estimate fingertip candidates: extreme points (local maxima of distance from contour centroid)
        cx,cy = np.mean(smooth, axis=0)
        dists = np.linalg.norm(smooth - np.array([cx,cy]), axis=1)
        tip_idx = int(np.argmax(dists))
        tip_pt = tuple((smooth[tip_idx] + np.array([off_x, off_y])).astype(int))
        # 4) extract local arc around tip along smooth contour
        arc_radius = max(10, int(len(smooth)*0.06))
        n = len(smooth)
        arc_indices = [(tip_idx + i) % n for i in range(-arc_radius, arc_radius+1)]
        arc_pts = smooth[arc_indices]
        # 5) split arc into left/right halves relative to tip, fit lines via clustering by angle
        mid = len(arc_pts)//2
        left_pts = arc_pts[:mid]
        right_pts = arc_pts[mid:]
        if len(left_pts) < 3 or len(right_pts) < 3:
            return tip_pt, None
        L_line = self._line_from_points_least_squares(left_pts)
        R_line = self._line_from_points_least_squares(right_pts)
        if L_line is None or R_line is None:
            return tip_pt, None
        # compute direction bisector (image coordinates: x right, y down)
        vL = np.array([L_line[0], L_line[1]])
        vR = np.array([R_line[0], R_line[1]])
        vLn = vL / (np.linalg.norm(vL)+1e-9)
        vRn = vR / (np.linalg.norm(vR)+1e-9)
        bis = vLn + vRn
        if np.linalg.norm(bis) < 1e-6:
            bis = vLn - vRn
        bis = bis / (np.linalg.norm(bis)+1e-9)
        # ensure bisector points away from hand centroid: if dot((tip-centroid),bis) <0 flip
        vec_tip_cent = np.array(tip_pt) - np.array([cx+off_x, cy+off_y])
        if np.dot(vec_tip_cent, bis) < 0:
            bis = -bis
        dir_vec = (float(bis[0]), float(bis[1]))
        return tip_pt, dir_vec

    def _contour_sample_dp(self, contour, Step=70, D=7):
        """
        Sample contour with step-wise max-deviation rule: for each segment of length Step, if max deviation < D choose endpoint, else choose deviation-max point.
        contour: Nx2 numpy array
        returns list of sampled points as Nx2 numpy array
        """
        n = len(contour)
        if n == 0:
            return np.array([])
        sampled = []
        i = 0
        while i < n:
            j = (i + Step) % n
            # handle wrap properly by constructing segment point list
            if i < j:
                seg = contour[i:j+1]
            else:
                seg = np.vstack((contour[i:], contour[:j+1]))
            p0 = seg[0]
            p1 = seg[-1]
            # line distance
            if len(seg) <= 2:
                chosen = p1
            else:
                # compute perpendicular distances
                v = p1 - p0
                vnorm = v / (np.linalg.norm(v)+1e-9)
                rel = seg - p0
                proj_len = np.dot(rel, vnorm)
                proj = np.outer(proj_len, vnorm) + p0
                dists = np.linalg.norm(seg - proj, axis=1)
                idx_max = int(np.argmax(dists))
                if dists[idx_max] < D:
                    chosen = p1
                else:
                    chosen = seg[idx_max]
            sampled.append(chosen)
            i = (i + Step) % n
            if len(sampled) > n:
                break
        sampled = np.array(sampled)
        # ensure unique in order
        # remove near-duplicates
        out = [sampled[0]]
        for p in sampled[1:]:
            if np.linalg.norm(p - out[-1]) > 1.0:
                out.append(p)
        return np.array(out)

    def _smooth_contour_bezier(self, pts, seg_steps=5):
        """
        Fast quadratic Bezier smoothing over consecutive triplets.
        pts: Nx2 array of sampled contour points
        seg_steps: samples per segment
        returns Mx2 float array
        """
        if len(pts) < 3:
            return pts.copy()
        out = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i+1)%n]
            p2 = pts[(i+2)%n]
            # control point for quadratic Bezier: approximate using p1
            for t in np.linspace(0.0,1.0,seg_steps,endpoint=False):
                b = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
                out.append(b)
        return np.array(out)

    def _line_from_points_least_squares(self, pts):
        """Fit line y = ax + b in least squares in parametric form; return unit direction (dx,dy) and a point on line (px,py).
        pts: Nx2 array
        returns (dx,dy,(px,py)) or None
        """
        if len(pts) < 2:
            return None
        pts = np.array(pts, dtype=float)
        # centroid
        cx,cy = pts.mean(axis=0)
        # covariance
        cov = np.cov(pts.T)
        # principal component
        w,v = np.linalg.eigh(cov)
        idx = np.argmax(w)
        dirv = v[:,idx]
        if np.linalg.norm(dirv) < 1e-6:
            return None
        dirv = dirv / np.linalg.norm(dirv)
        return (float(dirv[0]), float(dirv[1]), (float(cx), float(cy)))

# helper
def analyze_frame(frame, smooth_len=7):
    gr = GestureRecognizer(smooth_len=smooth_len)
    return gr.analyze(frame)
