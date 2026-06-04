import math
import numpy as np
from enum import Enum
from config import (DWELL_TRIGGER_SEC, FPS, TARGET_ANGLE_THRESH, TARGET_DIST_COEFF,
                    DWELL_DECAY_RATE, DWELL_TOLERANCE_SEC, DWELL_MIN_HOLD_RATIO)


class SystemState(Enum):
    """State machine states for the hand-target-track-shake workflow."""
    IDLE = "IDLE"
    DWELL_WAIT = "DWELL_WAIT"
    CONFIRMING = "CONFIRMING"
    TRACKING = "TRACKING"
    UNLOCKING = "UNLOCKING"


def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float); v2 = np.array(v2, dtype=float)
    nv1 = v1 / (np.linalg.norm(v1)+1e-6); nv2 = v2 / (np.linalg.norm(v2)+1e-6)
    return math.degrees(math.acos(np.clip(np.dot(nv1, nv2), -1.0, 1.0)))

def sample_depth(depth_img, u, v, win=6):
    h, w = depth_img.shape[:2]
    u0, v0 = int(max(0,min(w-1,int(u)))), int(max(0,min(h-1,int(v))))
    patch = depth_img[max(0,v0-win):min(h-1,v0+win)+1, max(0,u0-win):min(w-1,u0+win)+1]
    if patch.size == 0: return 0
    vals = patch.flatten(); vals = vals[vals > 0]
    return int(np.median(vals)) if vals.size > 0 else 0

def pixel_to_point(depth_mm, u, v, intrinsics):
    if depth_mm == 0: return None
    Z = depth_mm / 1000.0
    return ((u-intrinsics['cx'])*Z/intrinsics['fx'], (v-intrinsics['cy'])*Z/intrinsics['fy'], Z)

def point_to_pixel(X, Y, Z, intrinsics):
    if Z == 0: return None
    return (intrinsics['fx']*X/Z+intrinsics['cx'], intrinsics['fy']*Y/Z+intrinsics['cy'])

def pad_to_square(img, color=(0,0,0)):
    h, w = img.shape[:2]; size = max(h, w)
    pt, pl = (size-h)//2, (size-w)//2
    padded = np.full((size, size, img.shape[2] if img.ndim==3 else 1), color if isinstance(color,tuple) else color, dtype=img.dtype)
    if img.ndim == 2:
        padded[pt:pt+h, pl:pl+w] = img
    else:
        padded[pt:pt+h, pl:pl+w, :] = img
    return padded, pl, pt, size, size, w, h

def map_normalized_landmark_to_original(lm_x, lm_y, pad_left, pad_top, pad_w, pad_h, orig_w, orig_h):
    return (max(0,min(orig_w-1,int(round(lm_x*pad_w-pad_left)))),
            max(0,min(orig_h-1,int(round(lm_y*pad_h-pad_top)))))

def ray_intersect_depth(origin_3d, dir_3d, depth_img, intrinsics, z_step=0.02, z_max=3.0, thresh_mm=80):
    d = np.array(dir_3d, dtype=float); d /= (np.linalg.norm(d)+1e-9)
    ox, oy, oz = origin_3d
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    h, w = depth_img.shape[:2]
    ss = z_step*5; candidates = []; z = ss
    while z <= z_max:
        X,Y,Z = ox+d[0]*z, oy+d[1]*z, oz+d[2]*z
        if Z <= 0: z += ss; continue
        u, v = fx*X/Z+cx, fy*Y/Z+cy
        if u<0 or u>=w or v<0 or v>=h: z += ss; continue
        dm = sample_depth(depth_img, u, v, win=2)
        if dm > 0 and abs(dm-Z*1000) <= thresh_mm: candidates.append((z,u,v,Z,dm))
        z += ss
    if not candidates: return None
    zmin = max(z_step, candidates[0][0]-ss); zmax = min(z_max, candidates[0][0]+ss); z = zmin
    while z <= zmax:
        X,Y,Z = ox+d[0]*z, oy+d[1]*z, oz+d[2]*z
        if Z <= 0: z += z_step; continue
        u, v = fx*X/Z+cx, fy*Y/Z+cy
        if u<0 or u>=w or v<0 or v>=h: z += z_step; continue
        dm = sample_depth(depth_img, u, v, win=2)
        if dm > 0 and abs(dm-Z*1000) <= thresh_mm*0.5: return (u,v,Z)
        z += z_step
    b = min(candidates, key=lambda c: c[0]); return (b[1],b[2],b[3])

def choose_target_2d(hands, dets, angle_thresh_deg=None, dist_coeff=None):
    atd = angle_thresh_deg or TARGET_ANGLE_THRESH; dc = dist_coeff or TARGET_DIST_COEFF
    if not hands or not dets: return None
    tip = np.array(hands[0]['pts'][8][:2], dtype=float); base = np.array(hands[0]['pts'][5][:2], dtype=float)
    dv = tip - base; n = np.linalg.norm(dv)
    if n < 1e-3: return None
    dv /= n; best_j = None; best_s = 0.0
    for j, det in enumerate(dets):
        cx, cy = bbox_center(det['box']); vec = np.array([cx-tip[0],cy-tip[1]], dtype=float)
        dist = np.linalg.norm(vec)
        if dist < 1.0: continue
        f = vec/dist; ac = max(-1.0,min(1.0,float(np.dot(dv,f)))); ad = math.degrees(math.acos(ac))
        if ad > atd: continue
        perp = abs(f[0]*vec[1]-f[1]*vec[0]); x1,y1,x2,y2 = det['box']
        s = max(0,ac)*max(0,1-perp/(math.hypot(abs(x2-x1),abs(y2-y1))*dc))
        if s > best_s: best_s = s; best_j = j
    return best_j if best_s > 0 else None

def point_in_box(point, box):
    x,y = point; x1,y1,x2,y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def compute_target_expectation(target_points, window_size=10):
    if not target_points or len(target_points) < 3: return None
    arr = np.array(target_points[-window_size:], dtype=float)
    mean = np.mean(arr, axis=0); std = np.std(arr, axis=0)
    if std[0]>0 and std[1]>0:
        f = arr[(np.abs(arr[:,0]-mean[0])<=2*std[0])&(np.abs(arr[:,1]-mean[1])<=2*std[1])]
        if len(f) > 0: mean = np.mean(f, axis=0)
    return tuple(mean)


class TargetDwellTracker:
    """Track how long target point dwells in each detection region.
    
    Features:
    - Tolerance window: brief leave (< DWELL_TOLERANCE_SEC) keeps count unchanged.
    - Gradual decay: after tolerance, count decays by DWELL_DECAY_RATE per frame
      instead of instant reset. 0.5 = retain 50% each frame.
    - Minimum hold: count only resets to 0 when it drops below
      DWELL_MIN_HOLD_RATIO * threshold.
    """
    def __init__(self, dwell_trigger_sec=None, fps=None,
                 decay_rate=None, tolerance_sec=None, min_hold_ratio=None):
        self.dwell_trigger_sec = dwell_trigger_sec or DWELL_TRIGGER_SEC
        self.fps = fps or FPS
        self.dwell_threshold_frames = int(self.dwell_trigger_sec * self.fps)
        self.decay_rate = decay_rate if decay_rate is not None else DWELL_DECAY_RATE
        self.tolerance_frames = int((tolerance_sec if tolerance_sec is not None else DWELL_TOLERANCE_SEC) * self.fps)
        self.min_hold_frames = int((min_hold_ratio if min_hold_ratio is not None else DWELL_MIN_HOLD_RATIO) * self.dwell_threshold_frames)

        # Per-region state
        self.region_dwell = {}        # region_id -> dwell count (float for decay)
        self.region_points = {}       # region_id -> list of target points
        self.region_leave = {}        # region_id -> frames since target left
        self.triggered_target = None
        self.triggered_box = None
        self.is_triggered = False

    def update(self, target_pt, dets):
        if target_pt is None or not dets:
            self._decay_all()
            return self.triggered_target, self.triggered_box

        # Find which region the target is currently in
        cur = None
        for i, det in enumerate(dets):
            if point_in_box(target_pt, det['box']):
                cur = i; break

        for i in range(len(dets)):
            if i == cur:
                # Target is IN this region: increment count, reset leave counter
                self.region_dwell[i] = self.region_dwell.get(i, 0) + 1
                self.region_leave.pop(i, None)
                self.region_points.setdefault(i, []).append(target_pt)
            else:
                # Target is NOT in this region: apply tolerance + decay
                if i in self.region_dwell:
                    leave_count = self.region_leave.get(i, 0) + 1
                    self.region_leave[i] = leave_count

                    if leave_count <= self.tolerance_frames:
                        # Within tolerance window: keep count unchanged (no increment, no decay)
                        pass
                    else:
                        # Beyond tolerance: decay the count
                        self.region_dwell[i] *= self.decay_rate
                        # If decayed below minimum hold, remove this region's count entirely
                        if self.region_dwell[i] < self.min_hold_frames:
                            self.region_dwell.pop(i, None)
                            self.region_points.pop(i, None)
                            self.region_leave.pop(i, None)

        # Check if any region reached threshold
        for i, cnt in self.region_dwell.items():
            if cnt >= self.dwell_threshold_frames and not self.is_triggered:
                self.triggered_target = compute_target_expectation(self.region_points[i], window_size=20)
                self.triggered_box = tuple(dets[i]['box'])
                self.is_triggered = True
                return self.triggered_target, self.triggered_box

        return self.triggered_target, self.triggered_box

    def _decay_all(self):
        """Decay all regions when target_pt is None or no detections."""
        to_remove = []
        for i in list(self.region_dwell.keys()):
            leave_count = self.region_leave.get(i, 0) + 1
            self.region_leave[i] = leave_count
            if leave_count > self.tolerance_frames:
                self.region_dwell[i] *= self.decay_rate
                if self.region_dwell[i] < self.min_hold_frames:
                    to_remove.append(i)
        for i in to_remove:
            self.region_dwell.pop(i, None)
            self.region_points.pop(i, None)
            self.region_leave.pop(i, None)

    def get_dwell_progress(self):
        if not self.region_dwell: return 0.0
        return min(1.0, max(self.region_dwell.values()) / self.dwell_threshold_frames)

    def reset(self):
        self.triggered_target = None; self.triggered_box = None; self.is_triggered = False
        self.region_dwell.clear(); self.region_points.clear(); self.region_leave.clear()
