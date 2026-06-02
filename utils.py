import math
import numpy as np

def bbox_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    nv1 = v1 / (np.linalg.norm(v1)+1e-6)
    nv2 = v2 / (np.linalg.norm(v2)+1e-6)
    cos = np.clip(np.dot(nv1, nv2), -1.0, 1.0)
    return math.degrees(math.acos(cos))


def sample_depth(depth_img, u, v, win=6):
    """Sample depth image (uint16 mm) around (u,v) and return median of non-zero samples in mm.
    Returns 0 if no valid depth. Default win=6 for improved noise reduction.
    """
    h, w = depth_img.shape[:2]
    u0 = int(max(0, min(w-1, int(u))))
    v0 = int(max(0, min(h-1, int(v))))
    x1 = max(0, u0 - win)
    x2 = min(w-1, u0 + win)
    y1 = max(0, v0 - win)
    y2 = min(h-1, v0 + win)
    patch = depth_img[y1:y2+1, x1:x2+1]
    if patch.size == 0:
        return 0
    vals = patch.flatten()
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0
    return int(np.median(vals))


def pixel_to_point(depth_mm, u, v, intrinsics):
    """Backproject pixel (u,v) with depth in mm to 3D point in meters using intrinsics dict{'fx','fy','cx','cy'}.
    Returns (X,Y,Z) in meters.
    """
    if depth_mm == 0:
        return None
    Z = depth_mm / 1000.0
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return (X, Y, Z)


def point_to_pixel(X, Y, Z, intrinsics):
    """Project 3D point in meters to pixel coordinates (u,v) using intrinsics.
    Returns (u,v) floats.
    """
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    if Z == 0:
        return None
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return (u, v)


def pad_to_square(img, color=(0,0,0)):
    """Pad image (H,W,...) to square by adding borders. Returns (padded_img, pad_x, pad_y, new_w, new_h, orig_w, orig_h).
    pad_x = number of pixels added on left, pad_y on top.
    """
    h, w = img.shape[:2]
    size = max(h, w)
    pad_vert = size - h
    pad_horiz = size - w
    pad_top = pad_vert // 2
    pad_bottom = pad_vert - pad_top
    pad_left = pad_horiz // 2
    pad_right = pad_horiz - pad_left
    if img.ndim == 2:
        padded = np.full((size, size), color[0] if isinstance(color, tuple) else color, dtype=img.dtype)
        padded[pad_top:pad_top+h, pad_left:pad_left+w] = img
    else:
        padded = np.full((size, size, img.shape[2]), color, dtype=img.dtype)
        padded[pad_top:pad_top+h, pad_left:pad_left+w, :] = img
    return padded, pad_left, pad_top, size, size, w, h


def map_normalized_landmark_to_original(lm_x, lm_y, pad_left, pad_top, pad_w, pad_h, orig_w, orig_h):
    """Map normalized landmark coords (0..1 relative to padded image) to original image pixel coords.
    Returns (x,y) clamped to original image bounds.
    """
    x_pad = lm_x * pad_w
    y_pad = lm_y * pad_h
    x = int(round(x_pad - pad_left))
    y = int(round(y_pad - pad_top))
    x = max(0, min(orig_w-1, x))
    y = max(0, min(orig_h-1, y))
    return x, y


def ray_intersect_depth(origin_3d, dir_3d, depth_img, intrinsics, z_step=0.02, z_max=3.0, thresh_mm=80):
    """Cast a ray from origin_3d (meters) along dir_3d (unit vector) and find intersection with depth map.
    Uses sparse sampling + bilinear interpolation for efficiency instead of dense iteration.
    depth_img is in millimeters and aligned to color.
    Returns (u,v,Z_m) if found, else None.
    """
    d = np.array(dir_3d, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-9)
    ox, oy, oz = origin_3d
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    h, w = depth_img.shape[:2]

    # Sparse sampling: test at larger intervals to find rough candidate
    sparse_step = z_step * 5
    candidates = []
    z = sparse_step
    while z <= z_max:
        X = ox + d[0] * z
        Y = oy + d[1] * z
        Z = oz + d[2] * z
        if Z <= 0:
            z += sparse_step
            continue
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        if u < 0 or u >= w or v < 0 or v >= h:
            z += sparse_step
            continue
        depth_mm = sample_depth(depth_img, u, v, win=2)
        if depth_mm > 0:
            Z_mm = Z * 1000.0
            if abs(depth_mm - Z_mm) <= thresh_mm:
                candidates.append((z, u, v, Z, depth_mm))
        z += sparse_step

    if not candidates:
        return None

    # Refine within the closest candidate region using finer steps
    z_min = max(z_step, candidates[0][0] - sparse_step)
    z_max_refine = min(z_max, candidates[0][0] + sparse_step)
    z = z_min
    while z <= z_max_refine:
        X = ox + d[0] * z
        Y = oy + d[1] * z
        Z = oz + d[2] * z
        if Z <= 0:
            z += z_step
            continue
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        ui = int(round(u)); vi = int(round(v))
        if ui < 0 or ui >= w or vi < 0 or vi >= h:
            z += z_step
            continue
        depth_mm = sample_depth(depth_img, ui, vi, win=2)
        if depth_mm == 0:
            z += z_step
            continue
        if abs(depth_mm/1000.0 - Z) <= (thresh_mm/1000.0):
            return (u, v, Z)
        z += z_step
    return None


def choose_target_2d(hands, dets, alpha=0.6, angle_thresh_deg=30.0, dist_coeff=1.0):
    """
    Fast 2D selection using index joint7->8 ray and YOLO detections as priors.
    - hands: list with hand dicts containing 'pts' (list of (x,y,...) tuples)
    - dets: list of detections with 'box' = (x1,y1,x2,y2)
    Returns selected detection index or None.
    Stateful low-pass filter stored as choose_target_2d.last_dir (numpy array).
    """
    import math
    import numpy as np
    if not hands or not dets:
        return None
    h = hands[0]
    try:
        p7 = h['pts'][7]
        p8 = h['pts'][8]
        tip = np.array([float(p8[0]), float(p8[1])], dtype=float)
        base = np.array([float(p7[0]), float(p7[1])], dtype=float)
    except Exception:
        return None
    v = tip - base
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return None
    dirv = v / norm
    # one-pole low-pass filter (store state on function)
    last = getattr(choose_target_2d, 'last_dir', None)
    if last is None:
        filt = dirv
    else:
        filt = alpha * last + (1.0 - alpha) * dirv
        fn = np.linalg.norm(filt)
        if fn > 1e-6:
            filt = filt / fn
        else:
            filt = dirv
    choose_target_2d.last_dir = filt

    best_j = None
    best_score = -1.0
    for j, det in enumerate(dets):
        try:
            x1,y1,x2,y2 = det['box']
        except Exception:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        vec = np.array([cx, cy], dtype=float) - tip
        proj = np.dot(vec, filt)
        # only consider objects roughly in front of finger (positive projection)
        if proj <= 0:
            continue
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            dist = 1e-6
        u = vec / dist
        # angle score (cosine), require within threshold
        angle_cos = float(np.dot(u, filt))
        angle_cos = max(-1.0, min(1.0, angle_cos))
        angle_deg = math.degrees(math.acos(angle_cos))
        if angle_deg > angle_thresh_deg:
            continue
        # perpendicular pixel distance from ray to center
        perp = abs(filt[0] * vec[1] - filt[1] * vec[0])
        bw = max(1.0, abs(x2 - x1))
        bh = max(1.0, abs(y2 - y1))
        diag = math.hypot(bw, bh)
        overlap_score = max(0.0, 1.0 - (perp / (diag * dist_coeff)))
        angle_score = max(0.0, angle_cos)
        score = angle_score * overlap_score
        if score > best_score:
            best_score = score
            best_j = j
    if best_score <= 0.0:
        return None
    return best_j


def point_in_box(point, box):
    """
    Check if point (x, y) is inside box (x1, y1, x2, y2).
    Returns True if point is inside, False otherwise.
    """
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def compute_target_expectation(target_points, window_size=10):
    """
    Compute expectation (mean) of target points with outlier filtering.
    target_points: list of (x, y) tuples
    window_size: number of recent points to use
    Returns: (mean_x, mean_y) or None if insufficient data
    """
    if not target_points or len(target_points) < 3:
        return None

    # Use recent points
    recent = target_points[-window_size:] if len(target_points) > window_size else target_points
    points_arr = np.array(recent, dtype=float)

    # Compute mean
    mean = np.mean(points_arr, axis=0)

    # Compute std dev for outlier filtering
    std = np.std(points_arr, axis=0)

    # Filter outliers: points within mean ± 2*std
    if std[0] > 0 and std[1] > 0:
        mask = (
            (np.abs(points_arr[:, 0] - mean[0]) <= 2.0 * std[0]) &
            (np.abs(points_arr[:, 1] - mean[1]) <= 2.0 * std[1])
        )
        filtered = points_arr[mask]
        if len(filtered) > 0:
            mean = np.mean(filtered, axis=0)

    return tuple(mean)


class TargetDwellTracker:
    """
    Track how long target point dwells in each detection region.
    """
    def __init__(self, dwell_threshold_frames=90, fps=30):  # 90 frames = 3 seconds at 30fps
        self.dwell_threshold_frames = dwell_threshold_frames
        self.fps = fps
        self.region_dwell = {}  # box_id -> frame count
        self.region_points = {}  # box_id -> list of target points
        self.locked_target = None
        self.locked_box = None
        self.is_locked = False

    def update(self, target_pt, dets):
        """
        Update dwell time for each detection.
        target_pt: (x, y) or None
        dets: list of detections with 'box' field
        Returns: (locked_target, locked_box) or (None, None) if not locked
        """
        if target_pt is None or not dets:
            self._reset_all()
            return self.locked_target, self.locked_box

        # Find which detection the target is in
        current_det_idx = None
        for i, det in enumerate(dets):
            box = det['box']
            if point_in_box(target_pt, box):
                current_det_idx = i
                break

        # Reset all counters and collect points for current detection
        for i in range(len(dets)):
            if i == current_det_idx:
                # Increment dwell time for current detection
                self.region_dwell[i] = self.region_dwell.get(i, 0) + 1
                if i not in self.region_points:
                    self.region_points[i] = []
                self.region_points[i].append(target_pt)
            else:
                # Reset other detections
                if i in self.region_dwell:
                    del self.region_dwell[i]
                if i in self.region_points:
                    del self.region_points[i]

        # Check if any detection exceeded dwell threshold
        for i, dwell_count in self.region_dwell.items():
            if dwell_count >= self.dwell_threshold_frames and not self.is_locked:
                # Lock target at the point with highest dwell in this region
                self.locked_target = compute_target_expectation(self.region_points[i], window_size=20)
                self.locked_box = tuple(dets[i]['box'])
                self.is_locked = True
                return self.locked_target, self.locked_box

        return self.locked_target, self.locked_box

    def reset(self):
        """Reset tracking state (called when shake detected)."""
        self.locked_target = None
        self.locked_box = None
        self.is_locked = False
        self.region_dwell.clear()
        self.region_points.clear()

    def _reset_all(self):
        """Reset all counters when target not in any region."""
        self.region_dwell.clear()
        self.region_points.clear()
