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


def sample_depth(depth_img, u, v, win=5):
    """Sample depth image (uint16 mm) around (u,v) and return median of non-zero samples in mm.
    Returns 0 if no valid depth.
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
