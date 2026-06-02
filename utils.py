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
    depth_img is in millimeters and aligned to color.
    Returns (u,v,Z_m) if found, else None.
    """
    # normalize dir
    d = np.array(dir_3d, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-9)
    ox, oy, oz = origin_3d
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    h, w = depth_img.shape[:2]
    z = z_step
    while z <= z_max:
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
        # compare
        if abs(depth_mm/1000.0 - Z) <= (thresh_mm/1000.0):
            return (u, v, Z)
        z += z_step
    return None
