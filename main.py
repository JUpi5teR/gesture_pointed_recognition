import cv2
import time
import logging
import collections
from hardware_camera import LocalCamera, AzureKinect, _HAS_K4A
from hand_module import HandDetector, GestureRecognizer
from face_module import FaceExpression
from object_module import ObjectDetector
from face_gesture import FaceGestureRecognizer
from kalman_tracker import TargetTracker
from background_model import BackgroundModel
from utils import bbox_center, angle_between, sample_depth, pixel_to_point, choose_target_2d, point_in_box, compute_target_expectation, TargetDwellTracker


# Logging to file to capture runtime messages
_log_path = r"e:\Code\CV_lab\my_homework\run_log.txt"
logging.basicConfig(filename=_log_path, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.info('main.py started')

# Use AzureKinect if available to get aligned depth; otherwise fallback to second webcam (no depth).

def draw_info(frame, face_expr, face_pts, hands, dets, selected_idx, target_pt=None, tracking_mode=False):
    """Draw information on frame. Show face landmarks, hands, and objects when target overlaps."""
    if face_expr:
        status = "TRACKING" if tracking_mode else "IDLE"
        cv2.putText(frame, f'Face: {face_expr} | {status}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

    # Draw face landmarks
    if face_pts is not None and len(face_pts) > 0:
        # Key points to display: nose (1), left shoulder (11), right shoulder (12)
        key_indices = [1, 11, 12]
        key_names = ['Nose', 'LShoulder', 'RShoulder']
        for idx, name in zip(key_indices, key_names):
            if idx < len(face_pts):
                x, y = face_pts[idx]
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
                cv2.putText(frame, name, (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Draw hand landmarks
    for i, hand in enumerate(hands):
        for idx, p in enumerate(hand['pts']):
            cv2.circle(frame, (p[0], p[1]), 2, (255,0,0), -1)
        st = hand['fingers']
        cv2.putText(frame, f"Hand{i} idx_ext:{st['index']}", (10,40+20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)

    # Only show detections if target_pt is inside them
    for j, d in enumerate(dets):
        x1,y1,x2,y2 = d['box']
        show_det = False

        # If tracking and target_pt is in this box, show it
        if tracking_mode and target_pt is not None:
            if point_in_box(target_pt, (x1, y1, x2, y2)):
                show_det = True

        if show_det:
            color = (0, 0, 255)  # Red for tracked
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 3)
            cv2.putText(frame, f"{d['label']}:{d['conf']:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def choose_target_3d(hands, dets, depth_img, intrinsics):
    """Choose detection by comparing 3D direction from index finger to detection centers.
    Returns index of selected detection or None.
    """
    if not hands or not dets or depth_img is None:
        return None
    h = hands[0]
    tip_px = h['pts'][8][:2]
    base_px = h['pts'][5][:2]
    # get depths
    tip_d = sample_depth(depth_img, tip_px[0], tip_px[1], win=4)
    base_d = sample_depth(depth_img, base_px[0], base_px[1], win=4)
    if tip_d == 0 or base_d == 0:
        return None
    tip_3d = pixel_to_point(tip_d, tip_px[0], tip_px[1], intrinsics)
    base_3d = pixel_to_point(base_d, base_px[0], base_px[1], intrinsics)
    if tip_3d is None or base_3d is None:
        return None
    dir_3d = (tip_3d[0]-base_3d[0], tip_3d[1]-base_3d[1], tip_3d[2]-base_3d[2])
    best_j = None
    best_ang = 180.0
    for j,d in enumerate(dets):
        cx, cy = bbox_center(d['box'])
        c_d = sample_depth(depth_img, cx, cy, win=4)
        if c_d == 0:
            continue
        c_3d = pixel_to_point(c_d, cx, cy, intrinsics)
        if c_3d is None:
            continue
        vec = (c_3d[0]-tip_3d[0], c_3d[1]-tip_3d[1], c_3d[2]-tip_3d[2])
        ang = angle_between(dir_3d, vec)
        if ang < best_ang:
            best_ang = ang
            best_j = j
    if best_ang < 25.0:
        return best_j
    return None


def main():
    capF = LocalCamera(0)
    # A-view: try AzureKinect first
    depth_img = None
    intrinsics = None
    if _HAS_K4A:
        try:
            ak = AzureKinect()
            use_ak = True
            intrinsics = ak.get_intrinsics()
        except Exception as e:
            logger.warning('AzureKinect init failed, falling back to webcam A: %s', e)
            use_ak = False
            capA = LocalCamera(1)
    else:
        use_ak = False
        capA = LocalCamera(1)

    face = FaceExpression()
    recognizer = GestureRecognizer()
    # legacy direct detector also available as recognizer.detector
    try:
        obj = ObjectDetector()
    except Exception as e:
            logger.warning('Object detector init failed: %s', e)
            obj = None

    # Initialize new modules for face gesture recognition, tracking, and background modeling
    face_gesture = FaceGestureRecognizer(history_len=15, shake_threshold=0.25)
    target_tracker = TargetTracker()
    bg_model = BackgroundModel(roi_pad=30)
    dwell_tracker = TargetDwellTracker(dwell_threshold_frames=90)  # 3 seconds at 30fps

    # State machine: IDLE (normal pointing) or TRACKING (locked target)
    tracking_state = 'IDLE'
    tracked_object_box = None
    target_point_history = collections.deque(maxlen=10)

    while True:
        okF, frameF = capF.read()
        if use_ak:
            okA, (frameA, depth_img) = ak.read()
        else:
            okA, frameA = capA.read()
            depth_img = None
        if not okF or not okA:
            logger.error('Camera read failed')
            break

        # Detect face expression and get facial landmarks
        face_expr, face_pts = face.detect(frameF)

        # Detect face gestures (shake only)
        gesture = face_gesture.detect(face_pts)

        # Check for shake to exit tracking
        if gesture == 'shake_detected' and tracking_state == 'TRACKING':
            # Exit tracking mode
            tracking_state = 'IDLE'
            bg_model.reset()
            target_tracker.reset()
            dwell_tracker.reset()
            target_point_history.clear()
            logger.info('Tracking exited (shake detected)')

        # Hand gesture and object detection
        analysis = recognizer.analyze(frameA)
        hands = analysis['hands']
        dets = obj.detect(frameA) if obj is not None else []

        # 2D-selection: fast, depth-free target selection using index joint7->8
        sel = choose_target_2d(hands, dets)

        # Compute ray intersection target if depth available
        target_pt = None
        target_3d = None
        tip_for_draw = None
        if use_ak and depth_img is not None:
            try:
                # Prefer new keypoint-based pointing (robust and efficient)
                tip_3d_kp, dir_3d_kp = recognizer.detect_pointing_direction_keypoint(hands, depth_img, intrinsics, use_pca=False)
                if tip_3d_kp is not None and dir_3d_kp is not None:
                    from utils import ray_intersect_depth
                    hit = ray_intersect_depth(tip_3d_kp, dir_3d_kp, depth_img, intrinsics, z_step=0.02, z_max=3.0, thresh_mm=80)
                    if hit is not None:
                        hu, hv, hZ = hit
                        target_pt = (int(round(hu)), int(round(hv)))
                        target_3d = (hu, hv, hZ)

                # Fallback to contour-based pointing if keypoint method fails
                if target_pt is None:
                    tip, dir_img = recognizer.detect_pointing_direction(frameA, roi=None, D=7, Step=70, alpha0=30.0, beta0=20.0, theta0=30.0)
                    if tip is not None and dir_img is not None:
                        tip_px = (int(tip[0]), int(tip[1]))
                        tip_d = sample_depth(depth_img, tip_px[0], tip_px[1], win=6)
                        if tip_d and tip_d > 0:
                            tip_3d = pixel_to_point(tip_d, tip_px[0], tip_px[1], intrinsics)
                            far_px = (tip_px[0] + dir_img[0]*30.0, tip_px[1] + dir_img[1]*30.0)
                            far_3d = pixel_to_point(tip_d, far_px[0], far_px[1], intrinsics)
                            if tip_3d is not None and far_3d is not None:
                                dir_3d = (far_3d[0]-tip_3d[0], far_3d[1]-tip_3d[1], far_3d[2]-tip_3d[2])
                                from utils import ray_intersect_depth
                                hit = ray_intersect_depth(tip_3d, dir_3d, depth_img, intrinsics, z_step=0.01, z_max=3.0, thresh_mm=100)
                                if hit is not None:
                                    hu, hv, hZ = hit
                                    target_pt = (int(round(hu)), int(round(hv)))
                                    target_3d = (hu, hv, hZ)

                # Final fallback to landmark-based 3D selection if target not found
                if target_pt is None and hands:
                    h0 = hands[0]
                    tip_px = h0['pts'][8][:2]
                    base_px = h0['pts'][5][:2]
                    tip_d = sample_depth(depth_img, tip_px[0], tip_px[1], win=6)
                    base_d = sample_depth(depth_img, base_px[0], base_px[1], win=6)
                    if tip_d and base_d:
                        tip_3d = pixel_to_point(tip_d, tip_px[0], tip_px[1], intrinsics)
                        base_3d = pixel_to_point(base_d, base_px[0], base_px[1], intrinsics)
                        if tip_3d and base_3d:
                            dir_3d = (tip_3d[0]-base_3d[0], tip_3d[1]-base_3d[1], tip_3d[2]-base_3d[2])
                            from utils import ray_intersect_depth
                            hit = ray_intersect_depth(tip_3d, dir_3d, depth_img, intrinsics, z_step=0.02, z_max=3.0, thresh_mm=80)
                            if hit is not None:
                                hu, hv, hZ = hit
                                target_pt = (int(round(hu)), int(round(hv)))
                                target_3d = (hu, hv, hZ)
            except Exception as e:
                logger.warning('target compute failed: %s', e)

        # Time-based target locking: check dwell time in regions
        if tracking_state == 'IDLE':
            locked_pt, locked_box = dwell_tracker.update(target_pt, dets)
            if locked_pt is not None and locked_box is not None:
                # Target locked after dwelling 3 seconds in a region
                tracking_state = 'TRACKING'
                target_tracker.initialize(locked_pt, locked_box)
                bg_model.initialize(frameA)
                target_pt = locked_pt
                tracked_object_box = locked_box
                logger.info('Target locked at %s after 3s dwell', locked_pt)
        else:
            # In TRACKING mode, dwell tracker stays inactive
            dwell_tracker._reset_all()

        # Update tracking if in TRACKING mode
        if tracking_state == 'TRACKING' and target_pt is not None:
            # Find object containing target point
            for det in dets:
                box = det['box']
                if point_in_box(target_pt, box):
                    tracked_object_box = box
                    # Use Kalman filter to smooth tracking
                    tracked_pt, tracked_box = target_tracker.track(target_pt, box)
                    target_pt = tuple(int(x) for x in tracked_pt[:2])
                    tracked_object_box = tuple(int(x) for x in tracked_box)
                    break

        # normalize frames for display
        def _norm_frame(f):
            import numpy as _np
            if f is None:
                return _np.zeros((480,640,3), dtype=_np.uint8)
            if _np.ndim(f) == 2:
                f2 = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            elif f.shape[2] == 4:
                f2 = f[:, :, :3]
            else:
                f2 = f
            f2 = _np.ascontiguousarray(f2)
            if f2.dtype != _np.uint8:
                f2 = f2.astype(_np.uint8)
            return f2
        fF = _norm_frame(frameF)
        fA = _norm_frame(frameA)
        draw_info(fF, face_expr, face_pts, [], [], None, target_pt=None, tracking_mode=False)
        draw_info(fA, None, None, hands, dets, sel, target_pt=target_pt, tracking_mode=(tracking_state == 'TRACKING'))

        # Draw target point only (no ray)
        if target_pt is not None:
            try:
                cv2.circle(fA, target_pt, 8, (0, 0, 255), -1)
            except Exception:
                pass
        combined = cv2.hconcat([cv2.resize(fF, (640,480)), cv2.resize(fA, (640,480))])
        cv2.imshow('F (face) | A (user view)', combined)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    capF.release()
    if use_ak:
        ak.release()
    else:
        capA.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import traceback, sys
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        err_path = r"e:\Code\CV_lab\my_homework\run_err.txt"
        try:
            with open(err_path, 'w', encoding='utf-8') as f:
                f.write(tb)
        except Exception:
            # fallback: print to stderr
            print('Failed to write run_err.txt; exception:\n' + tb, file=sys.stderr)
        # Re-raise so the process shows failure in console if attached
        raise
