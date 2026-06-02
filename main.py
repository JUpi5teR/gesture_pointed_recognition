import cv2
import time
import logging
from hardware_camera import LocalCamera, AzureKinect, _HAS_K4A
from hand_module import HandDetector, GestureRecognizer
from face_module import FaceExpression
from object_module import ObjectDetector
from utils import bbox_center, angle_between, sample_depth, pixel_to_point


# Logging to file to capture runtime messages
_log_path = r"e:\Code\CV_lab\my_homework\run_log.txt"
logging.basicConfig(filename=_log_path, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.info('main.py started')

# Use AzureKinect if available to get aligned depth; otherwise fallback to second webcam (no depth).

def draw_info(frame, face_expr, hands, dets, selected_idx):
    if face_expr:
        cv2.putText(frame, f'Face: {face_expr}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
    for i, hand in enumerate(hands):
        for idx, p in enumerate(hand['pts']):
            cv2.circle(frame, (p[0], p[1]), 2, (255,0,0), -1)
        st = hand['fingers']
        cv2.putText(frame, f"Hand{i} idx_ext:{st['index']}", (10,40+20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
    for j, d in enumerate(dets):
        x1,y1,x2,y2 = d['box']
        color = (0,255,0)
        if j==selected_idx:
            color = (0,0,255)
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, f"{d['label']}:{d['conf']:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1)


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
        face_expr, _ = face.detect(frameF)
        analysis = recognizer.analyze(frameA)
        hands = analysis['hands']
        dets = obj.detect(frameA) if obj is not None else []
        sel = choose_target_3d(hands, dets, depth_img, intrinsics) if use_ak else None
        # compute ray intersection target if depth available and hand present
        target_pt = None
        if use_ak and depth_img is not None and hands:
            try:
                # first hand
                h0 = hands[0]
                tip_px = h0['pts'][8][:2]
                base_px = h0['pts'][5][:2]
                tip_d = sample_depth(depth_img, tip_px[0], tip_px[1], win=3)
                base_d = sample_depth(depth_img, base_px[0], base_px[1], win=3)
                if tip_d and base_d:
                    tip_3d = pixel_to_point(tip_d, tip_px[0], tip_px[1], intrinsics)
                    base_3d = pixel_to_point(base_d, base_px[0], base_px[1], intrinsics)
                    if tip_3d and base_3d:
                        dir_3d = (tip_3d[0]-base_3d[0], tip_3d[1]-base_3d[1], tip_3d[2]-base_3d[2])
                        # cast ray starting slightly beyond tip (0.02m) to avoid immediate hit
                        origin = tip_3d
                        # call utility
                        hit = None
                        try:
                            from utils import ray_intersect_depth
                            hit = ray_intersect_depth(origin, dir_3d, depth_img, intrinsics, z_step=0.02, z_max=3.0, thresh_mm=80)
                        except Exception as e:
                            logger.warning('ray intersect failed: %s', e)
                        if hit is not None:
                            hu, hv, hZ = hit
                            target_pt = (int(round(hu)), int(round(hv)))
            except Exception as e:
                logger.warning('target compute failed: %s', e)

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
        draw_info(fF, face_expr, [], [], None)
        draw_info(fA, None, hands, dets, sel)
        # draw target if found
        if target_pt is not None:
            try:
                cv2.circle(fA, target_pt, 8, (0,0,255), -1)
                # draw line from fingertip to target
                tip = (int(hands[0]['pts'][8][0]), int(hands[0]['pts'][8][1]))
                cv2.line(fA, tip, target_pt, (0,0,255), 2)
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
