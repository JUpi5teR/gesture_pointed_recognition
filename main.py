import cv2
import time
import logging
import numpy as np
from hardware_camera import LocalCamera, AzureKinect, _HAS_K4A
from hand_module import HandDetector, GestureRecognizer
from face_module import FaceExpression
from object_module import ObjectDetector
from face_gesture import HeadGestureRecognizer
from pose_module import PoseDetector
from kalman_tracker import TargetTracker
from background_model import BackgroundModel
from utils import (bbox_center, angle_between, sample_depth, pixel_to_point,
                   choose_target_2d, point_in_box, compute_target_expectation,
                   TargetDwellTracker, SystemState)

# Logging to file to capture runtime messages
_log_path = r"e:\Code\CV_lab\my_homework\run_log.txt"
logging.basicConfig(filename=_log_path, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.info('main.py started')


def draw_skeleton(frame, pose_pts):
    """Draw pose skeleton connections on frame.
    Key landmarks: 0=nose, 11=left shoulder, 12=right shoulder.
    """
    if pose_pts is None or len(pose_pts) < 13:
        return
    # Draw key points
    key_indices = [0, 11, 12]
    key_colors = [(0, 255, 255), (255, 165, 0), (255, 165, 0)]
    key_names = ['Nose', 'LShldr', 'RShldr']
    for idx, color, name in zip(key_indices, key_colors, key_names):
        x, y = int(pose_pts[idx][0]), int(pose_pts[idx][1])
        cv2.circle(frame, (x, y), 5, color, -1)
        cv2.putText(frame, name, (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw shoulder line
    ls = (int(pose_pts[11][0]), int(pose_pts[11][1]))
    rs = (int(pose_pts[12][0]), int(pose_pts[12][1]))
    nose_pt = (int(pose_pts[0][0]), int(pose_pts[0][1]))
    cv2.line(frame, ls, rs, (255, 165, 0), 2)
    # Draw neck line (shoulder center -> nose)
    sc = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    cv2.line(frame, sc, nose_pt, (0, 255, 255), 2)


def draw_info(frame, state, head_gesture, hands, dets, target_pt, tracked_box, dwell_progress):
    """Draw comprehensive HUD information on frame."""
    # --- State indicator (top-left) ---
    state_colors = {
        SystemState.IDLE: (0, 255, 0),
        SystemState.DWELL_WAIT: (0, 255, 255),
        SystemState.TRACKING: (0, 0, 255),
        SystemState.UNLOCKING: (255, 0, 255),
    }
    state_labels = {
        SystemState.IDLE: "IDLE - Free Pointing",
        SystemState.DWELL_WAIT: "DWELL_WAIT - Counting",
        SystemState.TRACKING: "TRACKING - Object Lock",
        SystemState.UNLOCKING: "UNLOCKING - Releasing",
    }
    color = state_colors.get(state, (255, 255, 255))
    cv2.putText(frame, f"State: {state_labels.get(state, str(state))}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- Head gesture counts (top-left, below state) ---
    if head_gesture is not None:
        pitch_str = f"Pitch:{head_gesture['pitch']:.1f}" if head_gesture['pitch'] is not None else "Pitch:N/A"
        yaw_str = f"Yaw:{head_gesture['yaw']:.1f}" if head_gesture['yaw'] is not None else "Yaw:N/A"
        cv2.putText(frame, f"Nod:{head_gesture['nod_count']}  Shake:{head_gesture['shake_count']}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"{pitch_str}  {yaw_str}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        nd = head_gesture.get('nod_down', False)
        sl = head_gesture.get('shake_left', False)
        cv2.putText(frame, f"nod_down:{nd} shake_left:{sl}",
                    (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # --- Dwell progress bar ---
    if state == SystemState.DWELL_WAIT and dwell_progress > 0:
        bar_w = 200
        bar_h = 16
        bar_x, bar_y = 10, 105
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill_w = int(bar_w * dwell_progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 255), -1)
        cv2.putText(frame, f"{dwell_progress*100:.0f}%", (bar_x + bar_w + 5, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # --- Hand landmarks ---
    for i, hand in enumerate(hands):
        for idx, p in enumerate(hand['pts']):
            cv2.circle(frame, (p[0], p[1]), 2, (255, 0, 0), -1)
        st = hand['fingers']
        cv2.putText(frame, f"Hand{i} idx:{st['index']}", (10, 130 + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # --- Detections and tracking box ---
    for j, d in enumerate(dets):
        x1, y1, x2, y2 = d['box']
        show_det = False
        if state in (SystemState.TRACKING, SystemState.UNLOCKING) and target_pt is not None:
            if point_in_box(target_pt, (x1, y1, x2, y2)):
                show_det = True
        elif state in (SystemState.IDLE, SystemState.DWELL_WAIT):
            show_det = True

        if show_det:
            if tracked_box is not None and tuple(d['box']) == tuple(tracked_box):
                color = (0, 0, 255)
                thickness = 3
            else:
                color = (0, 200, 0)
                thickness = 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"{d['label']}:{d['conf']:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # --- Target point ---
    if target_pt is not None:
        try:
            if state == SystemState.TRACKING:
                cv2.circle(frame, target_pt, 10, (0, 0, 255), -1)
                cv2.circle(frame, target_pt, 14, (0, 0, 255), 2)
            elif state == SystemState.DWELL_WAIT:
                cv2.circle(frame, target_pt, 8, (0, 255, 255), -1)
            else:
                cv2.circle(frame, target_pt, 6, (0, 255, 0), -1)
        except Exception:
            pass


def main():
    # --- Camera setup ---
    capF = LocalCamera(0)
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

    # --- Module initialization ---
    face = FaceExpression()
    recognizer = GestureRecognizer()
    try:
        obj = ObjectDetector()
    except Exception as e:
        logger.error('ObjectDetector init failed: %s', e)
        obj = None
    pose_detector = PoseDetector()
    head_gesture_rec = HeadGestureRecognizer()
    target_tracker = TargetTracker()
    bg_model = BackgroundModel()
    dwell_tracker = TargetDwellTracker(dwell_threshold_frames=120)  # 4 seconds at 30fps

    # --- State machine ---
    state = SystemState.IDLE
    target_pt = None
    target_3d = None
    tracked_object_box = None
    prev_shake_count = 0
    prev_nod_count = 0

    logger.info('Main loop started')

    while True:
        # --- Read frames ---
        retF, frameF = capF.read()
        if not retF or frameF is None:
            time.sleep(0.01)
            continue
        frameA = None
        depth_img = None
        if use_ak:
            retA, (frameA_color, depth_a) = ak.read()
            if retA and frameA_color is not None:
                frameA = frameA_color
                depth_img = depth_a
            else:
                frameA = np.zeros_like(frameF)
        else:
            retA, frameA = capA.read()
            if not retA or frameA is None:
                frameA = np.zeros_like(frameF)

        # --- Face expression (F camera) ---
        face_expr, face_pts = face.detect(frameF)

        # --- Pose detection (F camera) for head gesture ---
        pose_pts = pose_detector.detect(frameF)
        head_result = head_gesture_rec.update(pose_pts)

        # --- Hand detection (A camera) ---
        hand_result = recognizer.analyze(frameA)
        hands = hand_result['hands']

        # --- Object detection (A camera) ---
        dets = []
        if obj is not None:
            try:
                dets = obj.detect(frameA)
            except Exception as e:
                logger.warning('Object detection failed: %s', e)

        # --- Compute fingertip target point (only in IDLE / DWELL_WAIT) ---
        if state in (SystemState.IDLE, SystemState.DWELL_WAIT):
            target_pt = None
            target_3d = None
            sel = choose_target_2d(hands, dets)
            if sel is not None:
                cx, cy = bbox_center(dets[sel]['box'])
                target_pt = (int(cx), int(cy))
            elif hands:
                h0 = hands[0]
                tip_px = h0['pts'][8][:2]
                target_pt = (int(tip_px[0]), int(tip_px[1]))

        # =====================================================================
        # STATE MACHINE
        # =====================================================================
        dwell_progress = dwell_tracker.get_dwell_progress() if state == SystemState.DWELL_WAIT else 0.0

        if state == SystemState.IDLE:
            """Free pointing mode. Detect fingertip, transition to DWELL_WAIT if target in region."""
            if target_pt is not None and dets:
                # Check if target is inside any detection region
                in_region = False
                for det in dets:
                    if point_in_box(target_pt, det['box']):
                        in_region = True
                        break
                if in_region:
                    state = SystemState.DWELL_WAIT
                    logger.info('IDLE -> DWELL_WAIT: target entered detection region')

        elif state == SystemState.DWELL_WAIT:
            """Target in region, counting dwell time. Check for lock or return to IDLE."""
            locked_pt, locked_box = dwell_tracker.update(target_pt, dets)
            dwell_progress = dwell_tracker.get_dwell_progress()

            if locked_pt is not None and locked_box is not None:
                # Target locked after 4s dwell -> freeze and start tracking
                target_pt = locked_pt
                tracked_object_box = locked_box
                target_tracker.initialize(locked_pt, locked_box)
                bg_model.initialize(frameA)
                state = SystemState.TRACKING
                # Reset head gesture counters when entering tracking
                head_gesture_rec.reset()
                prev_shake_count = 0
                logger.info('DWELL_WAIT -> TRACKING: target locked at %s after 4s dwell', locked_pt)
            elif target_pt is None or not dets:
                # Target left all regions -> back to IDLE
                dwell_tracker.reset()
                state = SystemState.IDLE
                logger.info('DWELL_WAIT -> IDLE: target left region')
            else:
                # Check if target still in any region
                still_in = False
                for det in dets:
                    if point_in_box(target_pt, det['box']):
                        still_in = True
                        break
                if not still_in:
                    dwell_tracker.reset()
                    state = SystemState.IDLE
                    logger.info('DWELL_WAIT -> IDLE: target left region')

        elif state == SystemState.TRACKING:
            """Object tracking active. Monitor for head shake to unlock."""
            # Update tracking: find object containing target point
            if target_pt is not None:
                found = False
                for det in dets:
                    box = det['box']
                    if point_in_box(target_pt, box):
                        tracked_pt, tracked_box = target_tracker.track(target_pt, box)
                        target_pt = tuple(int(x) for x in tracked_pt[:2])
                        tracked_object_box = tuple(int(x) for x in tracked_box)
                        found = True
                        break
                if not found and dets:
                    # Target lost in current detections, use tracker prediction
                    pred_pt, pred_box = target_tracker.predict(), target_tracker.box_tracker.predict()
                    target_pt = tuple(int(x) for x in pred_pt[:2])
                    tracked_object_box = tuple(int(x) for x in pred_box)

            # Check for head shake
            current_shake = head_result['shake_count']
            if current_shake > prev_shake_count:
                prev_shake_count = current_shake
                # Shake detected -> unlock
                state = SystemState.UNLOCKING
                logger.info('TRACKING -> UNLOCKING: shake detected (count=%d)', current_shake)

        elif state == SystemState.UNLOCKING:
            """Shake detected, release lock and return to IDLE."""
            # Terminate tracking, clear target
            target_pt = None
            target_3d = None
            tracked_object_box = None
            target_tracker.reset()
            bg_model.reset()
            dwell_tracker.reset()
            head_gesture_rec.reset()
            prev_shake_count = 0
            prev_nod_count = 0

            # Transition back to IDLE
            state = SystemState.IDLE
            logger.info('UNLOCKING -> IDLE: lock released, hand pointing restored')

        # --- Normalize frames for display ---
        def _norm_frame(f):
            if f is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            if np.ndim(f) == 2:
                f2 = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            elif f.shape[2] == 4:
                f2 = f[:, :, :3]
            else:
                f2 = f
            f2 = np.ascontiguousarray(f2)
            if f2.dtype != np.uint8:
                f2 = f2.astype(np.uint8)
            return f2

        fF = _norm_frame(frameF)
        fA = _norm_frame(frameA)

        # --- Draw face expression info on F camera ---
        if face_expr:
            cv2.putText(fF, f'Face: {face_expr}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Draw skeleton + head gesture info on F camera ---
        draw_skeleton(fF, pose_pts)
        if head_result['pitch'] is not None:
            cv2.putText(fF, f"Nod:{head_result['nod_count']}  Shake:{head_result['shake_count']}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(fF, f"Pitch:{head_result['pitch']:.1f} Yaw:{head_result['yaw']:.1f}",
                        (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            nd = head_result.get('nod_down', False)
            sl = head_result.get('shake_left', False)
            cv2.putText(fF, f"nod_down:{nd} shake_left:{sl}",
                        (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # --- Draw info on A camera ---
        draw_info(fA, state, head_result, hands, dets, target_pt, tracked_object_box, dwell_progress)

        # --- Display ---
        combined = cv2.hconcat([cv2.resize(fF, (640, 480)), cv2.resize(fA, (640, 480))])
        cv2.imshow('F (face+pose) | A (user view)', combined)
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
            print('Failed to write run_err.txt; exception:\n' + tb, file=sys.stderr)
        raise
