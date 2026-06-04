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
from config import (CAM_FRONT_ID, CAM_SIDE_ID, CAM_WIDTH, CAM_HEIGHT,
                    CONFIRM_TIMEOUT_SEC, POPUP_RADIUS,
                    HAND_SMOOTH_LEN, YOLO_MODEL, YOLO_IMGSZ, YOLO_CONF_THRESH,
                    POSE_DETECTION_CONF, POSE_TRACKING_CONF,
                    FACE_DETECTION_CONF, FACE_MAX_NUM,
                    KALMAN_POINT_PROC_NOISE, KALMAN_POINT_MEAS_NOISE,
                    KALMAN_BOX_PROC_NOISE, KALMAN_BOX_MEAS_NOISE,
                    BG_ROI_PAD, BG_LEARNING_RATE,
                    WIN_W, WIN_H)
from ar_layer import ARManager

STDOUT_PATH = r"e:\Code\CV_lab\my_homework\run_stdout.txt"
LOG_PATH = r"e:\Code\CV_lab\my_homework\run_log.txt"
ERR_PATH = r"e:\Code\CV_lab\my_homework\run_err.txt"

logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def draw_skeleton(frame, pose_pts):
    if pose_pts is None or len(pose_pts) < 13: return
    for idx, clr, nm in zip([0,11,12], [(0,255,255),(255,165,0),(255,165,0)], ["Nose","LShldr","RShldr"]):
        x, y = int(pose_pts[idx][0]), int(pose_pts[idx][1])
        cv2.circle(frame, (x, y), 5, clr, -1)
        cv2.putText(frame, nm, (x+6, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1)
    ls = (int(pose_pts[11][0]), int(pose_pts[11][1]))
    rs = (int(pose_pts[12][0]), int(pose_pts[12][1]))
    ns = (int(pose_pts[0][0]), int(pose_pts[0][1]))
    cv2.line(frame, ls, rs, (255,165,0), 2)
    cv2.line(frame, ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2), ns, (0,255,255), 2)

def draw_confirm_popup(frame, target_pt, elapsed_ratio):
    if target_pt is None: return
    cx, cy = int(target_pt[0]), int(target_pt[1])
    r = POPUP_RADIUS
    cv2.circle(frame, (cx, cy), r+4, (0,0,0), -1)
    cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, int(360*elapsed_ratio), (0,255,255), 3)
    cv2.circle(frame, (cx, cy), r, (0,255,255), 2)
    cv2.putText(frame, "Nod=OK  Shake=Cancel", (cx-80, cy+r+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

def draw_info(frame, state, head_gesture, hands, dets, target_pt, tracked_box, tracked_label, dwell_progress, object_visible):
    state_colors = {
        SystemState.IDLE: (0,255,0), SystemState.DWELL_WAIT: (0,255,255),
        SystemState.CONFIRMING: (0,200,255), SystemState.TRACKING: (0,0,255),
        SystemState.UNLOCKING: (255,0,255),
    }
    state_labels = {
        SystemState.IDLE: "IDLE", SystemState.DWELL_WAIT: "DWELL_WAIT",
        SystemState.CONFIRMING: "CONFIRMING", SystemState.TRACKING: "TRACKING",
        SystemState.UNLOCKING: "UNLOCKING",
    }
    clr = state_colors.get(state, (255,255,255))
    cv2.putText(frame, f"State: {state_labels.get(state, str(state))}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)
    if tracked_label and state in (SystemState.TRACKING, SystemState.CONFIRMING):
        vis_str = " [LOST]" if not object_visible else ""
        cv2.putText(frame, f"Target: {tracked_label}{vis_str}", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255) if object_visible else (0,165,255), 2)
    if head_gesture is not None:
        p = head_gesture.get("pitch"); y = head_gesture.get("yaw")
        ps = f"P:{p:.1f}" if p is not None else "P:N/A"
        ys = f"Y:{y:.1f}" if y is not None else "Y:N/A"
        oy = 70 if tracked_label and state in (SystemState.TRACKING, SystemState.CONFIRMING) else 50
        cv2.putText(frame, f"Nod:{head_gesture.get('nod_count',0)} Shake:{head_gesture.get('shake_count',0)}",
                    (10, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.putText(frame, f"{ps} {ys}", (10, oy+22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    if state == SystemState.DWELL_WAIT and dwell_progress > 0:
        by = 115 if tracked_label and state in (SystemState.TRACKING, SystemState.CONFIRMING) else 92
        bx, bw, bh = 10, 200, 16
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (80,80,80), -1)
        cv2.rectangle(frame, (bx, by), (bx+int(bw*dwell_progress), by+bh), (0,255,255), -1)
        cv2.putText(frame, f"{dwell_progress*100:.0f}%", (bx+bw+5, by+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    for i, hand in enumerate(hands):
        for p in hand.get("pts", []):
            cv2.circle(frame, (p[0], p[1]), 2, (255,0,0), -1)
        st = hand.get("fingers", {})
        cv2.putText(frame, f"Hand{i} idx:{st.get('index',0)}", (10, 140+20*i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    for j, d in enumerate(dets):
        x1,y1,x2,y2 = d["box"]
        show = False
        if state in (SystemState.TRACKING, SystemState.UNLOCKING):
            if d.get("label") == tracked_label: show = True
        elif state in (SystemState.IDLE, SystemState.DWELL_WAIT, SystemState.CONFIRMING):
            show = True
        if show:
            is_tracked = (d.get("label") == tracked_label and
                         state in (SystemState.TRACKING, SystemState.CONFIRMING) and object_visible)
            tc = (0,0,255) if is_tracked else (0,200,0)
            tk = 3 if is_tracked else 1
            cv2.rectangle(frame, (x1,y1),(x2,y2), tc, tk)
            cv2.putText(frame, f"{d['label']}:{d['conf']:.2f}", (x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 1)
    if target_pt is not None and object_visible:
        try:
            if state == SystemState.TRACKING:
                cv2.circle(frame, target_pt, 10, (0,0,255), -1)
                cv2.circle(frame, target_pt, 14, (0,0,255), 2)
            elif state == SystemState.DWELL_WAIT:
                cv2.circle(frame, target_pt, 8, (0,255,255), -1)
            elif state != SystemState.CONFIRMING:
                cv2.circle(frame, target_pt, 6, (0,255,0), -1)
        except Exception:
            pass
    if tracked_box is not None and state == SystemState.TRACKING and object_visible:
        bx1,by1,bx2,by2 = [int(v) for v in tracked_box]
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,0,255), 2)

def _get_clipboard():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        text = root.clipboard_get()
        root.destroy()
        return text
    except Exception:
        return None


def _detect_hands_only(detector, frame):
    """Hand detection with gesture encoding - no ArUco tracker to avoid CSRT crash."""
    try:
        hands_raw = detector.detect(frame)
        hands = []
        for hr in hands_raw:
            pts = hr.get("pts", [])
            if not pts or len(pts) < 13:
                continue
            wrist = np.array(pts[0][:2], dtype=float)
            fingers = {}
            # thumb: tip(4)-pip(2) ratio
            d1 = np.linalg.norm(np.array(pts[4][:2]) - np.array(pts[2][:2]))
            d2 = np.linalg.norm(np.array(pts[2][:2]) - wrist) + 1e-6
            fingers["thumb"] = (d1 / d2) > 0.5
            for tip_idx, mcp_idx, pip_idx, name in [
                (8, 5, 6, "index"), (12, 9, 10, "middle"),
                (16, 13, 14, "ring"), (20, 17, 18, "pinky")
            ]:
                tip = np.array(pts[tip_idx][:2])
                mcp = np.array(pts[mcp_idx][:2])
                pip = np.array(pts[pip_idx][:2])
                d3 = np.linalg.norm(tip - mcp)
                d4 = np.linalg.norm(mcp - pip) + 1e-6
                fingers[name] = d3 > d4 * 0.7
            order = ["thumb", "index", "middle", "ring", "pinky"]
            code = 0
            for i, n in enumerate(order):
                if fingers.get(n, False):
                    code |= (1 << i)
            hands.append({"pts": pts, "fingers": fingers, "code": code})
        return {"hands": hands, "aruco": []}
    except Exception:
        return {"hands": [], "aruco": []}

LOGIN_WIN_W, LOGIN_WIN_H = 500, 380

def show_login_page(ar_manager):
    if ar_manager.login.is_logged_in:
        return True
    api_key_input = ""
    active_input = True
    logger.info("Showing login page")
    while True:
        canvas = np.zeros((LOGIN_WIN_H, LOGIN_WIN_W, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (LOGIN_WIN_W, LOGIN_WIN_H), (30, 30, 50), -1)
        cv2.putText(canvas, "AR Information System", (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
        cv2.putText(canvas, "ChatGPT Login", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 200), 1)
        cv2.putText(canvas, "Enter your OpenAI API Key:", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(canvas, (60, 155), (440, 185), (60, 60, 80), -1)
        cv2.rectangle(canvas, (60, 155), (440, 185), (100, 100, 180) if active_input else (80, 80, 100), 2)
        display_key = api_key_input[:42] + ("..." if len(api_key_input) > 42 else "")
        if not display_key:
            display_key = "Paste or type your API key here..."
            cv2.putText(canvas, display_key, (65, 177), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 120), 1)
        else:
            cv2.putText(canvas, display_key, (65, 177), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        cv2.putText(canvas, "Get your API key: platform.openai.com/api-keys", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 140), 1)
        cv2.putText(canvas, "Ctrl+V  ->  Paste  |  Enter  ->  Login", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 200, 180), 1)
        cv2.putText(canvas, "Esc  ->  Exit  |  Backspace  ->  Delete", (60, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 160), 1)
        cv2.imshow("AR System - Login", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            return False
        elif key == 13:
            if api_key_input.strip():
                if ar_manager.login.login(api_key_input.strip()):
                    cv2.destroyWindow("AR System - Login")
                    logger.info("Login successful")
                    return True
        elif key == 22:
            pasted = _get_clipboard()
            if pasted:
                pasted = pasted.replace("\r", "").replace("\n", "").replace("\t", "").strip()
                api_key_input += pasted
        elif key == 8:
            if api_key_input:
                api_key_input = api_key_input[:-1]
        elif 32 <= key <= 126:
            api_key_input += chr(key)
    return False

def main_ar():
    capF = LocalCamera(CAM_FRONT_ID)
    depth_img = None; intrinsics = None
    if _HAS_K4A:
        try:
            ak = AzureKinect(); use_ak = True; intrinsics = ak.get_intrinsics()
        except Exception as e:
            logger.warning("AzureKinect init failed: %s", e); use_ak = False
            capA = LocalCamera(CAM_SIDE_ID)
    else:
        use_ak = False; capA = LocalCamera(CAM_SIDE_ID)

    face = FaceExpression()
    recognizer = GestureRecognizer(smooth_len=HAND_SMOOTH_LEN)
    try:
        obj = ObjectDetector(model_path=YOLO_MODEL)
    except Exception as e:
        logger.error("ObjectDetector init failed: %s", e); obj = None
    pose_detector = PoseDetector(min_detection_confidence=POSE_DETECTION_CONF,
                                 min_tracking_confidence=POSE_TRACKING_CONF)
    head_gesture_rec = HeadGestureRecognizer()
    target_tracker = TargetTracker(point_process_noise=KALMAN_POINT_PROC_NOISE,
                                   point_measurement_noise=KALMAN_POINT_MEAS_NOISE,
                                   box_process_noise=KALMAN_BOX_PROC_NOISE,
                                   box_measurement_noise=KALMAN_BOX_MEAS_NOISE)
    bg_model = BackgroundModel(roi_pad=BG_ROI_PAD, learning_rate=BG_LEARNING_RATE)
    dwell_tracker = TargetDwellTracker()

    state = SystemState.IDLE
    target_pt = None; target_3d = None
    tracked_object_box = None; tracked_label = None
    prev_shake_count = 0; confirm_start_time = None
    last_real_pt = None; last_real_box = None
    object_visible = True
    ar = ARManager()
    if not show_login_page(ar):
        logger.info("Login cancelled, exiting")
        return
    logger.info("AR System ready - main loop starting")

    while True:
        retF, frameF = capF.read()
        if not retF or frameF is None: time.sleep(0.01); continue
        frameA = None; depth_img = None
        if use_ak:
            retA, (frameA_color, depth_a) = ak.read()
            frameA = frameA_color if retA and frameA_color is not None else np.zeros_like(frameF)
            depth_img = depth_a if retA else None
        else:
            retA, frameA = capA.read()
            if not retA or frameA is None: frameA = np.zeros_like(frameF)

        face_expr, face_pts = face.detect(frameF)
        pose_pts = pose_detector.detect(frameF)
        head_result = head_gesture_rec.update(pose_pts)

        hand_result = _detect_hands_only(recognizer.detector, frameA)
        hands = hand_result["hands"]
        # ar.update_hand removed - hands passed directly to try_open_panel

        dets = []
        if obj is not None:
            try: dets = obj.detect(frameA)
            except Exception as e: logger.warning("Object detection failed: %s", e)

        if state in (SystemState.IDLE, SystemState.DWELL_WAIT):
            target_pt = None; target_3d = None
            sel = choose_target_2d(hands, dets)
            if sel is not None:
                cx, cy = bbox_center(dets[sel]["box"])
                target_pt = (int(cx), int(cy)); tracked_label = dets[sel]["label"]
            elif hands:
                tip = hands[0]["pts"][8][:2]
                target_pt = (int(tip[0]), int(tip[1])); tracked_label = None

        dwell_progress = dwell_tracker.get_dwell_progress() if state == SystemState.DWELL_WAIT else 0.0

        if state == SystemState.IDLE:
            if target_pt is not None and dets:
                if any(point_in_box(target_pt, d["box"]) for d in dets):
                    state = SystemState.DWELL_WAIT
                    logger.info("IDLE -> DWELL_WAIT")

        elif state == SystemState.DWELL_WAIT:
            triggered_pt, triggered_box = dwell_tracker.update(target_pt, dets)
            dwell_progress = dwell_tracker.get_dwell_progress()
            if triggered_pt is not None and triggered_box is not None:
                target_pt = triggered_pt; tracked_object_box = triggered_box
                for d in dets:
                    if tuple(d["box"]) == tuple(triggered_box):
                        tracked_label = d["label"]; break
                state = SystemState.CONFIRMING
                confirm_start_time = time.time()
                head_gesture_rec.reset_confirm_state()
                logger.info("DWELL_WAIT -> CONFIRMING: label=%s", tracked_label)
            elif dwell_progress <= 0.0:
                dwell_tracker.reset(); state = SystemState.IDLE
                logger.info("DWELL_WAIT -> IDLE: dwell fully decayed")

        elif state == SystemState.CONFIRMING:
            if head_gesture_rec.check_nod_confirm(pose_pts):
                cx, cy = bbox_center(tracked_object_box)
                target_tracker.initialize((cx, cy), tracked_object_box)
                bg_model.initialize(frameA)
                last_real_pt = (cx, cy); last_real_box = tracked_object_box
                object_visible = True
                head_gesture_rec.reset(); prev_shake_count = 0
                state = SystemState.TRACKING
                logger.info("CONFIRMING -> TRACKING: bound to %s", tracked_label)
                ar.on_target_bound(frameA, tracked_object_box, tracked_label)
            elif head_gesture_rec.check_shake_deny(pose_pts):
                target_pt = None; target_3d = None
                tracked_object_box = None; tracked_label = None
                last_real_pt = None; last_real_box = None; object_visible = True
                dwell_tracker.reset(); head_gesture_rec.reset_confirm_state()
                state = SystemState.IDLE
                logger.info("CONFIRMING -> IDLE: denied")
            elif confirm_start_time is not None and (time.time() - confirm_start_time) > CONFIRM_TIMEOUT_SEC:
                target_pt = None; target_3d = None
                tracked_object_box = None; tracked_label = None
                last_real_pt = None; last_real_box = None; object_visible = True
                dwell_tracker.reset(); head_gesture_rec.reset_confirm_state()
                state = SystemState.IDLE
                logger.info("CONFIRMING -> IDLE: timeout")

        elif state == SystemState.TRACKING:
            search_center = np.array(last_real_pt, dtype=float) if last_real_pt is not None else np.array([0,0], dtype=float)
            if last_real_box is not None:
                x1,y1,x2,y2 = last_real_box
                base_radius = max(0.5 * np.linalg.norm([x2-x1, y2-y1]), 50.0)
            else:
                base_radius = 100.0
            best_det = None; best_dist = base_radius
            for det in dets:
                if tracked_label and det.get("label") != tracked_label:
                    continue
                cx, cy = bbox_center(det["box"])
                dist = np.linalg.norm(search_center - np.array([cx, cy]))
                if dist < best_dist:
                    best_dist = dist; best_det = det
            if best_det is not None:
                cx, cy = bbox_center(best_det["box"])
                target_pt = (int(cx), int(cy))
                tracked_object_box = tuple(best_det["box"])
                last_real_pt = (cx, cy)
                last_real_box = tuple(best_det["box"])
                object_visible = True
                target_tracker.update((cx, cy), tuple(best_det["box"]))
                ar.on_target_visible((cx, cy), tracked_object_box)
            else:
                target_pt = None
                object_visible = False
                ar.on_target_lost()
            current_shake = head_result["shake_count"]
            if current_shake > prev_shake_count:
                prev_shake_count = current_shake
                state = SystemState.UNLOCKING
                logger.info("TRACKING -> UNLOCKING: shake detected")

        elif state == SystemState.UNLOCKING:
            target_pt = None; target_3d = None
            tracked_object_box = None; tracked_label = None
            last_real_pt = None; last_real_box = None; object_visible = True
            target_tracker.reset(); bg_model.reset(); dwell_tracker.reset()
            head_gesture_rec.reset(); prev_shake_count = 0; confirm_start_time = None
            state = SystemState.IDLE
            logger.info("UNLOCKING -> IDLE: lock released")
            ar.on_target_unlock()

        if state == SystemState.TRACKING:
            ar.try_open_panel(hands, frameA)
        if state == SystemState.TRACKING or ar.state.name == "AR_CONTROL":
            ar.update_scroll(hands)

        def _norm(f):
            if f is None: return np.zeros((480,640,3), dtype=np.uint8)
            if np.ndim(f)==2: f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            elif f.shape[2]==4: f = f[:,:,:3]
            f = np.ascontiguousarray(f)
            return f.astype(np.uint8) if f.dtype!=np.uint8 else f

        fF = _norm(frameF); fA = _norm(frameA)
        if face_expr:
            cv2.putText(fF, f"Face: {face_expr}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        draw_skeleton(fF, pose_pts)
        if head_result.get("pitch") is not None:
            cv2.putText(fF, f"Nod:{head_result.get('nod_count',0)} Shake:{head_result.get('shake_count',0)}",
                        (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            cv2.putText(fF, f"Pitch:{head_result.get('pitch',0):.1f} Yaw:{head_result.get('yaw',0):.1f}",
                        (10,72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        draw_info(fA, state, head_result, hands, dets, target_pt,
                  tracked_object_box, tracked_label, dwell_progress, object_visible)
        fA = ar.draw(fA)

        if state == SystemState.CONFIRMING and target_pt is not None and confirm_start_time is not None:
            ratio = min(1.0, (time.time() - confirm_start_time) / CONFIRM_TIMEOUT_SEC)
            draw_confirm_popup(fA, target_pt, ratio)

        combined = cv2.hconcat([cv2.resize(fF, (WIN_W, WIN_H)), cv2.resize(fA, (WIN_W, WIN_H))])
        cv2.imshow("F (face+pose) | A (user view + AR)", combined)
        if cv2.waitKey(1) & 0xFF == 27: break

    capF.release()
    if use_ak: ak.release()
    else: capA.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import traceback, sys
    # Redirect stdout to file
    try:
        sys.stdout = open(STDOUT_PATH, "w", encoding="utf-8")
    except Exception:
        pass
    try:
        main_ar()
    except Exception:
        tb = traceback.format_exc()
        logger.error("Crash: %s", tb)
        try:
            with open(ERR_PATH, "w", encoding="utf-8") as f:
                f.write(tb + "\n")
        except Exception:
            pass
        print(tb, file=sys.__stdout__)
        raise
