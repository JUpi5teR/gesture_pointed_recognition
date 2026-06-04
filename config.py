# ============================================================
#  项目全局配置文件 —— 所有可调参数集中在此
# ============================================================

# ---------- 摄像头 ----------
CAM_FRONT_ID = 0
CAM_SIDE_ID = 1
CAM_WIDTH = 640
CAM_HEIGHT = 480

# ---------- 停留冻结 ----------
DWELL_TRIGGER_SEC = 1.5     # 指尖在目标区域内停留多少秒后弹出确认弹窗
FPS = 30                    # 帧率估计，用于帧数<->秒数换算
CONFIRM_TIMEOUT_SEC = 5.0   # 弹窗等待确认的超时秒数（超时自动冻结）
POPUP_RADIUS = 40           # 确认弹窗圆环半径（像素）

# ---------- 停留计时容错 ----------
DWELL_DECAY_RATE = 0.5      # 离开区域时每帧衰退比例（0=立刻清零，1=不衰退）
                            # 0.5 表示每帧保留50%计数值，约2秒衰退到0
DWELL_TOLERANCE_SEC = 0.3   # 容忍短暂离开的最大秒数（在此时间内回到区域不衰退）
DWELL_MIN_HOLD_RATIO = 0.3  # 衰退低于此比例时才彻底清零（防止微小抖动反复清零）

# ---------- 手部检测 ----------
HAND_MAX_NUM = 2
HAND_DETECTION_CONF = 0.5
HAND_TRACKING_CONF = 0.5
HAND_SMOOTH_LEN = 7

# ---------- 目标检测（YOLOv8） ----------
YOLO_MODEL = 'yolov8n.pt'
YOLO_IMGSZ = 640
YOLO_CONF_THRESH = 0.35

# ---------- 头部动作识别阈值（度） ----------
PITCH_NOD_DOWN = 18.0
PITCH_NOD_UP = 8.0
YAW_SHAKE_LEFT = -12.0
YAW_SHAKE_RIGHT = 12.0
CALIBRATION_FRAMES = 15

# ---------- Pose 检测 ----------
POSE_DETECTION_CONF = 0.5
POSE_TRACKING_CONF = 0.5

# ---------- 人脸表情 ----------
FACE_MAX_NUM = 1
FACE_DETECTION_CONF = 0.5

# ---------- Kalman 跟踪器 ----------
KALMAN_POINT_PROC_NOISE = 0.05
KALMAN_POINT_MEAS_NOISE = 2.0
KALMAN_BOX_PROC_NOISE = 0.02
KALMAN_BOX_MEAS_NOISE = 5.0

# ---------- 背景建模 ----------
BG_ROI_PAD = 30
BG_LEARNING_RATE = 0.01

# ---------- 2D 目标选择 ----------
TARGET_ANGLE_THRESH = 30.0
TARGET_DIST_COEFF = 1.0

# ---------- 显示窗口 ----------
WIN_W = 640
WIN_H = 480
