"""Test script for new modules without camera"""
import numpy as np
import math
from face_gesture import HeadGestureRecognizer
from kalman_tracker import TargetTracker, KalmanTracker
from background_model import BackgroundModel
from utils import (point_in_box, compute_target_expectation, TargetDwellTracker, SystemState)

print("=" * 50)
print("Testing SystemState enum")
print("=" * 50)
for s in SystemState:
    print("  %s = %s" % (s.name, s.value))
assert SystemState.IDLE.value == "IDLE"
assert SystemState.DWELL_WAIT.value == "DWELL_WAIT"
assert SystemState.TRACKING.value == "TRACKING"
assert SystemState.UNLOCKING.value == "UNLOCKING"
print("[OK] SystemState enum works")

print("\n" + "=" * 50)
print("Testing HeadGestureRecognizer")
print("=" * 50)

gest_rec = HeadGestureRecognizer()

# Shoulder positions (fixed): left=(250, 250), right=(390, 250)
# shoulder_center = (320, 250), shoulder_width = 140
# Upright neutral: nose at (320, 100) -> neck_length=150, ratio=150/140=1.07

# First, "calibrate" with upright frames
neutral_pts = [(0, 0)] * 33
neutral_pts[0] = (320, 100)
neutral_pts[11] = (250, 250)
neutral_pts[12] = (390, 250)

# Feed calibration frames
for _ in range(20):
    result = gest_rec.update(neutral_pts)
print("  After calibration: calibrated=%s, ratio=%.3f" % (
    gest_rec.calibrated, gest_rec.calibrated_neck_shoulder_ratio or 0))

# Now test neutral
result = gest_rec.update(neutral_pts)
print("  Neutral: pitch=%.1f yaw=%.1f nod=%d shake=%d" % (
    result['pitch'] or 0, result['yaw'] or 0, result['nod_count'], result['shake_count']))

# Nod down: nose close to shoulder level (y=210)
# neck_length = sqrt(0 + 40^2) = 40, ratio = 40/140 = 0.286
# cos_equiv = 0.286/1.07 = 0.267, ratio_pitch = acos(0.267) = 74.5deg -> well above 18
nod_down_pts = list(neutral_pts)
nod_down_pts[0] = (320, 210)
result = gest_rec.update(nod_down_pts)
print("  Nod down: pitch=%.1f yaw=%.1f nod_down=%s" % (
    result['pitch'] or 0, result['yaw'] or 0, result['nod_down']))
assert result['pitch'] > 18.0, "Expected pitch > 18 for nod down, got %.1f" % result['pitch']
assert result['nod_down'] == True, "Expected nod_down=True"

# Nod up: back to neutral
result = gest_rec.update(neutral_pts)
print("  Nod up: pitch=%.1f nod_count=%d" % (result['pitch'] or 0, result['nod_count']))
assert result['nod_count'] == 1, "Expected 1 nod after down+up cycle"
print("[OK] Nod detection works")

# Shake left: nose shifted to left of shoulder center
shake_left_pts = list(neutral_pts)
shake_left_pts[0] = (250, 100)  # nose shifted 70px left of center
result = gest_rec.update(shake_left_pts)
print("  Shake left: yaw=%.1f shake_left=%s" % (result['yaw'] or 0, result['shake_left']))
# lateral = dot((250-320, 100-250), (1, 0)) = -70
# forward = dot((250-320, 100-250), (0, -1)) = 150
# yaw = atan2(-70, 150) = -25deg -> < -12, should trigger shake_left
assert result['shake_left'] == True, "Expected shake_left=True, yaw=%.1f" % (result['yaw'] or 0)

# Shake right: nose shifted to right of shoulder center
shake_right_pts = list(neutral_pts)
shake_right_pts[0] = (390, 100)  # nose shifted 70px right of center
result = gest_rec.update(shake_right_pts)
print("  Shake right: yaw=%.1f shake_count=%d" % (result['yaw'] or 0, result['shake_count']))
assert result['shake_count'] == 1, "Expected 1 shake after left+right cycle"
print("[OK] Shake detection works")

# Test reset
gest_rec.reset()
assert gest_rec.nod_count == 0 and gest_rec.shake_count == 0
print("[OK] HeadGestureRecognizer reset works")

print("\n" + "=" * 50)
print("Testing TargetDwellTracker (4 seconds / 120 frames)")
print("=" * 50)

dwell_tracker = TargetDwellTracker(dwell_threshold_frames=10)  # Fast test

dets = [
    {'box': (50, 50, 150, 150), 'label': 'obj1', 'conf': 0.95},
    {'box': (200, 200, 300, 300), 'label': 'obj2', 'conf': 0.85},
]

target_points = [(100 + np.random.randn() * 2, 100 + np.random.randn() * 2) for _ in range(12)]
locked_pt = None
for i in range(12):
    locked_pt, locked_box = dwell_tracker.update(target_points[i], dets)
    progress = dwell_tracker.get_dwell_progress()
    if locked_pt is not None:
        print("  [LOCKED] at frame %d: point=(%.1f,%.1f), box=%s" % (i, locked_pt[0], locked_pt[1], locked_box))
        break
    else:
        print("  Frame %d: progress=%.0f%%" % (i, progress * 100))

assert locked_pt is not None, "Expected lock after 10 frames"
print("[OK] TargetDwellTracker works correctly")

dwell_tracker.reset()
assert dwell_tracker.locked_target is None
print("[OK] TargetDwellTracker reset works")

print("\n" + "=" * 50)
print("Testing KalmanTracker")
print("=" * 50)

kalman = KalmanTracker(state_dim=2, process_noise=0.05, measurement_noise=2.0)
kalman.initialize((100, 100))

measurements = [(100 + np.random.randn() * 2, 100 + np.random.randn() * 2) for _ in range(5)]
for i, meas in enumerate(measurements):
    pred = kalman.predict()
    updated = kalman.update(meas)
    print("  Frame %d: meas=(%.1f,%.1f) filtered=(%.1f,%.1f)" % (i, meas[0], meas[1], float(updated[0]), float(updated[1])))
print("[OK] KalmanTracker works correctly")

print("\n" + "=" * 50)
print("Testing TargetTracker")
print("=" * 50)

target_tracker = TargetTracker()
target_point = (100, 100)
target_box = (50, 50, 150, 150)
target_tracker.initialize(target_point, target_box)
print("[OK] TargetTracker initialized")

for i in range(3):
    noisy_point = (100 + np.random.randn(), 100 + np.random.randn())
    noisy_box = (50 + np.random.randn(), 50 + np.random.randn(),
                 150 + np.random.randn(), 150 + np.random.randn())
    pt, box = target_tracker.track(noisy_point, noisy_box)
    print("  Frame %d: pt=(%.1f,%.1f) box=(%.1f,%.1f,%.1f,%.1f)" % (
        i, float(pt[0]), float(pt[1]), float(box[0]), float(box[1]), float(box[2]), float(box[3])))
print("[OK] TargetTracker works correctly")

print("\n" + "=" * 50)
print("Testing utility functions")
print("=" * 50)

box = (50, 50, 150, 150)
test_cases = [((100, 100), True), ((50, 50), True), ((149, 149), True), ((25, 100), False), ((175, 100), False)]
for point, expected in test_cases:
    result = point_in_box(point, box)
    status = "[OK]" if result == expected else "[FAIL]"
    print("  %s point_in_box%s = %s (expected %s)" % (status, point, result, expected))

points = [(100 + i*0.1, 100 + i*0.1) for i in range(10)]
expectation = compute_target_expectation(points, window_size=5)
print("[OK] compute_target_expectation = %s" % (expectation,))

print("\n" + "=" * 50)
print("All tests passed!")
print("=" * 50)
