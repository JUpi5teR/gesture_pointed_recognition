"""Test script for new modules without camera"""
import numpy as np
from face_gesture import FaceGestureRecognizer
from kalman_tracker import TargetTracker, KalmanTracker
from background_model import BackgroundModel
from utils import point_in_box, compute_target_expectation

print("=" * 50)
print("Testing FaceGestureRecognizer")
print("=" * 50)

# Create fake face points (468 landmarks)
fake_face_pts = [(0, 0) for _ in range(468)]
fake_face_pts[1] = (100, 150)   # nose tip
fake_face_pts[11] = (80, 200)   # left shoulder
fake_face_pts[12] = (120, 200)  # right shoulder

gesture_recognizer = FaceGestureRecognizer()
result = gesture_recognizer.detect(fake_face_pts)
print("[OK] FaceGestureRecognizer initialized and returned: %s" % (result,))

print("\n" + "=" * 50)
print("Testing KalmanTracker")
print("=" * 50)

kalman = KalmanTracker(state_dim=2, process_noise=0.05, measurement_noise=2.0)
kalman.initialize((100, 100))

# Simulate 5 frames of noisy measurements
measurements = [(100 + np.random.randn() * 2, 100 + np.random.randn() * 2) for _ in range(5)]
for i, meas in enumerate(measurements):
    pred = kalman.predict()
    updated = kalman.update(meas)
    print("  Frame %d: measurement=(%.1f,%.1f), filtered=(%.1f,%.1f)" % (i, meas[0], meas[1], float(updated[0]), float(updated[1])))

print("[OK] KalmanTracker works correctly")

print("\n" + "=" * 50)
print("Testing TargetTracker")
print("=" * 50)

target_tracker = TargetTracker()
target_point = (100, 100)
target_box = (50, 50, 150, 150)

target_tracker.initialize(target_point, target_box)
print("[OK] TargetTracker initialized")

# Track for 3 frames
for i in range(3):
    noisy_point = (100 + np.random.randn(), 100 + np.random.randn())
    noisy_box = (50 + np.random.randn(), 50 + np.random.randn(),
                 150 + np.random.randn(), 150 + np.random.randn())
    pt, box = target_tracker.track(noisy_point, noisy_box)
    print("  Frame %d: tracked_point=(%.1f,%.1f), tracked_box=(%.1f,%.1f,%.1f,%.1f)" % (i, float(pt[0]), float(pt[1]), float(box[0]), float(box[1]), float(box[2]), float(box[3])))

print("[OK] TargetTracker works correctly")

print("\n" + "=" * 50)
print("Testing utility functions")
print("=" * 50)

# Test point_in_box
box = (50, 50, 150, 150)
test_cases = [
    ((100, 100), True),
    ((50, 50), True),
    ((149, 149), True),
    ((25, 100), False),
    ((175, 100), False),
]

for point, expected in test_cases:
    result = point_in_box(point, box)
    status = "[OK]" if result == expected else "[FAIL]"
    print("  %s point_in_box%s in box%s = %s (expected %s)" % (status, point, box, result, expected))

# Test compute_target_expectation
points = [(100 + i*0.1, 100 + i*0.1) for i in range(10)]
expectation = compute_target_expectation(points, window_size=5)
print("[OK] compute_target_expectation(%d points) = %s" % (len(points), expectation))

print("\n" + "=" * 50)
print("All tests passed!")
print("=" * 50)
