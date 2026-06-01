Prototype for dual-camera pointing-based object selection.

Structure:
- hardware_camera.py: local webcam and Azure Kinect capture
- face_module.py: face detection and simple expression heuristics (MediaPipe)
- hand_module.py: hand landmarks, finger state, index finger direction (MediaPipe)
- object_module.py: object detection (YOLOv8 via ultralytics)
- utils.py: helper math and bbox utilities
- main.py: orchestrates capture, processing, and display

Quick start:
1. Create a Python venv: python -m venv .venv && .\.venv\Scripts\activate
2. pip install -r requirements.txt
3. Ensure Azure Kinect SDK and drivers are installed if using Azure Kinect DK
4. Run: python main.py

Notes:
- This is a prototype: calibration between cameras and accurate 3D mapping requires extra steps.
- GPU is recommended for YOLO performance.
