import cv2
import numpy as np
try:
    from pyk4a import PyK4A, Config, ColorResolution, DepthMode
    _HAS_K4A = True
except Exception:
    _HAS_K4A = False

class LocalCamera:
    def __init__(self, cam_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()

class AzureKinect:
    """Minimal Azure Kinect wrapper.
    Provides color image and depth image aligned to color (both as numpy arrays),
    and simple intrinsics (can be set or attempted to read from SDK).
    """
    def __init__(self, intrinsics=None, color_resolution=ColorResolution.RES_720P, depth_mode=DepthMode.NFOV_UNBINNED):
        if not _HAS_K4A:
            raise RuntimeError("pyk4a not available. Install pyk4a and Azure Kinect SDK.")
        self.k4a = PyK4A(Config(color_resolution=color_resolution, depth_mode=depth_mode))
        self.k4a.start()
        # Default intrinsics (approx for 720p color); user should calibrate/override for accuracy
        self.intrinsics = intrinsics or {'fx': 600.0, 'fy': 600.0, 'cx': 640.0/2.0, 'cy': 720.0/2.0}
        # try to read real intrinsics from pyk4a if available
        try:
            if hasattr(self.k4a, 'calibration') and self.k4a.calibration is not None:
                cam = self.k4a.calibration.color_camera_calibration
                # pyk4a calibration structure varies; attempt common fields
                fx = float(cam.fx) if hasattr(cam, 'fx') else float(cam.intrinsics.parameters.param.fx)
                fy = float(cam.fy) if hasattr(cam, 'fy') else float(cam.intrinsics.parameters.param.fy)
                cx = float(cam.cx) if hasattr(cam, 'cx') else float(cam.intrinsics.parameters.param.cx)
                cy = float(cam.cy) if hasattr(cam, 'cy') else float(cam.intrinsics.parameters.param.cy)
                self.intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        except Exception:
            # keep defaults
            pass

    def read(self):
        """Returns (ok, (color_bgr, depth_mm_aligned))
        depth image is uint16 or int32 in millimeters aligned to color image.
        """
        capture = self.k4a.get_capture()
        color = capture.color
        depth = None
        # transformed_depth may be provided by pyk4a (depth mapped to color space)
        if hasattr(capture, 'transformed_depth'):
            depth = capture.transformed_depth
        else:
            # fall back to raw depth (not aligned)
            try:
                depth = capture.depth
            except Exception:
                depth = None
        return (color is not None), (color, depth)

    def get_intrinsics(self):
        return self.intrinsics

    def release(self):
        try:
            self.k4a.stop()
        except Exception:
            pass
