import cv2
import numpy as np
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        if not _HAS_YOLO:
            raise RuntimeError('ultralytics not installed. Install ultralytics and torch.')
        self.model = YOLO(model_path)

    def detect(self, image):
        # Ensure 3-channel BGR input for YOLO
        import numpy as _np
        if image is None:
            return []
        if _np.ndim(image) == 3 and image.shape[2] == 4:
            # drop alpha channel
            image = image[:, :, :3]
        # returns list of dicts: {box:[x1,y1,x2,y2], conf, cls, label}
        res = self.model.predict(image, imgsz=640, conf=0.35, verbose=False)
        out = []
        if len(res) == 0:
            return out
        boxes = res[0].boxes
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            label = self.model.names.get(cls, str(cls)) if hasattr(self.model, 'names') else str(cls)
            out.append({"box": [x1,y1,x2,y2], "conf": conf, "cls": cls, "label": label})
        return out
