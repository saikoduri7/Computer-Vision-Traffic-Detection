"""YOLO adapter. Pixel boxes retain the original frame dimensions."""
from .config import CLASSES


def select_device(device):
    if device != "auto":
        return device
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


class Detector:
    def __init__(self, config):
        from ultralytics import YOLO
        self.config = config.validate()
        self.model = YOLO(config.model)
        if self.model.task != "detect" or any(self.model.names.get(i) != name for i, name in CLASSES.items()):
            raise ValueError("Use a COCO object-detection model, such as yolo11n.pt")
        self.device = select_device(config.device)

    def options(self):
        return dict(conf=self.config.confidence, imgsz=self.config.image_size,
                    classes=self.config.detect_classes, device=self.device, verbose=False)

    @staticmethod
    def records(result):
        boxes = result.boxes
        if boxes is None:
            return []
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        return [dict(track_id=track_id, class_id=int(cls), class_name=CLASSES[int(cls)],
                     confidence=round(float(score), 4), bbox=[round(float(x), 2) for x in box])
                for track_id, cls, score, box in zip(ids, boxes.cls.cpu().tolist(),
                    boxes.conf.cpu().tolist(), boxes.xyxy.cpu().tolist())]

    def detect(self, frame):
        return self.records(self.model.predict(frame, **self.options())[0])
