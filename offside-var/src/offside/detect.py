import supervision as sv
from ultralytics import YOLO

from .config import get_device

COCO_PERSON, COCO_BALL = 0, 32


class Detector:
    """YOLO wrapper returning (players, ball) as supervision Detections.

    Starts with COCO-pretrained weights. The ball is small and often missed;
    in a later week, fine-tune on a football dataset (e.g. from Roboflow Universe) on Colab.
    """

    def __init__(self, weights="yolov8s.pt", imgsz=960, conf=0.25):
        self.model = YOLO(weights)
        self.device = get_device()
        self.imgsz = imgsz
        self.conf = conf

    def __call__(self, frame):
        result = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        det = sv.Detections.from_ultralytics(result)
        players = det[det.class_id == COCO_PERSON]
        ball = det[det.class_id == COCO_BALL]
        return players, ball
