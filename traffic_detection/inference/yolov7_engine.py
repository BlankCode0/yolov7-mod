# traffic_detection/inference/yolov7_engine.py

import torch
import numpy as np

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.torch_utils import select_device


class YOLOv7Engine:
    """
    Inference engine wrapping a modified YOLOv7 model optimized
    for small and partially occluded vehicle detection.
    """

    def __init__(self, weights_path: str, device: str = "cuda"):
        self.device = select_device(device)
        self.model = self._load_model(weights_path)
        self.model.eval()

    def _load_model(self, weights_path: str):
        model = attempt_load(weights_path, map_location=self.device)
        return model

    @torch.no_grad()
    def infer(
        self,
        image: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45
    ):
        """
        Run inference on a single image.

        Args:
            image (np.ndarray): Preprocessed image tensor (C, H, W)
            conf_thres (float): Confidence threshold
            iou_thres (float): NMS IoU threshold

        Returns:
            List of detections: [x1, y1, x2, y2, conf, class_id]
        """

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        image = torch.from_numpy(image).to(self.device).float()
        image /= 255.0

        pred = self.model(image)[0]

        pred = non_max_suppression(
            pred,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )

        detections = []

        if pred[0] is not None and len(pred[0]):
            for *xyxy, conf, cls in pred[0]:
                detections.append([
                    int(xyxy[0]),
                    int(xyxy[1]),
                    int(xyxy[2]),
                    int(xyxy[3]),
                    float(conf),
                    int(cls)
                ])

        return detections

