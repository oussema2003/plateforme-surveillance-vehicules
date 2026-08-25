"""
Étape 1 : Détection de véhicules dans une image.
Utilise un modèle YOLO pré-entraîné COCO (pas besoin d'entraînement,
ce modèle sait déjà détecter des voitures/bus/camions).
"""
from ultralytics import YOLO
import numpy as np

import config


class CarDetector:
    def __init__(self, model_path=None, conf=None):
        self.model = YOLO(model_path or config.VEHICLE_MODEL_PATH)
        self.conf = conf or config.CAR_CONF_THRESHOLD
        self.class_ids = list(config.COCO_VEHICLE_CLASS_IDS.keys())

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Retourne une liste de véhicules détectés :
        [{"bbox": [x1,y1,x2,y2], "confidence": float, "class": str}, ...]
        """
        results = self.model.predict(
            image,
            conf=self.conf,
            classes=self.class_ids,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 4),
                    "class": config.COCO_VEHICLE_CLASS_IDS.get(cls_id, str(cls_id)),
                })

        return detections