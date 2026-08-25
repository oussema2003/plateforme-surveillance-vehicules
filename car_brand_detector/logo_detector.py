"""
Étape 2 : Détection de la marque (logo) à l'intérieur
d'une zone recadrée (face avant du véhicule).
Utilise TON modèle entraîné sur le dataset fusionné.
"""
from ultralytics import YOLO
import numpy as np

import config


class LogoDetector:
    def __init__(self, model_path=None, conf=None):
        self.model = YOLO(model_path or config.LOGO_MODEL_PATH)
        self.conf = conf or config.LOGO_CONF_THRESHOLD
        self.names = self.model.names

    def detect(self, image_crop: np.ndarray) -> list[dict]:
        """
        Retourne une liste de logos détectés dans le crop, triée par confiance décroissante :
        [{"brand": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
        Les bbox sont relatives au crop (pas à l'image originale).
        """
        if image_crop.size == 0:
            return []

        results = self.model.predict(
            image_crop,
            conf=self.conf,
            imgsz=config.LOGO_IMGSZ,
            augment=True,      # TTA : améliore la précision en inférence
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    "brand": self.names[cls_id],
                    "confidence": round(conf, 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

        detections.sort(key=lambda d: -d["confidence"])
        return detections