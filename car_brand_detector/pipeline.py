"""
Pipeline complet : Image -> Détection voiture -> Crop face avant -> Détection marque
Point d'entrée principal à utiliser par l'application/l'agent.
"""
import cv2
import numpy as np
from datetime import datetime, timezone

from car_detector import CarDetector
from logo_detector import LogoDetector
import config


def crop_front_region(bbox, image_shape,
                       width_ratio=None, height_ratio=None):
    """
    Recadre la zone centrale de la bbox du véhicule, là où se trouve
    quasi systématiquement le logo (face avant, centré).
    """
    width_ratio = width_ratio or config.FRONT_CROP_WIDTH_RATIO
    height_ratio = height_ratio or config.FRONT_CROP_HEIGHT_RATIO

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2

    new_w, new_h = w * width_ratio, h * height_ratio

    img_h, img_w = image_shape[:2]
    nx1 = int(max(0, cx - new_w / 2))
    nx2 = int(min(img_w, cx + new_w / 2))
    ny1 = int(max(0, cy - new_h / 2))
    ny2 = int(min(img_h, cy + new_h / 2))

    return (nx1, ny1, nx2, ny2)


class CarBrandPipeline:
    def __init__(self):
        self.car_detector = CarDetector()
        self.logo_detector = LogoDetector()

    def analyze_image(self, image_path: str) -> dict:
        """
        Point d'entrée principal. Retourne un dict prêt à être
        stocké dans la base de ressources.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable ou illisible : {image_path}")

        cars = self.car_detector.detect(image)
        detections = []

        for car in cars:
            x1, y1, x2, y2 = car["bbox"]
            fx1, fy1, fx2, fy2 = crop_front_region((x1, y1, x2, y2), image.shape)

            crop = image[fy1:fy2, fx1:fx2]
            logos = self.logo_detector.detect(crop)

            brand_result = None
            if logos:
                best = logos[0]
                lx1, ly1, lx2, ly2 = best["bbox"]
                # Conversion des coordonnées du crop -> coordonnées de l'image originale
                abs_bbox = [fx1 + lx1, fy1 + ly1, fx1 + lx2, fy1 + ly2]
                brand_result = {
                    "brand": best["brand"],
                    "confidence": best["confidence"],
                    "logo_bbox": abs_bbox,
                }

            detections.append({
                "car_bbox": car["bbox"],
                "car_confidence": car["confidence"],
                "front_crop_bbox": [fx1, fy1, fx2, fy2],
                "brand_detection": brand_result,
            })

        return {
            "image_path": image_path,
            "model_version": config.MODEL_VERSION,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "num_vehicles_detected": len(cars),
            "detections": detections,
        }

    def analyze_batch(self, image_paths: list[str]) -> list[dict]:
        return [self.analyze_image(p) for p in image_paths]


# ---- Exemple d'utilisation standalone ----
if __name__ == "__main__":
    import sys
    import json

    pipeline = CarBrandPipeline()
    result = pipeline.analyze_image(sys.argv[1] if len(sys.argv) > 1 else "test.jpg")
    print(json.dumps(result, indent=2, ensure_ascii=False))