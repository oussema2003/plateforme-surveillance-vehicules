# TÂCHE : Intégration du module de détection de marque de voiture

## Contexte
Un module Python `car_brand_detector` a été développé. Il analyse une image
de voiture et retourne la marque détectée (logo), avec un score de confiance
et les coordonnées de la boîte englobante.

## Objectif
Intégrer ce module dans le backend du site marketing existant, afin que
chaque nouvelle ressource (image) ajoutée à la base de ressources soit
automatiquement analysée, et que le résultat (marque détectée) soit
enregistré comme métadonnée de cette ressource.

## API disponible
Le module expose une API REST (`api.py`, FastAPI) avec les endpoints suivants :

### POST /analyze
- Input : fichier image (multipart/form-data)
- Output (JSON) :
```json
{
  "image_path": "string",
  "model_version": "car-brand-detector-v1.0",
  "analyzed_at": "2026-07-19T12:00:00Z",
  "num_vehicles_detected": 1,
  "detections": [
    {
      "car_bbox": [120, 45, 480, 360],
      "car_confidence": 0.91,
      "front_crop_bbox": [180, 90, 420, 300],
      "brand_detection": {
        "brand": "BMW",
        "confidence": 0.87,
        "logo_bbox": [250, 150, 300, 190]
      }
    }
  ]
}