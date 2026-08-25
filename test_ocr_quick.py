"""Test rapide de l'OCR sur des frames spécifiques de la vidéo."""

import cv2
from vehicle_pipeline import VehiclePipeline

# Initialiser le pipeline
pipeline = VehiclePipeline()

# Charger la vidéo
video_path = "Enregistrement 2026-07-24 100339.mp4"
cap = cv2.VideoCapture(video_path)

# Frames d'intérêt (où les plaques sont visibles)
test_frames = [
    (30, "255 3466"),   # Plaque Volkswagen grise
    (90, "169 2431"),   # Plaque Volkswagen noire
    (50, "208 7725"),   # Plaque noire (si visible)
    (120, "114 8706"),  # Plaque Peugeot (si visible)
]

print("=" * 80)
print("TEST OCR SUR FRAMES SPÉCIFIQUES")
print("=" * 80)

for frame_num, expected_plate in test_frames:
    # Aller à la frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        print(f"\n❌ Frame {frame_num}: Impossible de lire")
        continue
    
    print(f"\n📷 Frame {frame_num} (attendu: {expected_plate})")
    print("-" * 80)
    
    # Détecter les véhicules
    vehicle_boxes = pipeline._detect_vehicles(frame, confidence_threshold=0.25)
    
    if not vehicle_boxes:
        print("   ⚠ Aucun véhicule détecté")
        continue
    
    for idx, (x1, y1, x2, y2) in enumerate(vehicle_boxes[:1]):  # Prendre seulement le plus grand
        vehicle_crop = frame[y1:y2, x1:x2]
        
        # Détecter la plaque
        plate_text, confidence = pipeline._detect_plate_in_crop(
            vehicle_crop,
            confidence_threshold=0.3
        )
        
        if plate_text:
            match = "✓" if plate_text == expected_plate else "✗"
            print(f"   {match} OCR détecté: {plate_text} (confiance: {confidence:.2f})")
        else:
            print(f"   ✗ Plaque non détectée")

cap.release()

print("\n" + "=" * 80)
print("Test terminé")
print("=" * 80)
