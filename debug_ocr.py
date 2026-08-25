"""Script de debug pour visualiser ce que voit l'OCR."""

import cv2
import os
from vehicle_pipeline import VehiclePipeline

# Initialiser le pipeline
pipeline = VehiclePipeline()

# Créer le dossier de debug s'il n'existe pas
debug_dir = "debug_ocr_crops"
os.makedirs(debug_dir, exist_ok=True)

# Charger la vidéo
video_path = "Enregistrement 2026-07-24 100339.mp4"
cap = cv2.VideoCapture(video_path)

# Frames d'intérêt
test_frames = [20, 30, 40, 50, 60, 90, 100, 120, 150, 180]

print("=" * 80)
print("DEBUG OCR - Sauvegarde des crops de plaques")
print("=" * 80)

for frame_num in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    print(f"\n📷 Frame {frame_num}")
    
    # Détecter les véhicules
    vehicle_boxes = pipeline._detect_vehicles(frame, confidence_threshold=0.25)
    
    if not vehicle_boxes:
        print("   ⚠ Aucun véhicule détecté")
        continue
    
    for idx, (x1, y1, x2, y2) in enumerate(vehicle_boxes[:1]):
        vehicle_crop = frame[y1:y2, x1:x2]
        
        # Sauvegarder le crop du véhicule
        vehicle_path = f"{debug_dir}/frame_{frame_num:04d}_vehicle.jpg"
        cv2.imwrite(vehicle_path, vehicle_crop)
        
        # Détecter la plaque avec le modèle YOLO
        results = pipeline.plate_model.predict(
            vehicle_crop,
            conf=0.3,
            device=pipeline.device,
            verbose=False,
        )
        
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_box = max(boxes, key=lambda box: float(box.conf[0]))
            px1, py1, px2, py2 = map(int, best_box.xyxy[0])
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            
            # Sauvegarder le crop de la plaque
            plate_path = f"{debug_dir}/frame_{frame_num:04d}_plate_original.jpg"
            cv2.imwrite(plate_path, plate_crop)
            
            # Sauvegarder les versions prétraitées
            preprocessed_versions = pipeline._preprocess_plate_crop(plate_crop)
            for i, preprocessed in enumerate(preprocessed_versions):
                preprocessed_path = f"{debug_dir}/frame_{frame_num:04d}_plate_preprocessed_{i}.jpg"
                cv2.imwrite(preprocessed_path, preprocessed)
            
            # Tester l'OCR
            plate_text = pipeline._extract_plate_text(plate_crop)
            print(f"   ✓ Plaque détectée: {plate_text}")
            print(f"   📁 Crops sauvegardés dans {debug_dir}/")
        else:
            print(f"   ✗ Plaque non détectée par le modèle YOLO")

cap.release()

print(f"\n" + "=" * 80)
print(f"Tous les crops sauvegardés dans: {debug_dir}/")
print("=" * 80)
