"""
example_vehicle_detection.py
-----------------------------
Exemple complet d'utilisation du pipeline de détection de véhicules.
Détecte : plaque + marque + couleur et enregistre dans la base de données.
"""

from pathlib import Path
import cv2
from vehicle_recorder import VehicleDetectionPipeline, VehicleRecorder


def main():
    """Exemple d'utilisation du pipeline complet."""
    
    # 1. Initialiser le recorder (base de données SQLite)
    recorder = VehicleRecorder("data/vehicles.db")
    
    # 2. Initialiser le pipeline de détection
    pipeline = VehicleDetectionPipeline(
        plate_model_path="models/best.pt",      # Modèle détection plaque
        brand_model_path="models/brand_yolo.pt", # Modèle détection marque
        recorder=recorder,
    )
    
    # 3. Traiter les images du dossier de test
    test_images_dir = Path("data")
    
    if test_images_dir.exists():
        for image_file in test_images_dir.glob("*.jpg"):
            print(f"\n📸 Traitement: {image_file.name}")
            
            vehicles = pipeline.process_image(
                str(image_file),
                confidence_threshold=0.5,
            )
            
            if vehicles:
                print(f"   ✓ {len(vehicles)} véhicule(s) détecté(s)")
                for vehicle in vehicles:
                    print(f"     - Plaque: {vehicle.plaque}")
                    print(f"       Marque: {vehicle.marque}")
                    print(f"       Couleur: {vehicle.couleur}")
                    print(f"       Confiances: plaque={vehicle.confiance_plaque:.2%}, "
                          f"marque={vehicle.confiance_marque:.2%}, "
                          f"couleur={vehicle.confiance_couleur:.2%}")
            else:
                print(f"   ❌ Aucun véhicule détecté")
    
    # 4. Afficher les statistiques
    recorder.print_statistics()
    
    # 5. Exporter les données
    print("\n📊 Export des données...")
    recorder.export_to_json("data/vehicles_export.json")
    recorder.export_to_csv("data/vehicles_export.csv")


def process_video(video_path: str, output_path: str = "data/vehicles_from_video.json"):
    """Exemple: traiter une vidéo et enregistrer les véhicules détectés."""
    
    recorder = VehicleRecorder("data/vehicles.db")
    
    pipeline = VehicleDetectionPipeline(
        plate_model_path="models/best.pt",
        brand_model_path="models/brand_yolo.pt",
        recorder=recorder,
    )
    
    print(f"\n🎬 Traitement vidéo: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detected_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Traiter chaque frame (adapter la fréquence selon vos besoins)
        if frame_count % 10 == 0:  # Traiter 1 frame sur 10
            vehicles = pipeline.process_frame(
                frame,
                confidence_threshold=0.5,
                source_image=f"{video_path}@frame_{frame_count}",
            )
            
            if vehicles:
                detected_count += len(vehicles)
                for vehicle in vehicles:
                    print(f"   Frame {frame_count}: {vehicle.plaque} ({vehicle.marque}, {vehicle.couleur})")
    
    cap.release()
    
    print(f"\n✓ Traitement terminé:")
    print(f"   Total frames: {frame_count}")
    print(f"   Véhicules détectés: {detected_count}")
    
    # Exporter
    recorder.export_to_json(output_path)
    recorder.print_statistics()


def query_vehicles_by_color(color: str = "noir"):
    """Exemple: requêter les véhicules par couleur."""
    
    recorder = VehicleRecorder("data/vehicles.db")
    vehicles = recorder.get_all_vehicles()
    
    matching = [v for v in vehicles if color.lower() in v.couleur.lower()]
    
    print(f"\n🎨 Véhicules {color}:")
    for vehicle in matching:
        print(f"   {vehicle.plaque}: {vehicle.marque} ({vehicle.couleur})")
    
    return matching


def query_vehicles_by_brand(brand: str = "Toyota"):
    """Exemple: requêter les véhicules par marque."""
    
    recorder = VehicleRecorder("data/vehicles.db")
    vehicles = recorder.get_all_vehicles()
    
    matching = [v for v in vehicles if brand.lower() in v.marque.lower()]
    
    print(f"\n🏷️ Véhicules {brand}:")
    for vehicle in matching:
        print(f"   {vehicle.plaque}: {vehicle.marque} ({vehicle.couleur})")
    
    return matching


if __name__ == "__main__":
    # Exécuter l'exemple principal
    main()
    
    # Exemples supplémentaires (décommenter pour utiliser)
    
    # Traiter une vidéo:
    # process_video("test.mp4")
    
    # Requêter par couleur:
    # query_vehicles_by_color("noir")
    
    # Requêter par marque:
    # query_vehicles_by_brand("Toyota")
