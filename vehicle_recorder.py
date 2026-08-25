"""
vehicle_recorder.py
-------------------
Enregistrement des données de reconnaissance des véhicules :
  - Numéro de série/plaque (détecté par best.pt)
  - Marque du véhicule (détecté par brand_yolo.pt)
  - Couleur du véhicule (classifié par color_classifier.py)

Utilise une base de données SQLite pour stocker les informations et générer
des rapports en JSON/CSV.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from color_classifier import ColorClassifier, ColorResult


@dataclass
class VehicleData:
    """Représente les données complètes d'un véhicule détecté."""
    plaque: str  # Numéro de série/plaque
    marque: str  # Marque du véhicule
    couleur: str  # Couleur du véhicule
    confiance_plaque: float  # Confiance OCR plaque (0-1)
    confiance_marque: float  # Confiance détection marque (0-1)
    confiance_couleur: float  # Confiance classification couleur (0-1)
    timestamp: str  # Horodatage de détection (ISO 8601)
    source_image: Optional[str] = None  # Chemin ou URL source
    hsv_mean: Optional[tuple] = None  # Données debug couleur HSV


class VehicleRecorder:
    """Gestionnaire de base de données pour enregistrer les véhicules."""

    def __init__(self, db_path: str = "data/vehicles.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Crée les tables SQLite si elles n'existent pas."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table des véhicules détectés
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plaque TEXT NOT NULL UNIQUE,
                    marque TEXT NOT NULL,
                    couleur TEXT NOT NULL,
                    confiance_plaque REAL,
                    confiance_marque REAL,
                    confiance_couleur REAL,
                    timestamp TEXT NOT NULL,
                    source_image TEXT,
                    hsv_mean TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table d'historique (pour tracker les changements)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id INTEGER NOT NULL,
                    action TEXT NOT NULL,  -- 'insert', 'update', 'delete'
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id)
                )
            """)
            
            conn.commit()

    def save_vehicle(self, vehicle: VehicleData) -> int:
        """Enregistre ou met à jour un véhicule. Retourne l'ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # Essayer d'insérer
                cursor.execute("""
                    INSERT INTO vehicles 
                    (plaque, marque, couleur, confiance_plaque, confiance_marque,
                     confiance_couleur, timestamp, source_image, hsv_mean)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    vehicle.plaque,
                    vehicle.marque,
                    vehicle.couleur,
                    vehicle.confiance_plaque,
                    vehicle.confiance_marque,
                    vehicle.confiance_couleur,
                    vehicle.timestamp,
                    vehicle.source_image,
                    json.dumps(vehicle.hsv_mean) if vehicle.hsv_mean else None,
                ))
                
                vehicle_id = cursor.lastrowid
                
                # Enregistrer dans l'historique
                cursor.execute("""
                    INSERT INTO vehicle_history (vehicle_id, action, timestamp, details)
                    VALUES (?, ?, ?, ?)
                """, (
                    vehicle_id,
                    'insert',
                    datetime.now().isoformat(),
                    f"Nouveau véhicule: {vehicle.plaque} - {vehicle.marque} - {vehicle.couleur}"
                ))
                
                conn.commit()
                return vehicle_id
                
            except sqlite3.IntegrityError:
                # Plaque déjà existe, mettre à jour
                cursor.execute("""
                    SELECT id FROM vehicles WHERE plaque = ?
                """, (vehicle.plaque,))
                
                result = cursor.fetchone()
                if result:
                    vehicle_id = result[0]
                    cursor.execute("""
                        UPDATE vehicles 
                        SET marque = ?, couleur = ?, confiance_plaque = ?,
                            confiance_marque = ?, confiance_couleur = ?,
                            timestamp = ?, source_image = ?, hsv_mean = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (
                        vehicle.marque,
                        vehicle.couleur,
                        vehicle.confiance_plaque,
                        vehicle.confiance_marque,
                        vehicle.confiance_couleur,
                        vehicle.timestamp,
                        vehicle.source_image,
                        json.dumps(vehicle.hsv_mean) if vehicle.hsv_mean else None,
                        vehicle_id,
                    ))
                    
                    # Enregistrer dans l'historique
                    cursor.execute("""
                        INSERT INTO vehicle_history (vehicle_id, action, timestamp, details)
                        VALUES (?, ?, ?, ?)
                    """, (
                        vehicle_id,
                        'update',
                        datetime.now().isoformat(),
                        f"Mise à jour: {vehicle.plaque} - {vehicle.marque} - {vehicle.couleur}"
                    ))
                    
                    conn.commit()
                    return vehicle_id

    def get_vehicle(self, plaque: str) -> Optional[VehicleData]:
        """Récupère les données d'un véhicule par sa plaque."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vehicles WHERE plaque = ?", (plaque,))
            row = cursor.fetchone()
            
            if row:
                hsv = json.loads(row['hsv_mean']) if row['hsv_mean'] else None
                return VehicleData(
                    plaque=row['plaque'],
                    marque=row['marque'],
                    couleur=row['couleur'],
                    confiance_plaque=row['confiance_plaque'],
                    confiance_marque=row['confiance_marque'],
                    confiance_couleur=row['confiance_couleur'],
                    timestamp=row['timestamp'],
                    source_image=row['source_image'],
                    hsv_mean=tuple(hsv) if hsv else None,
                )
            return None

    def get_all_vehicles(self, limit: int = None) -> list[VehicleData]:
        """Récupère tous les véhicules enregistrés."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if limit:
                cursor.execute("SELECT * FROM vehicles ORDER BY updated_at DESC LIMIT ?", (limit,))
            else:
                cursor.execute("SELECT * FROM vehicles ORDER BY updated_at DESC")
            
            rows = cursor.fetchall()
            vehicles = []
            
            for row in rows:
                hsv = json.loads(row['hsv_mean']) if row['hsv_mean'] else None
                vehicles.append(VehicleData(
                    plaque=row['plaque'],
                    marque=row['marque'],
                    couleur=row['couleur'],
                    confiance_plaque=row['confiance_plaque'],
                    confiance_marque=row['confiance_marque'],
                    confiance_couleur=row['confiance_couleur'],
                    timestamp=row['timestamp'],
                    source_image=row['source_image'],
                    hsv_mean=tuple(hsv) if hsv else None,
                ))
            
            return vehicles

    def export_to_json(self, output_path: str = "data/vehicles_export.json") -> None:
        """Exporte tous les véhicules en JSON."""
        vehicles = self.get_all_vehicles()
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_vehicles': len(vehicles),
            'vehicles': [asdict(v) for v in vehicles]
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Exporté {len(vehicles)} véhicules vers {output_path}")

    def export_to_csv(self, output_path: str = "data/vehicles_export.csv") -> None:
        """Exporte tous les véhicules en CSV."""
        import csv
        
        vehicles = self.get_all_vehicles()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'plaque', 'marque', 'couleur',
                'confiance_plaque', 'confiance_marque', 'confiance_couleur',
                'timestamp', 'source_image'
            ])
            writer.writeheader()
            
            for v in vehicles:
                writer.writerow({
                    'plaque': v.plaque,
                    'marque': v.marque,
                    'couleur': v.couleur,
                    'confiance_plaque': v.confiance_plaque,
                    'confiance_marque': v.confiance_marque,
                    'confiance_couleur': v.confiance_couleur,
                    'timestamp': v.timestamp,
                    'source_image': v.source_image,
                })
        
        print(f"✓ Exporté {len(vehicles)} véhicules vers {output_path}")

    def get_statistics(self) -> dict:
        """Retourne des statistiques sur les véhicules."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total de véhicules
            cursor.execute("SELECT COUNT(*) FROM vehicles")
            total = cursor.fetchone()[0]
            
            # Marques les plus fréquentes
            cursor.execute("""
                SELECT marque, COUNT(*) as count
                FROM vehicles
                GROUP BY marque
                ORDER BY count DESC
                LIMIT 10
            """)
            marques = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Couleurs les plus fréquentes
            cursor.execute("""
                SELECT couleur, COUNT(*) as count
                FROM vehicles
                GROUP BY couleur
                ORDER BY count DESC
                LIMIT 10
            """)
            couleurs = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total_vehicles': total,
                'top_marques': marques,
                'top_couleurs': couleurs,
                'export_timestamp': datetime.now().isoformat(),
            }

    def print_statistics(self) -> None:
        """Affiche les statistiques."""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("📊 STATISTIQUES DES VÉHICULES")
        print("="*60)
        print(f"Total de véhicules: {stats['total_vehicles']}")
        print(f"\n🏷️ Marques les plus fréquentes:")
        for marque, count in list(stats['top_marques'].items())[:5]:
            print(f"   {marque}: {count}")
        print(f"\n🎨 Couleurs les plus fréquentes:")
        for couleur, count in list(stats['top_couleurs'].items())[:5]:
            print(f"   {couleur}: {count}")
        print("="*60 + "\n")


class VehicleDetectionPipeline:
    """Pipeline complet de détection : plaque + marque + couleur."""

    def __init__(
        self,
        plate_model_path: str,
        brand_model_path: str,
        recorder: Optional[VehicleRecorder] = None,
    ):
        """
        Args:
            plate_model_path: Chemin vers le modèle YOLO de détection de plaque
            brand_model_path: Chemin vers le modèle YOLO de détection de marque
            recorder: Instance VehicleRecorder (en crée une nouvelle si None)
        """
        self.plate_model = YOLO(plate_model_path)
        self.brand_model = YOLO(brand_model_path)
        self.color_classifier = ColorClassifier()
        self.recorder = recorder or VehicleRecorder()

    def process_image(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
    ) -> list[VehicleData]:
        """
        Traite une image et retourne les véhicules détectés.
        Enregistre automatiquement dans la base de données.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Impossible de lire l'image: {image_path}")
            return []
        
        return self.process_frame(
            frame,
            confidence_threshold=confidence_threshold,
            source_image=image_path,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        source_image: Optional[str] = None,
    ) -> list[VehicleData]:
        """
        Traite une frame et retourne les véhicules détectés.
        """
        vehicles = []
        
        # Détection des plaques
        plate_results = self.plate_model(frame, conf=confidence_threshold)
        
        for plate_result in plate_results:
            if not plate_result.boxes:
                continue
            
            for box in plate_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_plaque = float(box.conf[0])
                
                # Extraire la région du véhicule
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size == 0:
                    continue
                
                # Détection de la marque sur cette région
                marque, conf_marque = self._detect_brand(vehicle_crop, confidence_threshold)
                
                # Classification de la couleur
                color_result = self.color_classifier.get_dominant_color(vehicle_crop)
                
                # OCR de la plaque (utiliser easyocr ou votre méthode)
                plaque = self._extract_plate_text(vehicle_crop)
                
                vehicle = VehicleData(
                    plaque=plaque or "INCONNU",
                    marque=marque or "INCONNUE",
                    couleur=color_result.name,
                    confiance_plaque=conf_plaque,
                    confiance_marque=conf_marque,
                    confiance_couleur=color_result.confidence,
                    timestamp=datetime.now().isoformat(),
                    source_image=source_image,
                    hsv_mean=color_result.hsv_mean,
                )
                
                # Enregistrer dans la base de données
                self.recorder.save_vehicle(vehicle)
                vehicles.append(vehicle)
        
        return vehicles

    def _detect_brand(
        self,
        vehicle_crop: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> tuple[str, float]:
        """Détecte la marque du véhicule dans une région."""
        results = self.brand_model(vehicle_crop, conf=confidence_threshold)
        
        if results and results[0].boxes:
            box = results[0].boxes[0]
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Récupérer le nom de la classe (marque)
            class_name = results[0].names.get(class_id, f"Classe_{class_id}")
            return class_name, conf
        
        return None, 0.0

    def _extract_plate_text(self, vehicle_crop: np.ndarray) -> Optional[str]:
        """Extrait le texte de la plaque (OCR)."""
        try:
            import easyocr
            reader = easyocr.Reader(['en', 'fr'], gpu=False)
            results = reader.readtext(vehicle_crop)
            
            if results:
                # Combiner les textes détectés
                texts = [result[1] for result in results]
                plate_text = ' '.join(texts).upper()
                return plate_text.strip()
        except Exception as e:
            print(f"⚠️ Erreur OCR: {e}")
        
        return None


if __name__ == "__main__":
    # Exemple d'utilisation
    recorder = VehicleRecorder("data/vehicles.db")
    
    # Créer un véhicule de test
    test_vehicle = VehicleData(
        plaque="AB-123-CD",
        marque="Toyota",
        couleur="noir",
        confiance_plaque=0.95,
        confiance_marque=0.92,
        confiance_couleur=0.88,
        timestamp=datetime.now().isoformat(),
        hsv_mean=(0, 0, 50),
    )
    
    # Enregistrer le véhicule
    vehicle_id = recorder.save_vehicle(test_vehicle)
    print(f"✓ Véhicule enregistré avec ID: {vehicle_id}")
    
    # Afficher les statistiques
    recorder.print_statistics()
    
    # Exporter les données
    recorder.export_to_json()
    recorder.export_to_csv()
