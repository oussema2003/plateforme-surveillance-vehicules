# Guide Complet: Enregistrement des Données de Véhicules

## 📋 Vue d'ensemble

Ce système capture et enregistre automatiquement **3 informations clés** sur chaque véhicule détecté:
- **Série/Plaque** (numéro d'immatriculation) - via OCR sur le modèle `best.pt`
- **Marque** - via le modèle `brand_yolo.pt`
- **Couleur** - via le classificateur `color_classifier.py`

Les données sont persistées dans une **base de données SQLite** avec historique complet.

---

## 🛠️ Modules Créés

### 1. `vehicle_recorder.py` (cœur du système)

Contient 4 classes principales:

#### `VehicleData` (dataclass)
Représente les données d'un véhicule:
```python
@dataclass
class VehicleData:
    plaque: str                    # Ex: "AB-123-CD"
    marque: str                    # Ex: "Toyota"
    couleur: str                   # Ex: "noir"
    confiance_plaque: float        # 0-1
    confiance_marque: float        # 0-1
    confiance_couleur: float       # 0-1
    timestamp: str                 # ISO 8601
    source_image: Optional[str]    # Où le photo a été prise
    hsv_mean: Optional[tuple]      # Données debug couleur
```

#### `VehicleRecorder` (gestion base de données)
Gère la persistence SQLite avec:
- ✅ Tables: `vehicles` + `vehicle_history`
- ✅ Méthodes: `save_vehicle()`, `get_vehicle()`, `get_all_vehicles()`
- ✅ Export: JSON et CSV
- ✅ Stats: marques/couleurs les plus fréquentes

**Utilisation basique:**
```python
from vehicle_recorder import VehicleRecorder, VehicleData
from datetime import datetime

# Initialiser le recorder
recorder = VehicleRecorder("data/vehicles.db")

# Créer un véhicule
vehicle = VehicleData(
    plaque="AB-123-CD",
    marque="Toyota",
    couleur="noir",
    confiance_plaque=0.95,
    confiance_marque=0.92,
    confiance_couleur=0.88,
    timestamp=datetime.now().isoformat(),
)

# Enregistrer
vehicle_id = recorder.save_vehicle(vehicle)

# Récupérer
v = recorder.get_vehicle("AB-123-CD")

# Lister tous
all_vehicles = recorder.get_all_vehicles()

# Exporter
recorder.export_to_json("export.json")
recorder.export_to_csv("export.csv")

# Stats
recorder.print_statistics()
```

#### `VehicleDetectionPipeline` (détection intégrée)
Pipeline automatisé qui:
1. Détecte la plaque avec `plate_model`
2. Détecte la marque avec `brand_model`
3. Classe la couleur avec `ColorClassifier`
4. Enregistre automatiquement en BD

**Utilisation:**
```python
from vehicle_recorder import VehicleDetectionPipeline

pipeline = VehicleDetectionPipeline(
    plate_model_path="models/best.pt",
    brand_model_path="models/brand_yolo.pt",
)

# Traiter une image
vehicles = pipeline.process_image("photo.jpg", confidence_threshold=0.5)

# Ou traiter une frame vidéo
import cv2
frame = cv2.imread("frame.jpg")
vehicles = pipeline.process_frame(frame)

# Les données sont automatiquement enregistrées!
```

---

## 🚀 Utilisation (3 méthodes)

### Méthode 1: Script Python Direct

**example_vehicle_detection.py** - Exemples complets:

```bash
python example_vehicle_detection.py
```

Cela va:
1. Traiter toutes les images du dossier `data/`
2. Enregistrer les véhicules détectés
3. Afficher les stats
4. Exporter en JSON/CSV

**Exemples de code supplémentaires dans le fichier:**
```python
# Traiter une vidéo
process_video("test.mp4")

# Filtrer par couleur
vehicles_noirs = query_vehicles_by_color("noir")

# Filtrer par marque
vehicles_toyota = query_vehicles_by_brand("Toyota")
```

### Méthode 2: Interface CLI

**cli_vehicles.py** - Outil en ligne de commande complet:

```bash
# Traiter une image
python cli_vehicles.py image photo.jpg --confidence 0.5

# Traiter une vidéo
python cli_vehicles.py video test.mp4 --frame-skip 10 --confidence 0.5

# Lister les véhicules
python cli_vehicles.py list --limit 20

# Afficher un véhicule
python cli_vehicles.py get "AB-123-CD"

# Statistiques
python cli_vehicles.py stats

# Exporter
python cli_vehicles.py export --format all --output data/export

# Filtrer par couleur
python cli_vehicles.py filter --color noir

# Filtrer par marque
python cli_vehicles.py filter --brand Toyota

# Filtrer par confiance minimum
python cli_vehicles.py filter --confidence 0.85
```

### Méthode 3: Intégration dans votre Code Existant

Si vous avez déjà du code, intégrez simplement:

```python
from vehicle_recorder import VehicleDetectionPipeline

# Votre pipeline existant...
# ... détection plaque, marque, couleur ...

# Enregistrer les résultats:
from datetime import datetime
from vehicle_recorder import VehicleData, VehicleRecorder

recorder = VehicleRecorder("data/vehicles.db")

vehicle = VehicleData(
    plaque=detected_plaque,
    marque=detected_brand,
    couleur=detected_color,
    confiance_plaque=conf_plate,
    confiance_marque=conf_brand,
    confiance_couleur=conf_color,
    timestamp=datetime.now().isoformat(),
    source_image=image_path,
)

recorder.save_vehicle(vehicle)
```

---

## 📊 Structure de la Base de Données

### Table `vehicles`
```sql
CREATE TABLE vehicles (
    id INTEGER PRIMARY KEY,
    plaque TEXT NOT NULL UNIQUE,      -- "AB-123-CD"
    marque TEXT NOT NULL,              -- "Toyota"
    couleur TEXT NOT NULL,             -- "noir"
    confiance_plaque REAL,             -- 0.95
    confiance_marque REAL,             -- 0.92
    confiance_couleur REAL,            -- 0.88
    timestamp TEXT NOT NULL,           -- "2024-01-15T10:30:45.123456"
    source_image TEXT,                 -- "/path/to/image.jpg"
    hsv_mean TEXT,                     -- JSON: [0.0, 0.0, 50.0]
    created_at TEXT,                   -- Créé le
    updated_at TEXT                    -- Mise à jour
);
```

### Table `vehicle_history`
```sql
CREATE TABLE vehicle_history (
    id INTEGER PRIMARY KEY,
    vehicle_id INTEGER,               -- Lien vers vehicles.id
    action TEXT,                      -- "insert", "update", "delete"
    timestamp TEXT,                   -- Quand
    details TEXT                      -- Description
);
```

---

## 📈 Export des Données

### Format JSON (`vehicles_export.json`)
```json
{
  "export_timestamp": "2024-01-15T10:35:00.123456",
  "total_vehicles": 3,
  "vehicles": [
    {
      "plaque": "AB-123-CD",
      "marque": "Toyota",
      "couleur": "noir",
      "confiance_plaque": 0.95,
      "confiance_marque": 0.92,
      "confiance_couleur": 0.88,
      "timestamp": "2024-01-15T10:30:45.123456",
      "source_image": "test.jpg",
      "hsv_mean": [0.0, 0.0, 50.0]
    },
    ...
  ]
}
```

### Format CSV (`vehicles_export.csv`)
```csv
plaque,marque,couleur,confiance_plaque,confiance_marque,confiance_couleur,timestamp,source_image
AB-123-CD,Toyota,noir,0.95,0.92,0.88,2024-01-15T10:30:45.123456,test.jpg
...
```

---

## 📊 Statistiques Disponibles

```python
from vehicle_recorder import VehicleRecorder

recorder = VehicleRecorder("data/vehicles.db")
stats = recorder.get_statistics()

print(stats)
# {
#     'total_vehicles': 42,
#     'top_marques': {'Toyota': 8, 'BMW': 6, 'Mercedes': 5, ...},
#     'top_couleurs': {'noir': 15, 'blanc': 12, 'gris': 10, ...},
#     'export_timestamp': '2024-01-15T10:35:00.123456'
# }
```

Affichage formaté:
```python
recorder.print_statistics()
```

---

## 🔧 Configuration

Variables d'environnement (optionnelles):

```bash
# Base de données
export VEHICLES_DB="data/vehicles.db"

# Modèles YOLO
export PLATE_MODEL="models/best.pt"
export BRAND_MODEL="models/brand_yolo.pt"

# Seuil de confiance par défaut
export CONFIDENCE_THRESHOLD="0.5"
```

---

## 🎯 Cas d'Usage Typiques

### 1. **Surveillance caméra continue**
```python
pipeline = VehicleDetectionPipeline(...)

# Traiter chaque frame d'une vidéo de surveillance
for frame in camera_stream:
    vehicles = pipeline.process_frame(frame)
    # Automatiquement enregistré en BD
```

### 2. **Batch processing (traiter des images)**
```bash
python cli_vehicles.py image image1.jpg
python cli_vehicles.py image image2.jpg
python cli_vehicles.py image image3.jpg

# Puis voir tous
python cli_vehicles.py list
```

### 3. **Recherche de véhicules spécifiques**
```bash
# Tous les Toyota
python cli_vehicles.py filter --brand Toyota

# Tous les véhicules noirs
python cli_vehicles.py filter --color noir

# Véhicules détectés avec 90%+ de confiance
python cli_vehicles.py filter --confidence 0.9
```

### 4. **Rapports périodiques**
```python
# Script quotidien
recorder = VehicleRecorder("data/vehicles.db")
recorder.export_to_csv(f"rapports/vehicles_{date.today()}.csv")
recorder.print_statistics()
```

---

## ⚠️ Points Importants

1. **Unicité par plaque**: Chaque plaque n'apparaît qu'une fois en BD. Détections ultérieures **mettent à jour** l'enregistrement.

2. **Historique complet**: Chaque insert/update est tracé dans `vehicle_history`.

3. **Performance OCR**: L'OCR (easyocr) peut être lent. Pour les vidéos, c'est normal de traiter 1 frame sur 10.

4. **GPU**: Les modèles YOLO fonctionnent sur GPU si disponible. Sinon, CPU (plus lent).

5. **Données debug**: `hsv_mean` contient les données HSV brutes utiles pour debug/calibration couleur.

---

## 📝 Prochaines Étapes

Optionnel mais utile:

1. **Intégration API REST**: Utiliser FastAPI (voir `api.py` existant)
2. **Dashboard Web**: Visualiser les statistiques en temps réel
3. **Alertes**: Notifier si véhicule recherché détecté
4. **Stockage images**: Sauvegarder les crops de plaques/marques pour audit

---

## ❓ FAQ

**Q: Comment améliorer la détection de marque?**
A: Retrain `brand_yolo.pt` sur vos données locales si les résultats ne sont pas bons.

**Q: Peut-on utiliser avec plusieurs caméras?**
A: Oui! Chaque caméra lance son pipeline, tous écrivent dans la même `vehicles.db`.

**Q: Comment gérer les faux positifs?**
A: Ajuster `confidence_threshold` ou filtrer par `confiance_plaque >= 0.9` au query.

**Q: Où sont stockées les images?**
A: Actuellement, seul le chemin source est enregistré. Implémenter un stockage d'images si besoin.

---

## 📞 Support

Pour des questions ou problèmes, consultez:
- Code commenté dans `vehicle_recorder.py`
- Exemples dans `example_vehicle_detection.py`
- Aide CLI: `python cli_vehicles.py -h`
