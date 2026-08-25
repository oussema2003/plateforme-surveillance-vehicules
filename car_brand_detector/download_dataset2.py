"""
Télécharge UNIQUEMENT le dataset car_brand_detection-gbnlb depuis Roboflow
"""
from roboflow import Roboflow

ROBOFLOW_API_KEY = "Ss0AyGnvVeeZIJStca7n"   # <-- ta clé API

WORKSPACE = "fatince"
PROJECT_NAME = "car_brand_detection-gbnlb"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_NAME)

# Lister les versions disponibles pour être sûr du bon numéro
print("Versions disponibles :")
for v in project.versions():
    print(f"  - {v.version}")

# ⚠️ Remplace VERSION par le numéro correct affiché ci-dessus
VERSION = 5

dataset = project.version(VERSION).download("yolov8", location="./dataset_gbnlb")
print(f"\n✅ Dataset téléchargé dans : {dataset.location}")