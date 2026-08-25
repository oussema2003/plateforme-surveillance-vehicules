"""
api.py
------
Expose le pipeline de reconnaissance vehicule sous forme d'API HTTP, pour
etre integre facilement dans N'IMPORTE QUEL PROJET (VMS, application web,
script dans un autre langage...) sans avoir a connaitre Python/YOLO.

Endpoints :
  GET  /health                 -> verifie que le service tourne
  GET  /vehicules               -> tous les vehicules connus (JSON complet)
  GET  /vehicules/{matricule}   -> un vehicule precis (404 si inconnu)
  POST /analyze                 -> envoie une image, recoit couleur/marque/
                                    plaque en retour (et met a jour le JSON)

Lancement local :
    uvicorn api:app --host 0.0.0.0 --port 8000

Avec Docker : voir docker-compose.yml (service "api"). Le meme fichier JSON
(chemin OUTPUT_PATH) peut etre partage avec les workers RTSP qui tournent
en parallele sur chaque camera -- l'API sert alors une vue unifiee de tous
les vehicules vus par toutes les cameras.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vehicle_pipeline import (
    VehiclePipeline,
    VehicleJSONStore,
    normalize_plate_images_dir,
    plate_image_storage_path,
    resolve_plate_image_file,
)
from camera_manager import CameraManager

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", str(PROJECT_ROOT / "data" / "vehicules.json"))
VEHICLE_MODEL = os.environ.get("VEHICLE_MODEL", "models/yolo26n.pt")
PLATE_MODEL = os.environ.get("PLATE_MODEL", "models/plate_yolo.pt")
BRAND_MODEL = os.environ.get("BRAND_MODEL", "models/brand_yolo.pt")
PLATE_IMAGES_DIR = os.environ.get("PLATE_IMAGES_DIR", "data/plate_images")
PLATE_IMAGES_PATH = normalize_plate_images_dir(PLATE_IMAGES_DIR)

app = FastAPI(title="API Reconnaissance Vehicule", version="1.0")

# Les modeles sont charges UNE FOIS au demarrage du service (pas a chaque
# requete), pour que /analyze reponde vite.
pipeline = VehiclePipeline(
    vehicle_model_path=VEHICLE_MODEL,
    plate_model_path=PLATE_MODEL,
    brand_model_path=BRAND_MODEL,
    output_path=OUTPUT_PATH,
    plate_images_dir=PLATE_IMAGES_DIR,
)
# Store independant pour les lectures : permet de consulter le JSON meme
# s'il est aussi alimente par des workers RTSP externes a ce processus.
store = VehicleJSONStore(OUTPUT_PATH)

# Gestionnaire de cameras dynamiques : permet d'ajouter/arreter des flux
# RTSP (ou fichiers video) depuis le dashboard web, sans relancer Docker.
camera_manager = CameraManager(
    vehicle_model_path=VEHICLE_MODEL,
    plate_model_path=PLATE_MODEL,
    brand_model_path=BRAND_MODEL,
    output_path=OUTPUT_PATH,
    plate_images_dir=PLATE_IMAGES_DIR,
)

# Dossier des images de plaques sauvegardees par le pipeline : servi tel
# quel en statique pour que le dashboard puisse afficher les vignettes.
PLATE_IMAGES_PATH.mkdir(parents=True, exist_ok=True)
app.mount("/plate_images", StaticFiles(directory=str(PLATE_IMAGES_PATH)), name="plate_images")


def _sanitize_vehicle_record(record: dict) -> dict:
    """Expose une URL d'image uniquement si le fichier existe sur le disque."""
    out = dict(record)
    image_file = resolve_plate_image_file(
        out.get("plate_image"),
        out.get("matricule"),
        PLATE_IMAGES_PATH,
    )
    if image_file is not None:
        out["plate_image"] = plate_image_storage_path(image_file)
        out["plate_image_url"] = f"/plate_images/{image_file.name}"
    else:
        out.pop("plate_image", None)
        out.pop("plate_image_url", None)
    return out


def _repair_store_plate_images() -> None:
    """Corrige les chemins d'images invalides dans le JSON au demarrage."""
    store_path = Path(OUTPUT_PATH)
    if not store_path.is_file():
        return

    with open(store_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for record in data.values():
        if not isinstance(record, dict):
            continue
        image_file = resolve_plate_image_file(
            record.get("plate_image"),
            record.get("matricule"),
            PLATE_IMAGES_PATH,
        )
        if image_file is not None:
            fixed = plate_image_storage_path(image_file)
            if record.get("plate_image") != fixed:
                record["plate_image"] = fixed
                changed = True
        elif record.get("plate_image"):
            record.pop("plate_image", None)
            changed = True

    if changed:
        with open(store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


_repair_store_plate_images()


class CameraCreateRequest(BaseModel):
    name: str
    source: str          # ex: "rtsp://user:pass@ip:554/stream" ou "/videos/test.mp4"
    frame_skip: int = 10
    confidence: float = 0.5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/cameras")
def list_cameras():
    """Liste toutes les cameras ajoutees (en cours, arretees, en erreur...)."""
    return camera_manager.list_cameras()


@app.post("/cameras")
def add_camera(payload: CameraCreateRequest):
    """Ajoute et demarre une nouvelle camera (RTSP ou fichier video) a la volee."""
    if not payload.source.strip():
        raise HTTPException(status_code=400, detail="La source (URL RTSP ou chemin video) est requise")
    return camera_manager.add_camera(
        name=payload.name.strip() or payload.source,
        source=payload.source.strip(),
        frame_skip=payload.frame_skip,
        confidence=payload.confidence,
    )


@app.delete("/cameras/{camera_id}")
def remove_camera(camera_id: str):
    """Arrete proprement une camera et la retire de la liste."""
    removed = camera_manager.remove_camera(camera_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Camera inconnue")
    return {"status": "stopped", "id": camera_id}


@app.get("/cameras/{camera_id}/frame")
def get_camera_frame(camera_id: str):
    """Renvoie la derniere frame annotee de cette camera (JPEG), pour un
    apercu 'live' cote dashboard (rafraichi periodiquement en JS)."""
    jpeg_bytes = camera_manager.get_last_frame_jpeg(camera_id)
    if jpeg_bytes is None:
        raise HTTPException(status_code=404, detail="Pas encore de frame disponible pour cette camera")
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/vehicules")
def list_vehicules():
    with open(store.path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        key: _sanitize_vehicle_record(record)
        for key, record in data.items()
        if isinstance(record, dict)
    }


@app.get("/vehicules/{matricule}")
def get_vehicule(matricule: str):
    with open(store.path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if matricule not in data:
        raise HTTPException(status_code=404, detail="Matricule inconnu")
    return _sanitize_vehicle_record(data[matricule])


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Recoit une image (une photo ou une frame extraite ailleurs), detecte
    le(s) vehicule(s) et retourne leurs attributs. Met aussi a jour le JSON
    par matricule (meme regle de dedoublonnage horaire que le mode RTSP)."""
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        frame = cv2.imread(tmp_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Image illisible")
        summary, _annotated = pipeline.process_single_image(frame)
        return JSONResponse(content={"vehicules_detectes": summary})
    finally:
        os.remove(tmp_path)


# Remarque production : les appels YOLO sont synchrones (CPU/GPU-bound).
# Pour encaisser plusieurs requetes /analyze en parallele sans bloquer le
# serveur, lancez uvicorn avec plusieurs workers (--workers 2) ou deportez
# pipeline.process_single_image(...) dans un threadpool (fastapi.concurrency
# .run_in_threadpool). Pour un usage modere (quelques appels/seconde), la
# version simple ci-dessus suffit.


# --------------------------------------------------------------------------
# Dashboard web (static/index.html) : interface pour ajouter des cameras et
# visualiser les resultats en direct. MONTE EN DERNIER : les routes API
# definies plus haut (/cameras, /vehicules, /analyze...) restent prioritaires,
# seul ce qui ne matche aucune route API tombe sur les fichiers statiques.
# --------------------------------------------------------------------------
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="dashboard")
