"""
API REST exposant le pipeline de détection.
L'agent peut appeler cet endpoint depuis le backend du site marketing
pour analyser une image et ajouter le résultat à la base de ressources.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import uuid
import os

from pipeline import CarBrandPipeline

app = FastAPI(title="Car Brand Detection API", version="1.0")
pipeline = CarBrandPipeline()


class AnalyzeResponse(BaseModel):
    image_path: str
    model_version: str
    analyzed_at: str
    num_vehicles_detected: int
    detections: list


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    Reçoit une image, retourne les marques détectées.
    À appeler depuis le backend marketing lors de l'ajout
    d'une nouvelle ressource (image) à la base.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Le fichier doit être une image")

    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        result = pipeline.analyze_image(tmp_path)
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/analyze-url")
async def analyze_from_url(image_url: str):
    """
    Variante : analyse une image depuis une URL
    (utile si les ressources marketing sont déjà hébergées ailleurs).
    """
    import requests

    tmp_path = f"/tmp/{uuid.uuid4()}.jpg"
    try:
        r = requests.get(image_url, timeout=10)
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(r.content)
        result = pipeline.analyze_image(tmp_path)
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)