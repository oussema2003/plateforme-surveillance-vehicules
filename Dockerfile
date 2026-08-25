# Image CPU par defaut. Pour du GPU, voir la note en bas de fichier.
FROM python:3.11-slim

# Dependances systeme necessaires a OpenCV et a la lecture de flux RTSP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY color_classifier.py vehicle_pipeline.py api.py camera_manager.py ./
COPY static/ ./static/
# Les poids sont copies ici a la construction, ou montes en volume (voir
# docker-compose.yml) si vous preferez les mettre a jour sans reconstruire
# l'image.
COPY models/ ./models/

RUN mkdir -p /data
ENV OUTPUT_PATH=/data/vehicules.json

EXPOSE 8000

# Meme image pour les deux usages -- pas d'ENTRYPOINT fige, seulement une
# CMD par defaut (mode worker RTSP) qui peut etre entierement remplacee :
#   docker run ... reco-vehicule python vehicle_pipeline.py --source rtsp://...
#   docker run ... reco-vehicule uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["python", "vehicle_pipeline.py", "--source", "0", "--log", "/data/vehicules.json", "--no-display"]

# --------------------------------------------------------------------------
# Variante GPU (Jetson / serveur avec carte NVIDIA) : remplacez la ligne
# FROM ci-dessus par une image CUDA, par exemple :
#   FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# puis installez python3/pip (non presents dans cette image de base) et
# lancez le conteneur avec --gpus all (ou la section "deploy" du
# docker-compose.yml fournie).
# --------------------------------------------------------------------------
