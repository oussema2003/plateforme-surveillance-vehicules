"""
Configuration pour l'entraînement sur le dataset gbnlb uniquement
"""

DATASET_PATH = "./dataset_gbnlb"

MODEL_BASE = "yolo26m.pt"

# Stage 1 : entraînement principal
EPOCHS_STAGE1 = 100
IMGSZ_STAGE1 = 640
BATCH_STAGE1 = 16

# Stage 2 : fine-tuning haute résolution
EPOCHS_STAGE2 = 40
IMGSZ_STAGE2 = 896
BATCH_STAGE2 = 8

PROJECT_DIR = "car_logos_gbnlb"
NAME_STAGE1 = "stage1"
NAME_STAGE2 = "stage2_finetune"

WORKERS = 4
CACHE = "ram"

FINAL_MODEL_DEST = "./models/logo_detector_best.pt"