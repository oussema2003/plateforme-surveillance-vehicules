"""
Entraînement du modèle de détection de marque de voiture
Dataset : car_brand_detection-gbnlb (15 classes, 19709 images train)
"""
from ultralytics import YOLO
import torch
import yaml
import shutil
from pathlib import Path

import config as cfg

print("=" * 70)
print("ENTRAÎNEMENT - Dataset gbnlb (15 marques)")
print("=" * 70)

data_yaml_path = f"{cfg.DATASET_PATH}/data.yaml"

with open(data_yaml_path) as f:
    data_cfg = yaml.safe_load(f)

print(f"Classes ({data_cfg['nc']}) : {data_cfg['names']}")

device = 0 if torch.cuda.is_available() else "cpu"
print(f"\nDevice : {device}")
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# STAGE 1 : Entraînement principal
# ============================================================
print("\n=== STAGE 1 : Entraînement principal ===")

model = YOLO(cfg.MODEL_BASE)

model.train(
    data=data_yaml_path,
    epochs=cfg.EPOCHS_STAGE1,
    imgsz=cfg.IMGSZ_STAGE1,
    batch=cfg.BATCH_STAGE1,
    device=device,
    patience=25,
    project=cfg.PROJECT_DIR,
    name=cfg.NAME_STAGE1,
    optimizer="AdamW",

    lr0=0.0015,
    lrf=0.01,
    warmup_epochs=5,
    cos_lr=True,
    momentum=0.9,
    weight_decay=0.0005,

    box=7.5,
    cls=0.8,          # 15 classes bien réparties -> pas besoin de cls élevé
    dfl=1.5,

    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.0,
    degrees=8.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.0,
    fliplr=0.5,

    close_mosaic=15,
    dropout=0.0,
    freeze=0,
    multi_scale=False,
    cache=cfg.CACHE,
    workers=cfg.WORKERS,
    seed=42,
    plots=True,
    amp=True,
)

# ============================================================
# STAGE 2 : Fine-tuning haute résolution
# ============================================================
print("\n=== STAGE 2 : Fine-tuning ===")

best_stage1 = f"{cfg.PROJECT_DIR}/{cfg.NAME_STAGE1}/weights/best.pt"
model2 = YOLO(best_stage1)

model2.train(
    data=data_yaml_path,
    epochs=cfg.EPOCHS_STAGE2,
    imgsz=cfg.IMGSZ_STAGE2,
    batch=cfg.BATCH_STAGE2,
    device=device,
    patience=15,
    project=cfg.PROJECT_DIR,
    name=cfg.NAME_STAGE2,
    optimizer="AdamW",
    lr0=0.0003,
    lrf=0.01,
    cos_lr=True,
    mosaic=0.2,
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=10,
    dropout=0.0,
    freeze=0,
    multi_scale=False,
    cache=cfg.CACHE,
    workers=cfg.WORKERS,
    plots=True,
    amp=True,
)

# ============================================================
# VALIDATION FINALE
# ============================================================
print("\n=== VALIDATION FINALE ===")

best_final = f"{cfg.PROJECT_DIR}/{cfg.NAME_STAGE2}/weights/best.pt"
final_model = YOLO(best_final)

metrics = final_model.val(
    data=data_yaml_path,
    imgsz=cfg.IMGSZ_STAGE2,
    conf=0.25,
    iou=0.6,
    augment=True,
    plots=True,
)

print(f"\nmAP50    : {metrics.box.map50:.4f}")
print(f"mAP50-95 : {metrics.box.map:.4f}")

print("\nmAP50 par classe :")
for i, name in enumerate(data_cfg["names"]):
    try:
        print(f"  {name}: {metrics.box.maps[i]:.4f}")
    except IndexError:
        pass

if metrics.box.map50 > 0.92:
    print("\n✅ OBJECTIF ATTEINT : mAP50 > 0.92 !")
else:
    print(f"\n⚠️ mAP50 = {metrics.box.map50:.4f} (objectif 0.92)")

# ============================================================
# COPIE DU MODÈLE FINAL POUR LE PACKAGE DE DÉPLOIEMENT
# ============================================================
Path(cfg.FINAL_MODEL_DEST).parent.mkdir(parents=True, exist_ok=True)
shutil.copy(best_final, cfg.FINAL_MODEL_DEST)
print(f"\n✅ Modèle copié dans : {cfg.FINAL_MODEL_DEST}")

# Export ONNX (optionnel, pour déploiement)
final_model.export(format="onnx", imgsz=cfg.IMGSZ_STAGE2)