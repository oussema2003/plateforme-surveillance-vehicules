"""
test_plate_model.py
-------------------
Script de DIAGNOSTIC du modele de plaque (plate_yolo.pt) SEUL, sans le
reste du pipeline.

But : savoir si le modele detecte des plaques sur VOS images/videos, et a
quelle confiance. C'est l'etape indispensable avant de deboguer le pipeline
complet : si le modele ne detecte rien ici, il ne detectera rien non plus
dans vehicle_pipeline.py (d'ou les "plate=None" partout).

Deux modes de test :

  1) Mode "image entiere" (par defaut) :
     Le modele est applique directement sur chaque frame/image complete.
     Utile pour verifier que le modele fonctionne DU TOUT sur vos donnees.

  2) Mode "crops vehicules" (--on-crops) :
     On detecte d'abord les vehicules (avec le modele vehicule), puis on
     applique le modele plaque sur CHAQUE crop de vehicule. C'est EXACTEMENT
     ce que fait le pipeline en conditions reelles. Utile pour verifier que
     le modele marche sur des crops (parfois un modele entraine sur des
     images entieres rate les crops, et inversement).

Le script :
  - utilise un seuil de confiance TRES BAS par defaut (0.05) pour voir TOUTES
    les detections, meme faibles -> si le modele "voit" quelque chose, vous
    le verrez, meme avec une confiance mediocre.
  - dessine les boites detectees (avec la confiance) sur les frames.
  - sauvegarde dans un dossier `debug_plate/` les frames ou au moins une
    plaque a ete detectee (ou toutes les frames avec --save-all).
  - affiche un resume : nombre total de detections, confiance min/max/moy.

Utilisation :
    # test sur image entiere (le plus simple pour commencer)
    python test_plate_model.py --source video.mp4

    # test en conditions reelles (sur crops de vehicules, comme le pipeline)
    python test_plate_model.py --source video.mp4 --on-crops

    # test sur une image unique
    python test_plate_model.py --source image.jpg

    # forcer un seuil plus haut pour voir ce qui passerait dans le pipeline
    python test_plate_model.py --source video.mp4 --conf 0.45
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VEHICLE_CLASSES = [2, 3, 5, 7]  # voiture, moto, bus, camion


def draw_boxes(frame: np.ndarray, boxes_xyxy, confs, prefix: str = "") -> np.ndarray:
    out = frame.copy()
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, confs):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{prefix}{c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 255), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def predict_plate(model: YOLO, image: np.ndarray, conf: float, device: str):
    """Renvoie (xyxy numpy, confs numpy)."""
    res = model.predict(image, conf=conf, device=device, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()


def process_image_full(model: YOLO, frame: np.ndarray, conf: float, device: str):
    """Mode image entiere."""
    xyxy, confs = predict_plate(model, frame, conf, device)
    annotated = draw_boxes(frame, xyxy, confs, prefix="plaque ")
    return annotated, len(xyxy), confs


def process_image_crops(vehicle_model: YOLO, plate_model: YOLO,
                         frame: np.ndarray, conf: float, device: str):
    """Mode crops vehicules (comportement reel du pipeline)."""
    vres = vehicle_model.predict(
        frame, conf=0.25, device=device, classes=VEHICLE_CLASSES, verbose=False
    )[0]
    annotated = frame.copy()
    total_plates = 0
    all_confs = []

    if vres.boxes is not None and len(vres.boxes) > 0:
        h, w = frame.shape[:2]
        for (x1, y1, x2, y2) in vres.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            # dessine le crop vehicule en vert pour reperer
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
            crop = frame[y1:y2, x1:x2]
            pxyxy, pconfs = predict_plate(plate_model, crop, conf, device)
            total_plates += len(pxyxy)
            if len(pconfs) > 0:
                all_confs.extend(pconfs.tolist())
            # les coords plaque sont relatives au crop -> on les ramene au frame
            for (px1, py1, px2, py2), c in zip(pxyxy, pconfs):
                gx1, gy1 = int(px1 + x1), int(py1 + y1)
                gx2, gy2 = int(px2 + x1), int(py2 + y1)
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                label = f"plaque {c:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (gx1, gy1 - th - 8), (gx1 + tw + 4, gy1), (0, 0, 255), -1)
                cv2.putText(annotated, label, (gx1 + 2, gy1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated, total_plates, np.array(all_confs)


def main():
    p = argparse.ArgumentParser(description="Test isole du modele de plaque")
    p.add_argument("--source", required=True, help="image ou video a tester")
    p.add_argument("--plate-model", default="models/plate_yolo.pt")
    p.add_argument("--vehicle-model", default="models/yolo26n.pt",
                    help="utilise uniquement avec --on-crops")
    p.add_argument("--conf", type=float, default=0.05,
                    help="seuil de confiance (tres bas par defaut pour TOUT voir)")
    p.add_argument("--device", default=None)
    p.add_argument("--on-crops", action="store_true",
                    help="teste sur crops de vehicules (comportement reel du pipeline)")
    p.add_argument("--out-dir", default="debug_plate",
                    help="dossier ou sauver les frames annotees")
    p.add_argument("--save-all", action="store_true",
                    help="sauve TOUTES les frames, pas seulement celles avec detection")
    p.add_argument("--max-frames", type=int, default=0,
                    help="limite le nombre de frames traitees (0 = pas de limite)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # resolution device
    device = args.device
    if device is None:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"[test_plate] device = {device}")
    print(f"[test_plate] modele plaque = {args.plate_model}")
    print(f"[test_plate] seuil conf = {args.conf}")
    print(f"[test_plate] mode = {'crops vehicules' if args.on_crops else 'image entiere'}")
    print(f"[test_plate] sorties dans = {os.path.abspath(args.out_dir)}")

    plate_model = YOLO(args.plate_model)
    vehicle_model = YOLO(args.vehicle_model) if args.on_crops else None

    _, ext = os.path.splitext(args.source)
    is_image = ext.lower() in IMAGE_EXTS

    total_detections = 0
    all_confs = []
    frames_with_det = 0
    frames_processed = 0

    if is_image:
        frame = cv2.imread(args.source)
        if frame is None:
            raise RuntimeError(f"Image illisible : {args.source}")
        if args.on_crops:
            annotated, n, confs = process_image_crops(vehicle_model, plate_model, frame, args.conf, device)
        else:
            annotated, n, confs = process_image_full(plate_model, frame, args.conf, device)
        total_detections += n
        if len(confs) > 0:
            all_confs.extend(confs.tolist())
            frames_with_det += 1
        out_path = os.path.join(args.out_dir, "image_annotated.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"[test_plate] image : {n} plaque(s) detectee(s) -> {out_path}")
        frames_processed = 1
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir : {args.source}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[test_plate] video : {total} frames")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if args.max_frames > 0 and idx > args.max_frames:
                break
            if args.on_crops:
                annotated, n, confs = process_image_crops(vehicle_model, plate_model, frame, args.conf, device)
            else:
                annotated, n, confs = process_image_full(plate_model, frame, args.conf, device)

            frames_processed += 1
            total_detections += n
            if len(confs) > 0:
                all_confs.extend(confs.tolist())
                frames_with_det += 1

            if n > 0 or args.save_all:
                out_path = os.path.join(args.out_dir, f"frame_{idx:05d}.jpg")
                cv2.imwrite(out_path, annotated)

            if idx % 30 == 0:
                print(f"[test_plate] frame {idx}/{total} - detections cumul : {total_detections}")

            # affichage optionnel
            cv2.imshow("test_plate", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    # resume
    print("\n========== RESUME ==========")
    print(f"Frames traitees        : {frames_processed}")
    print(f"Frames avec >=1 plaque : {frames_with_det}")
    print(f"Total detections       : {total_detections}")
    if all_confs:
        print(f"Confiance min          : {min(all_confs):.3f}")
        print(f"Confiance max          : {max(all_confs):.3f}")
        print(f"Confiance moyenne      : {np.mean(all_confs):.3f}")
    else:
        print("Confiance              : AUCUNE DETECTION (le modele ne voit rien)")
    print(f"Frames annotees dans   : {os.path.abspath(args.out_dir)}")
    print("============================\n")

    if total_detections == 0:
        print("INTERPRETATION : le modele plate_yolo.pt ne detecte AUCUNE plaque sur vos")
        print("donnees, meme avec un seuil tres bas (conf=0.05). Le probleme vient du")
        print("modele lui-meme (pas entraine sur ce type de plaques / angles / qualite),")
        print("et non du pipeline. Solutions : reentrainer le modele sur vos plaques, ou")
        print("utiliser un modele de plaque generique adapte (ex. modele licence-plate")
        print("pre-entraine sur plaques de votre pays).")
    elif max(all_confs) < 0.45:
        print("INTERPRETATION : le modele detecte des plaques mais avec une confiance")
        print("toujours < 0.45. Dans le pipeline (seuil 0.4-0.5) elles seraient rejetees.")
        print("Solution : baisser --plate-min-conf dans le pipeline, ou ameliorer le modele.")
    else:
        print("INTERPRETATION : le modele detecte bien des plaques avec une bonne")
        print("confiance. Le probleme du pipeline vide vient donc d'ailleurs (ex. crops")
        print("trop petits, ou vehicules non detectes). Relancez avec --on-crops pour")
        print("verifier le comportement en conditions reelles.")


if __name__ == "__main__":
    main()