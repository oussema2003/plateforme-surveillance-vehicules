"""Pipeline de d�tection d'un v�hicule dans une image ou une vid�o.

Ce module d�tecte les plaques, lit le texte de la plaque, identifie la marque
et la couleur du v�hicule, puis enregistre le r�sultat dans un fichier JSON.

Améliorations:
- Consensus OCR : Accumule les lectures sur plusieurs frames et choisit le résultat le plus fréquent
- Détection couleur améliorée pour le noir
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

from color_classifier import ColorClassifier

PROJECT_ROOT = Path(__file__).resolve().parent


def normalize_plate_images_dir(plate_images_dir: str | Path) -> Path:
    """Chemin absolu du dossier d'images de plaques."""
    path = Path(plate_images_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def safe_plate_prefix(matricule: str) -> str:
    return matricule.replace(" ", "_").replace("/", "-")


def resolve_plate_image_file(
    stored_path: str | None,
    matricule: str | None,
    plate_images_dir: Path,
) -> Path | None:
    """Retourne le fichier image existant (chemin enregistré ou recherche par matricule)."""
    if stored_path:
        candidate = Path(stored_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    if matricule:
        prefix = safe_plate_prefix(matricule)
        matches = sorted(plate_images_dir.glob(f"{prefix}_*.jpg"), reverse=True)
        for match in matches:
            if match.is_file() and match.stat().st_size > 0:
                return match
    return None


def plate_image_storage_path(image_file: Path) -> str:
    """Chemin relatif stable pour le JSON."""
    return f"data/plate_images/{image_file.name}"


class VehicleJSONStore:
    """Stocke un dictionnaire de v�hicules par matricule dans un fichier JSON."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    def _read(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: dict) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    @staticmethod
    def _plate_similarity(plate1: str, plate2: str) -> float:
        """Calcule la similarité entre deux plaques."""
        if not plate1 or not plate2:
            return 0.0
        
        if plate1 == plate2:
            return 1.0
        
        # Extraire seulement les chiffres
        digits1 = ''.join(c for c in plate1 if c.isdigit())
        digits2 = ''.join(c for c in plate2 if c.isdigit())
        
        if not digits1 or not digits2:
            return 0.0
        
        if abs(len(digits1) - len(digits2)) > 1:
            return 0.0
        
        # Calculer le nombre de chiffres en commun à la même position
        min_len = min(len(digits1), len(digits2))
        matches = sum(1 for i in range(min_len) if digits1[i] == digits2[i])
        
        similarity = matches / max(len(digits1), len(digits2))
        
        return similarity

    def update(self, detections: List[dict]) -> dict:
        """Mets à jour les enregistrements existants avec les nouvelles détections."""
        store = self._read()

        for detection in detections:
            matricule = detection["matricule"]
            if not matricule:
                continue

            current_time = datetime.fromisoformat(detection["derniere_detection"])
            
            # Chercher une plaque identique ou similaire dans le store
            found_key = None
            best_similarity = 0.0
            
            for existing_key in store.keys():
                if existing_key == matricule:
                    found_key = existing_key
                    best_similarity = 1.0
                    break
                
                # Calculer la similarité
                similarity = self._plate_similarity(matricule, existing_key)
                if similarity > best_similarity and similarity >= 0.60:  # Seuil abaissé à 60%
                    best_similarity = similarity
                    found_key = existing_key
            
            if found_key:
                existing = store[found_key]
                previous_time = datetime.fromisoformat(existing["derniere_detection"])
                
                # Si déjà vu dans la même heure, ignorer
                if (
                    previous_time.date() == current_time.date()
                    and previous_time.hour == current_time.hour
                ):
                    continue

                # Mettre à jour avec la nouvelle détection
                # Si la nouvelle matricule a une meilleure confiance, l'utiliser
                new_conf = detection.get("confidence_plate", 0)
                old_conf = existing.get("confidence_plate", 0)
                
                if new_conf > old_conf and matricule != found_key:
                    # Remplacer la clé (meilleure lecture)
                    del store[found_key]
                    store[matricule] = {
                        "matricule": matricule,
                        "marque": detection["marque"],
                        "couleur": detection["couleur"],
                        "date": detection["date"],
                        "heure": detection["heure"],
                        "derniere_detection": detection["derniere_detection"],
                        "premiere_detection": existing.get("premiere_detection", detection["derniere_detection"]),
                        "confidence_plate": new_conf,
                        "plate_image": detection.get("plate_image", existing.get("plate_image")),
                    }
                else:
                    # Garder la clé existante, juste mettre à jour la date
                    store[found_key] = {
                        "matricule": found_key,
                        "marque": detection["marque"],
                        "couleur": detection["couleur"],
                        "date": detection["date"],
                        "heure": detection["heure"],
                        "derniere_detection": detection["derniere_detection"],
                        "premiere_detection": existing.get("premiere_detection", detection["derniere_detection"]),
                        "confidence_plate": max(new_conf, old_conf),
                        "plate_image": existing.get("plate_image") or detection.get("plate_image"),
                    }
            else:
                # Nouvelle plaque
                store[matricule] = {
                    "matricule": matricule,
                    "marque": detection["marque"],
                    "couleur": detection["couleur"],
                    "date": detection["date"],
                    "heure": detection["heure"],
                    "derniere_detection": detection["derniere_detection"],
                    "premiere_detection": detection["derniere_detection"],
                    "confidence_plate": detection.get("confidence_plate", 0),
                    "plate_image": detection.get("plate_image"),
                }

        self._write(store)
        return store

    def read(self) -> dict:
        return self._read()


class VehiclePipeline:
    """Pipeline de d�tection de plaque, marque et couleur."""

    def __init__(
        self,
        vehicle_model_path: str = "models/yolo26n.pt",
        plate_model_path: str = "models/best.pt",
        brand_model_path: str = "models/brand_yolo.pt",
        output_path: str = "models/vehicules.json",
        device: Optional[str] = None,
        ocr_consensus_frames: int = 10,  # Nombre de frames pour le consensus OCR
        save_plate_images: bool = True,  # Sauvegarder les images de plaques
        plate_images_dir: str = "data/plate_images",  # Dossier pour les images
    ):
        self.device = device or self._default_device()
        print(f"Device choisi pour les modèles: {self.device}")
 
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.brand_model = YOLO(brand_model_path)

        self._set_model_device(self.vehicle_model)
        self._set_model_device(self.plate_model)
        self._set_model_device(self.brand_model)

        self.color_classifier = ColorClassifier()
        self.store = VehicleJSONStore(output_path)
        self.ocr_readers = self._load_ocr_readers()
        
        # Consensus OCR : accumule les lectures par véhicule détecté
        self.ocr_consensus_frames = ocr_consensus_frames
        self.vehicle_ocr_buffer = defaultdict(list)  # {vehicle_id: [(plate_text, confidence), ...]}
        self.vehicle_brand_buffer = defaultdict(list)  # {vehicle_id: [(brand, confidence), ...]}
        self.vehicle_color_buffer = defaultdict(list)  # {vehicle_id: [color_name, ...]}
        self.vehicle_last_seen = {}  # {vehicle_id: frame_number}
        
        # Sauvegarde des images de plaques
        self.save_plate_images = save_plate_images
        self.plate_images_dir = normalize_plate_images_dir(plate_images_dir)
        if self.save_plate_images:
            self.plate_images_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Images de plaques seront sauvegardées dans: {self.plate_images_dir}")

        # Dernière frame annotée traitée (mise à jour à chaque frame dans
        # process_video). Permet à une interface externe (dashboard web,
        # camera_manager.py) de récupérer une image "live" sans avoir à
        # relire le flux elle-même.
        self.last_annotated_frame: Optional[np.ndarray] = None

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
 
    def _set_model_device(self, model: Any) -> None:
        """Set device for YOLO model - use .to() method"""
        try:
            if hasattr(model, "to"):
                model.to(self.device)
            # Also try setting device in model config if available
            if hasattr(model, "device"):
                model.device = self.device
        except Exception:
            pass

    @staticmethod
    def _arabic_to_latin_digits(text: str) -> str:
        """Convert Arabic-Indic numerals to Latin digits"""
        if not text:
            return text
        
        # Arabic-Indic digits mapping
        arabic_to_latin = {
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        }
        
        for arabic, latin in arabic_to_latin.items():
            text = text.replace(arabic, latin)
        
        return text

    @staticmethod
    def _correct_ocr_errors(text: str) -> str:
        """Corrige les erreurs courantes d'OCR sur les plaques tunisiennes.
        Les plaques tunisiennes ont le format: XXX XXXX ou XX XXXX (chiffres uniquement)."""
        if not text:
            return text
        
        # Corrections caractère par caractère (confusions fréquentes)
        corrections = {
            'O': '0', 'o': '0', 'D': '0', 'Q': '0',
            'I': '1', 'l': '1', '|': '1', 'i': '1',
            'Z': '2', 'z': '2',
            'S': '5', 's': '5',
            'G': '6', 'b': '6',
            'T': '7', 't': '7',
            'B': '8',
            'g': '9', 'q': '9',
        }
        
        corrected = text
        for old, new in corrections.items():
            corrected = corrected.replace(old, new)
        
        return corrected

    @staticmethod
    def _validate_tunisian_plate(text: str) -> bool:
        """Valide qu'une plaque correspond au format tunisien."""
        if not text:
            return False
        
        # Extraire les chiffres
        digits = ''.join(c for c in text if c.isdigit())
        
        # Format tunisien: 5 à 7 chiffres
        if len(digits) < 5 or len(digits) > 7:
            return False
        
        # Vérifier qu'il n'y a pas trop de chiffres identiques consécutifs (bruit OCR)
        for i in range(len(digits) - 3):
            if len(set(digits[i:i+4])) == 1:  # 4 chiffres identiques consécutifs
                return False
        
        return True

    @staticmethod
    def _normalize_plate_text(text: str) -> str:
        """
        Normalize Tunisian license plate text.
        Tunisian plates: mot arabe (تونس) + NUMÉROS FRANÇAIS (ex: 208 7725)
        Format: XXX XXXX (3 + 4 chiffres) OU XX XXXX (2 + 4 chiffres)
        """
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        # Convertir les chiffres arabes en chiffres latins
        text = VehiclePipeline._arabic_to_latin_digits(text)
        
        # Appliquer les corrections OCR
        text = VehiclePipeline._correct_ocr_errors(text)
        
        # Extraire UNIQUEMENT les chiffres latins (0-9)
        digits = re.findall(r'[0-9]', text)
        
        if not digits:
            return None
        
        # Joindre tous les chiffres trouvés
        plate_numbers = ''.join(digits)
        
        # Format Tunisien standard: XXX XXXX (3 + 4 chiffres) ou XX XXXX (2 + 4 chiffres)
        if len(plate_numbers) >= 7:
            # Prendre exactement 7 chiffres : XXX XXXX
            result = plate_numbers[:3] + ' ' + plate_numbers[3:7]
        elif len(plate_numbers) == 6:
            # Pour 6 chiffres: XX XXXX
            result = plate_numbers[:2] + ' ' + plate_numbers[2:6]
        elif len(plate_numbers) == 5:
            # Pour 5 chiffres: XX XXX
            result = plate_numbers[:2] + ' ' + plate_numbers[2:]
        else:
            # Pour moins de 5 chiffres, retourner tel quel si au moins 2 chiffres
            result = plate_numbers if len(plate_numbers) >= 2 else None
        
        # Valider le résultat
        if result and VehiclePipeline._validate_tunisian_plate(result):
            return result
        
        return None



    @staticmethod
    def _preprocess_plate_crop(crop: np.ndarray) -> list:
        """Retourne plusieurs prétraitements optimisés de la plaque pour améliorer l'OCR."""
        preprocessed = []
        
        # Redimensionner si trop petit
        h, w = crop.shape[:2]
        if h < 50:
            scale = 50 / h
            crop = cv2.resize(crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # 1. Version originale
        preprocessed.append(crop)
        
        # 2. Niveaux de gris avec redimensionnement 2x
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        # 3. CLAHE + débruitage (meilleure qualité)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 7, 7, 21)
        preprocessed.append(denoised)
        
        # 4. Binarisation Otsu
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(binary_otsu)
        
        return preprocessed

    def _load_ocr_readers(self) -> list:
        readers = []
        try:
            import easyocr
            use_gpu = False

            # EasyOCR ne supporte pas directement 'fr' et 'ar' ensemble.
            # Nous essayons des combinaisons séparées pour couvrir les plaques tunisiennes.
            for langs in (["en", "fr"], ["ar", "en"]):
                try:
                    readers.append(easyocr.Reader(langs, gpu=use_gpu))
                except Exception:
                    pass
        except Exception:
            pass
        return readers

    def _extract_plate_text(self, crop: np.ndarray) -> Optional[str]:
        """Extrait le texte de la plaque avec plusieurs tentatives de prétraitement optimisées."""
        if crop is None or crop.size == 0:
            return None

        all_candidates: list[tuple[str, int]] = []  # (texte, score de confiance)
        
        # Obtenir les versions prétraitées (réduit à 3 versions pour la vitesse)
        preprocessed_versions = self._preprocess_plate_crop(crop)

        # Essayer uniquement avec le premier lecteur (le plus rapide)
        if self.ocr_readers:
            reader = self.ocr_readers[0]  # Lecteur ["en", "fr"]
            
            # Essayer sur chaque version prétraitée
            for i, preprocessed in enumerate(preprocessed_versions):
                try:
                    results = reader.readtext(preprocessed, detail=1)
                    
                    if results:
                        # Trier par position X (gauche à droite)
                        results = sorted(results, key=lambda item: item[0][0][0])
                        
                        for result in results:
                            text = result[1]
                            confidence = result[2]
                            
                            if text and text.strip():
                                # Normaliser le texte
                                normalized = self._normalize_plate_text(text)
                                if normalized and re.search(r"\d", normalized):
                                    # Les versions traitées (CLAHE+débruitage, Otsu) sont privilégiées
                                    version_bonus = 1.2 if i > 0 else 1.0
                                    score = int(confidence * 100 * version_bonus)
                                    all_candidates.append((normalized, score))
                except Exception:
                    continue

        if not all_candidates:
            return None

        # Grouper les candidats similaires
        candidate_groups = {}
        for text, score in all_candidates:
            # Chercher un groupe similaire existant
            found_group = False
            for key in candidate_groups.keys():
                # Si les textes sont identiques ou très similaires
                if text == key:
                    candidate_groups[key].append((text, score))
                    found_group = True
                    break
                # Ou si seulement 1-2 caractères différents et même longueur
                elif len(text) == len(key) and sum(c1 != c2 for c1, c2 in zip(text, key)) <= 2:
                    candidate_groups[key].append((text, score))
                    found_group = True
                    break
            
            if not found_group:
                candidate_groups[text] = [(text, score)]
        
        # Choisir le groupe avec le meilleur score total
        best_group = None
        best_total_score = 0
        
        for key, candidates in candidate_groups.items():
            total_score = sum(s for _, s in candidates)
            if total_score > best_total_score:
                best_total_score = total_score
                best_group = candidates
        
        if best_group:
            # Dans le meilleur groupe, choisir le texte le plus fréquent
            from collections import Counter
            texts = [t for t, _ in best_group]
            most_common = Counter(texts).most_common(1)[0][0]
            return most_common
        
        return None

    def _detect_vehicles(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.25,
        only_largest: bool = True,
    ) -> list[tuple[int, int, int, int]]:
        results = self.vehicle_model.predict(
            frame,
            conf=confidence_threshold,
            device=self.device,
            verbose=False,
            classes=[2, 3, 5, 7],
        )

        if not results or len(results) == 0:
            return []
  
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return []
  
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
        
        # Si on veut seulement la voiture la plus proche (la plus grande)
        if only_largest and boxes:
            # Calculer la surface de chaque boîte
            boxes_with_area = [(box, (box[2]-box[0]) * (box[3]-box[1])) for box in boxes]
            # Garder seulement la plus grande
            largest = max(boxes_with_area, key=lambda x: x[1])[0]
            return [largest]
        
        return boxes

    def _save_plate_image(self, plate_crop: np.ndarray, plate_text: str, timestamp: str = None) -> Optional[str]:
        """Sauvegarde l'image de la plaque dans le dossier dédié.
        Retourne le chemin du fichier sauvegardé."""
        if not self.save_plate_images or plate_crop is None or plate_crop.size == 0:
            return None
        
        if not plate_text or plate_text == "INCONNU":
            return None
        
        try:
            # Nom du fichier: plaque_horodatage.jpg
            # Remplacer les espaces et caractères spéciaux
            safe_plate = plate_text.replace(' ', '_').replace('/', '-')
            
            if timestamp:
                filename = f"{safe_plate}_{timestamp}.jpg"
            else:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{safe_plate}_{timestamp}.jpg"
            
            filepath = self.plate_images_dir / filename
            self.plate_images_dir.mkdir(parents=True, exist_ok=True)

            if not cv2.imwrite(str(filepath), plate_crop):
                print(f"⚠ Échec sauvegarde image plaque (cv2.imwrite): {filepath}")
                return None
            if not filepath.is_file() or filepath.stat().st_size == 0:
                print(f"⚠ Fichier plaque vide ou absent après sauvegarde: {filepath}")
                return None

            return plate_image_storage_path(filepath)
        except Exception as e:
            print(f"⚠ Erreur sauvegarde image plaque: {e}")
            return None

    def _detect_plate_in_crop(
        self,
        crop: np.ndarray,
        confidence_threshold: float,
    ) -> tuple[Optional[str], float, Optional[np.ndarray]]:
        """Détecte la plaque dans un crop de véhicule.
        Retourne: (texte_plaque, confiance, crop_plaque)"""
        if crop is None or crop.size == 0:
            return None, 0.0, None

        results = self.plate_model.predict(
            crop,
            conf=confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not results or len(results) == 0:
            return None, 0.0, None
 
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, 0.0, None
 
        best_box = max(boxes, key=lambda box: float(box.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        plate_crop = crop[y1:y2, x1:x2]
        if plate_crop.size == 0:
            plate_crop = crop

        plate_text = self._extract_plate_text(plate_crop)
        confidence = float(best_box.conf[0])
        
        return plate_text, confidence, plate_crop

    def _expand_box(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        scale: float = 2.0,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        half_width = (x2 - x1) * scale / 2
        half_height = (y2 - y1) * scale / 2
        x1_new = max(0, int(cx - half_width))
        y1_new = max(0, int(cy - half_height))
        x2_new = min(w, int(cx + half_width))
        y2_new = min(h, int(cy + half_height))
        return frame[y1_new:y2_new, x1_new:x2_new], (x1_new, y1_new, x2_new, y2_new)

    def _detect_brand(
        self,
        crop: np.ndarray,
        confidence_threshold: float,
    ) -> Tuple[Optional[str], float]:
        if crop is None or crop.size == 0:
            return None, 0.0

        def detect(conf: float) -> Tuple[Optional[str], float]:
            results = self.brand_model.predict(
                crop,
                conf=conf,
                device=self.device,
                verbose=False,
            )
            if not results or len(results) == 0:
                return None, 0.0

            labels = results[0].boxes
            if labels is None or len(labels) == 0:
                return None, 0.0

            best_box = max(labels, key=lambda box: float(box.conf[0]))
            class_id = int(best_box.cls[0])
            conf_score = float(best_box.conf[0])
            class_name = results[0].names.get(class_id, f"Classe_{class_id}")
            return class_name, conf_score

        brand, conf = detect(confidence_threshold)
        if brand is None and confidence_threshold > 0.25:
            brand, conf = detect(max(confidence_threshold * 0.5, 0.25))

        return brand, conf

    def _build_detection_record(
        self,
        matricule: str,
        marque: str,
        couleur: str,
        source: Optional[str] = None,
    ) -> dict:
        now = datetime.now()
        return {
            "matricule": matricule,
            "marque": marque,
            "couleur": couleur,
            "date": now.date().isoformat(),
            "heure": now.time().replace(microsecond=0).isoformat(),
            "derniere_detection": now.isoformat(),
            "premiere_detection": now.isoformat(),
            "source": source,
        }

    def process_single_image(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        source: Optional[str] = None,
        persist: bool = True,
    ) -> Tuple[List[dict], np.ndarray]:
        annotated = frame.copy()
        detections: List[dict] = []

        vehicle_boxes = self._detect_vehicles(frame, confidence_threshold=0.25)
        fallback_to_full_frame = False

        if not vehicle_boxes:
            fallback_to_full_frame = True
            vehicle_boxes = [(0, 0, frame.shape[1], frame.shape[0])]

        for x1, y1, x2, y2 in vehicle_boxes:
            # Original crop for color detection (no expansion - we want just the vehicle)
            original_crop = frame[y1:y2, x1:x2]
            
            # Agrandir légèrement la boîte du véhicule pour mieux capturer la plaque et marque
            # (Mais pas la couleur - celle-ci doit être sur le crop original pour éviter le fond)
            expanded_crop, (exp_x1, exp_y1, exp_x2, exp_y2) = self._expand_box(
                frame, x1, y1, x2, y2, scale=1.15
            )
            if expanded_crop.size == 0:
                expanded_crop = frame[y1:y2, x1:x2]
                exp_x1, exp_y1, exp_x2, exp_y2 = x1, y1, x2, y2

            if expanded_crop.size == 0 or original_crop.size == 0:
                continue

            plate_text, score_plate, plate_crop = self._detect_plate_in_crop(
                expanded_crop,
                max(confidence_threshold * 0.8, 0.3),
            )
            marque, score_marque = self._detect_brand(expanded_crop, confidence_threshold)
            # Utiliser le crop original (sans expansion) pour la couleur
            color_result = self.color_classifier.get_dominant_color(original_crop)

            if marque is None and not fallback_to_full_frame:
                marque, score_marque = self._detect_brand(frame, max(confidence_threshold * 0.25, 0.2))

            if plate_text is None and not fallback_to_full_frame:
                plate_text, score_plate, plate_crop = self._detect_plate_in_crop(
                    frame,
                    max(confidence_threshold * 0.25, 0.2),
                )

            if plate_text is None:
                if score_plate > 0.0:
                    plate_text = "INCONNU"
                elif fallback_to_full_frame:
                    plate_text = "INCONNU"
                else:
                    continue
            
            # Sauvegarder l'image de la plaque si détectée
            plate_image_path = None
            if plate_crop is not None and plate_text and plate_text != "INCONNU":
                plate_image_path = self._save_plate_image(plate_crop, plate_text)

            detection = self._build_detection_record(
                matricule=plate_text,
                marque=marque or "INCONNUE",
                couleur=color_result.name,
                source=source,
            )
            detection["confidence_plate"] = round(score_plate, 3)
            detection["confidence_marque"] = round(score_marque, 3)
            detection["confidence_couleur"] = round(float(color_result.confidence), 3)
            
            if plate_image_path:
                detection["plate_image"] = plate_image_path

            detections.append(detection)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{plate_text} {detection['marque']} {detection['couleur']}"
            cv2.putText(
                annotated,
                text,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if persist and detections:
            self.store.update(detections)

        return detections, annotated
    @staticmethod
    def _plate_similarity(plate1: str, plate2: str) -> float:
        """Calcule la similarité entre deux plaques (0.0 à 1.0).
        Utilise un algorithme plus intelligent pour détecter les plaques similaires."""
        if not plate1 or not plate2:
            return 0.0
        
        if plate1 == plate2:
            return 1.0
        
        # Extraire seulement les chiffres
        digits1 = ''.join(c for c in plate1 if c.isdigit())
        digits2 = ''.join(c for c in plate2 if c.isdigit())
        
        if not digits1 or not digits2:
            return 0.0
        
        # Si longueurs très différentes (>2), peu probable que ce soit la même
        if abs(len(digits1) - len(digits2)) > 2:
            return 0.0
        
        # Calculer plusieurs métriques
        
        # 1. Nombre de chiffres en commun à la même position (strict)
        min_len = min(len(digits1), len(digits2))
        max_len = max(len(digits1), len(digits2))
        exact_matches = sum(1 for i in range(min_len) if digits1[i] == digits2[i])
        
        # 2. Nombre de chiffres présents dans les deux (ordre flexible)
        common_digits = sum(min(digits1.count(d), digits2.count(d)) for d in set(digits1))
        
        # 3. Distance d'édition (Levenshtein simplifiée)
        # Pour les plaques, chaque insertion/suppression/substitution coûte 1
        if len(digits1) == len(digits2):
            edit_distance = sum(c1 != c2 for c1, c2 in zip(digits1, digits2))
        else:
            # Si longueurs différentes, comparer les sous-chaînes possibles
            edit_distance = max_len
            for i in range(max_len - min_len + 1):
                if len(digits1) < len(digits2):
                    substring = digits2[i:i+len(digits1)]
                    dist = sum(c1 != c2 for c1, c2 in zip(digits1, substring))
                else:
                    substring = digits1[i:i+len(digits2)]
                    dist = sum(c1 != c2 for c1, c2 in zip(digits2, substring))
                edit_distance = min(edit_distance, dist + abs(len(digits1) - len(digits2)))
        
        # Calculer les scores
        position_score = exact_matches / max_len
        content_score = common_digits / max_len
        edit_score = 1.0 - (edit_distance / max_len)
        
        # Score final pondéré (position stricte compte plus)
        similarity = (position_score * 0.5 + content_score * 0.2 + edit_score * 0.3)
        
        return similarity

    def _get_consensus_result(self, values: List[str], min_consensus: int = 3) -> Optional[str]:
        """Obtient le résultat le plus fréquent si suffisamment de consensus.
        Si pas assez de consensus, retourne le plus fréquent quand même."""
        if not values:
            return None
        
        # Compter les occurrences
        counter = Counter(values)
        most_common = counter.most_common(1)[0]
        
        # Si on a au moins min_consensus occurrences, retourner
        if most_common[1] >= min_consensus:
            return most_common[0]
        
        # Sinon, si on a au moins 2 lectures, retourner la plus fréquente quand même
        if len(values) >= 2:
            return most_common[0]
        
        # Sinon, retourner la seule valeur disponible
        return values[0] if len(values) == 1 else None

    def _generate_vehicle_id(self, x1: int, y1: int, x2: int, y2: int) -> str:
        """Génère un ID basé sur la position du véhicule dans l'image."""
        # Utiliser le centre de la boîte, arrondi à 50 pixels près
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # Zone de 100x100 pixels pour regrouper les détections du même véhicule
        zone_x = cx // 100
        zone_y = cy // 100
        return f"v_{zone_x}_{zone_y}"

    def process_video(
        self,
        source: str,
        frame_skip: int = 10,
        confidence_threshold: float = 0.5,
        no_display: bool = False,
        stop_event: Optional["threading.Event"] = None,
    ) -> List[dict]:
        """
        stop_event : si fourni (threading.Event), la boucle s'arrête
        proprement dès que .set() est appelé dessus. Utilisé par
        camera_manager.py pour pouvoir stopper une caméra depuis le
        dashboard web sans tuer tout le processus.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source video : {source}")

        # Réinitialiser les buffers pour chaque nouvelle vidéo
        self.vehicle_ocr_buffer.clear()
        self.vehicle_brand_buffer.clear()
        self.vehicle_color_buffer.clear()
        self.vehicle_last_seen.clear()

        detections: List[dict] = []
        frame_number = 0
        processed_frames = 0
        
        # Stocker temporairement les détections en cours avec consensus
        temp_detections = {}  # {vehicle_id: detection_dict}
 
        while True:
            if stop_event is not None and stop_event.is_set():
                print("⏹ Arrêt demandé (stop_event) — fin du traitement de la caméra.")
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
 
            if frame_number % frame_skip != 0:
                continue
 
            processed_frames += 1
            if processed_frames % 10 == 0:
                print(f"Traitement: frame {frame_number} ({processed_frames} traitées)")

            # Nettoyer les véhicules qui ne sont plus visibles (pas vus depuis 20 frames)
            vehicles_to_remove = [
                vid for vid, last_frame in self.vehicle_last_seen.items()
                if frame_number - last_frame > 20
            ]
            for vid in vehicles_to_remove:
                # Finaliser le véhicule avec consensus avant de le supprimer
                if vid in temp_detections:
                    final_det = temp_detections[vid].copy()
                    
                    # Appliquer le consensus OCR
                    consensus_plate = self._get_consensus_result(
                        self.vehicle_ocr_buffer[vid], 
                        min_consensus=2  # Réduit à 2 pour être plus flexible
                    )
                    if consensus_plate:
                        final_det['matricule'] = consensus_plate
                    
                    # Appliquer le consensus marque
                    consensus_brand = self._get_consensus_result(
                        self.vehicle_brand_buffer[vid],
                        min_consensus=2
                    )
                    if consensus_brand:
                        final_det['marque'] = consensus_brand
                    
                    # Appliquer le consensus couleur
                    consensus_color = self._get_consensus_result(
                        self.vehicle_color_buffer[vid],
                        min_consensus=2  # Réduit à 2 pour être plus flexible
                    )
                    if consensus_color:
                        final_det['couleur'] = consensus_color
                    
                    # Toujours enregistrer, même si certains champs sont INCONNU
                    detections.append(final_det)
                    print(f"✓ Consensus véhicule {vid}: {final_det['matricule']} | {final_det['marque']} | {final_det['couleur']}")
                    del temp_detections[vid]
                
                del self.vehicle_last_seen[vid]
                del self.vehicle_ocr_buffer[vid]
                del self.vehicle_brand_buffer[vid]
                del self.vehicle_color_buffer[vid]

            image_source = f"{source}@{frame_number}"
            found, annotated = self.process_single_image(
                frame,
                confidence_threshold=confidence_threshold,
                source=image_source,
                persist=False,
            )

            # Mémorise la dernière frame annotée (lue par l'API pour l'aperçu live)
            self.last_annotated_frame = annotated

            # Accumuler les résultats pour le consensus
            for record in found:
                # Récupérer la position du véhicule pour générer un ID
                vehicle_boxes = self._detect_vehicles(frame, confidence_threshold=0.25)
                if vehicle_boxes:
                    x1, y1, x2, y2 = vehicle_boxes[0]  # On prend le premier (le plus grand)
                    vehicle_id = self._generate_vehicle_id(x1, y1, x2, y2)
                    
                    # Marquer comme vu sur cette frame
                    self.vehicle_last_seen[vehicle_id] = frame_number
                    
                    # Accumuler les détections
                    if record['matricule'] and record['matricule'] != "INCONNU":
                        self.vehicle_ocr_buffer[vehicle_id].append(record['matricule'])
                    
                    if record['marque'] and record['marque'] != "INCONNUE":
                        self.vehicle_brand_buffer[vehicle_id].append(record['marque'])
                    
                    if record['couleur']:
                        self.vehicle_color_buffer[vehicle_id].append(record['couleur'])
                    
                    # Garder une trace temporaire (sera finalisée avec consensus)
                    if vehicle_id not in temp_detections:
                        temp_detections[vehicle_id] = record
                    
                    # Limiter la taille des buffers pour éviter la mémoire excessive
                    if len(self.vehicle_ocr_buffer[vehicle_id]) > self.ocr_consensus_frames:
                        self.vehicle_ocr_buffer[vehicle_id] = self.vehicle_ocr_buffer[vehicle_id][-self.ocr_consensus_frames:]
                    if len(self.vehicle_brand_buffer[vehicle_id]) > self.ocr_consensus_frames:
                        self.vehicle_brand_buffer[vehicle_id] = self.vehicle_brand_buffer[vehicle_id][-self.ocr_consensus_frames:]
                    if len(self.vehicle_color_buffer[vehicle_id]) > self.ocr_consensus_frames:
                        self.vehicle_color_buffer[vehicle_id] = self.vehicle_color_buffer[vehicle_id][-self.ocr_consensus_frames:]
                    
                    # Afficher le progrès avec les buffers actuels
                    ocr_count = len(self.vehicle_ocr_buffer[vehicle_id])
                    brand_count = len(self.vehicle_brand_buffer[vehicle_id])
                    color_count = len(self.vehicle_color_buffer[vehicle_id])
                    print(f"Frame {frame_number:06d} [{vehicle_id}]: {record['matricule']} | {record['marque']} | {record['couleur']} (buffer: OCR={ocr_count}, brand={brand_count}, color={color_count})")

            if not no_display:
               try:
                   # Redimensionner pour afficher en dimensions réelles (ou adaptées à l'écran)
                   display_frame = annotated.copy()
                   h, w = display_frame.shape[:2]
                    
                   # Adapter la taille à l'écran (max 1280x720 pour éviter trop grand)
                   if w > 1280 or h > 720:
                       scale = min(1280 / w, 720 / h)
                       new_w = int(w * scale)
                       new_h = int(h * scale)
                       display_frame = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                   cv2.imshow("Vehicle detection", display_frame)
                   if cv2.waitKey(1) & 0xFF == ord("q"):
                       break
               except cv2.error as e:
                   # Si l'affichage échoue, continuer sans affichage
                   print(f"⚠ Affichage désactivé (erreur OpenCV): {str(e)[:100]}")
                   no_display = True

        # Finaliser tous les véhicules restants à la fin de la vidéo
        for vehicle_id in list(temp_detections.keys()):
            final_det = temp_detections[vehicle_id].copy()
            
            consensus_plate = self._get_consensus_result(
                self.vehicle_ocr_buffer[vehicle_id],
                min_consensus=2
            )
            if consensus_plate:
                final_det['matricule'] = consensus_plate
            
            consensus_brand = self._get_consensus_result(
                self.vehicle_brand_buffer[vehicle_id],
                min_consensus=2
            )
            if consensus_brand:
                final_det['marque'] = consensus_brand
            
            consensus_color = self._get_consensus_result(
                self.vehicle_color_buffer[vehicle_id],
                min_consensus=2
            )
            if consensus_color:
                final_det['couleur'] = consensus_color
            
            # Toujours enregistrer
            detections.append(final_det)
            print(f"✓ Consensus final véhicule {vehicle_id}: {final_det['matricule']} | {final_det['marque']} | {final_det['couleur']}")

        cap.release()
        if not no_display:
            cv2.destroyAllWindows()
  
        if detections:
            # Dédupliquer par similarité de plaque (pas seulement exacte)
            dedup_detections = []
            
            for det in detections:
                plate = det['matricule']
                if not plate or plate == "INCONNU":
                    continue
                
                # Chercher si une plaque similaire existe déjà
                found_similar = False
                for existing in dedup_detections:
                    similarity = self._plate_similarity(plate, existing['matricule'])
                    
                    # Si similarité > 60%, considérer comme la même plaque
                    if similarity >= 0.60:
                        found_similar = True
                        # Garder celui avec la meilleure confiance de plaque
                        if det.get('confidence_plate', 0) > existing.get('confidence_plate', 0):
                            # Remplacer par le nouveau (meilleure confiance)
                            dedup_detections[dedup_detections.index(existing)] = det
                        break
                
                if not found_similar:
                    dedup_detections.append(det)
            
            self.store.update(dedup_detections)
  
            print(f"\n✓ Fini. {len(dedup_detections)} détection(s) unique(s) enregistrée(s) ({len(detections)} avec consensus).")
        else:
            print(f"\n✓ Fini. Aucune détection.")
        
        print(f"✓ Sortie JSON : {self.store.path}")
        return dedup_detections if detections else []


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Traite une vid�o et enregistre les plaques, marques et couleurs dans un fichier JSON."
    )
    parser.add_argument("--source", required=True, help="Chemin de la vid�o ou du flux RTSP")
    parser.add_argument(
        "--log",
        default="models/vehicules.json",
        help="Chemin du fichier JSON de sortie (d�faut: models/vehicules.json)",
    )
    parser.add_argument(
        "--plate-model",
        default="models/best.pt",
        help="Chemin du mod�le de d�tection de plaques (d�faut: models/best.pt)",
    )
    parser.add_argument(
        "--brand-model",
        default="models/brand_yolo.pt",
        help="Chemin du mod�le de d�tection de marque (d�faut: models/brand_yolo.pt)",
    )
    parser.add_argument("--frame-skip", type=int, default=10, help="Traiter 1 frame sur N (d�faut: 10)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Seuil de confiance YOLO (d�faut: 0.5)")
    parser.add_argument("--no-display", action="store_true", help="Ne pas afficher les frames annot�es")
    args = parser.parse_args()

    pipeline = VehiclePipeline(
        plate_model_path=args.plate_model,
        brand_model_path=args.brand_model,
        output_path=args.log,
    )
    pipeline.process_video(
        source=args.source,
        frame_skip=args.frame_skip,
        confidence_threshold=args.confidence,
        no_display=args.no_display,
    )


if __name__ == "__main__":
    main()
