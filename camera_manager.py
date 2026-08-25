"""
camera_manager.py
------------------
Gere l'ajout / l'arret dynamique de flux camera (RTSP ou fichier video)
depuis l'API / le dashboard web, SANS avoir a relancer les conteneurs
Docker ou a editer docker-compose.yml.

Chaque camera ajoutee tourne dans son PROPRE thread avec sa PROPRE
instance de VehiclePipeline (donc ses propres modeles YOLO charges en
memoire). C'est plus simple et plus sur (pas d'etat partage entre
cameras) mais plus gourmand en RAM/VRAM si vous ajoutez beaucoup de
cameras simultanement -- pour un usage avec quelques cameras (2-6), ca
reste tout a fait raisonnable sur un CPU/GPU moderne.

Toutes les cameras ecrivent leurs detections dans le MEME fichier JSON
partage (VehicleJSONStore / OUTPUT_PATH), donc le dashboard voit un flux
unique de vehicules, toutes cameras confondues -- comme avec
docker-compose, mais sans redemarrage.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2

from vehicle_pipeline import VehiclePipeline


@dataclass
class CameraInfo:
    id: str
    name: str
    source: str
    frame_skip: int
    confidence: float
    status: str = "starting"          # starting | running | stopped | ended | error
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    pipeline: Optional[VehiclePipeline] = None
    thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)

    def to_public_dict(self) -> dict:
        """Version JSON-serialisable (sans le thread / pipeline / event)."""
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "frame_skip": self.frame_skip,
            "confidence": self.confidence,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
        }


class CameraManager:
    """Cree/arrete des workers VehiclePipeline a la volee, un thread par camera."""

    def __init__(
        self,
        vehicle_model_path: str,
        plate_model_path: str,
        brand_model_path: str,
        output_path: str,
        plate_images_dir: str = "data/plate_images",
    ):
        self._vehicle_model_path = vehicle_model_path
        self._plate_model_path = plate_model_path
        self._brand_model_path = brand_model_path
        self._output_path = output_path
        self._plate_images_dir = plate_images_dir
        self._cameras: Dict[str, CameraInfo] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def list_cameras(self) -> List[dict]:
        with self._lock:
            return [cam.to_public_dict() for cam in self._cameras.values()]

    def get_camera(self, camera_id: str) -> Optional[CameraInfo]:
        with self._lock:
            return self._cameras.get(camera_id)

    # ------------------------------------------------------------------ #
    def add_camera(
        self,
        name: str,
        source: str,
        frame_skip: int = 10,
        confidence: float = 0.5,
    ) -> dict:
        """Demarre une nouvelle camera dans un thread dedie et la reference."""
        camera_id = uuid.uuid4().hex[:8]
        cam = CameraInfo(
            id=camera_id,
            name=name,
            source=source,
            frame_skip=frame_skip,
            confidence=confidence,
        )
        with self._lock:
            self._cameras[camera_id] = cam

        thread = threading.Thread(target=self._run_camera, args=(cam,), daemon=True)
        cam.thread = thread
        thread.start()
        return cam.to_public_dict()

    def _run_camera(self, cam: CameraInfo) -> None:
        try:
            # Chaque camera charge ses propres modeles -> isole des autres.
            cam.pipeline = VehiclePipeline(
                vehicle_model_path=self._vehicle_model_path,
                plate_model_path=self._plate_model_path,
                brand_model_path=self._brand_model_path,
                output_path=self._output_path,
                plate_images_dir=self._plate_images_dir,
            )
            cam.status = "running"
            cam.pipeline.process_video(
                source=cam.source,
                frame_skip=cam.frame_skip,
                confidence_threshold=cam.confidence,
                no_display=True,
                stop_event=cam.stop_event,
            )
            cam.status = "stopped" if cam.stop_event.is_set() else "ended"
        except Exception as exc:  # noqa: BLE001 - on veut capturer tout et l'afficher au dashboard
            cam.status = "error"
            cam.error = str(exc)

    # ------------------------------------------------------------------ #
    def stop_camera(self, camera_id: str) -> bool:
        """Demande l'arret propre (le thread se termine a la prochaine frame)."""
        cam = self.get_camera(camera_id)
        if cam is None:
            return False
        cam.stop_event.set()
        return True

    def remove_camera(self, camera_id: str) -> bool:
        """Arrete puis retire la camera de la liste (le thread peut mettre
        jusqu'a une frame pour se terminer reellement, c'est normal)."""
        self.stop_camera(camera_id)
        with self._lock:
            return self._cameras.pop(camera_id, None) is not None

    # ------------------------------------------------------------------ #
    def get_last_frame_jpeg(self, camera_id: str) -> Optional[bytes]:
        """Encode la derniere frame annotee de cette camera en JPEG, pour
        que l'API puisse la servir directement au dashboard (aperçu live)."""
        cam = self.get_camera(camera_id)
        if cam is None or cam.pipeline is None or cam.pipeline.last_annotated_frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", cam.pipeline.last_annotated_frame)
        return buf.tobytes() if ok else None
