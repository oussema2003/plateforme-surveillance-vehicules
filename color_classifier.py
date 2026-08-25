"""
color_classifier.py
--------------------
Classification de la couleur dominante d'un crop de vehicule (OpenCV pur,
pas de modele ML).

Points cles pour la fiabilite (notamment noir vs gris fonce) :
  - On ne travaille PAS sur tout le crop du vehicule : le tiers superieur
    (pare-brise / vitres, qui refletent le ciel -> fausse le gris/bleu clair)
    et les bords (fond, route, ombre) sont exclus. On se concentre sur la
    carrosserie centrale.
  - On ecarte les pixels de reflet pur (tres haute luminosite + faible
    saturation) et les pixels trop sombres pour etre fiables (ombre profonde
    sans info de teinte), qui biaisent la moyenne.
  - Le pixel dominant est trouve par K-means (plus robuste qu'une moyenne ou
    qu'un histogramme brut face au bruit/aux reflets ponctuels).
  - Les seuils V (luminosite) noir / gris fonce / gris clair ont ete elargis
    car les cameras de securite (notamment en vision de nuit / IR) remontent
    des noirs avec un V nettement superieur a 0-30 (parfois jusqu'a 70-80).
    Un seuil noir trop bas classait donc systematiquement les vraies voitures
    noires en "gris fonce".
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ColorResult:
    name: str
    confidence: float
    hsv_mean: tuple  # utile pour debug/calibration


# --------------------------------------------------------------------------- #
# Seuils de luminosite (V, 0-255) et de saturation (S, 0-255) en HSV OpenCV.
# AMÉLIORATION MAJEURE : Seuils très stricts pour bien détecter le noir
# et éviter les faux positifs (gris foncé, blanc)
# --------------------------------------------------------------------------- #
V_BLACK_MAX = 120          # V en dessous -> noir (augmenté pour capturer les noirs en conditions réelles)
V_DARK_GRAY_MAX = 165      # V entre BLACK_MAX et ce seuil -> gris fonce (si S faible)
V_LIGHT_GRAY_MAX = 220     # V entre DARK_GRAY_MAX et ce seuil -> gris clair/argent (si S faible)
V_WHITE_MIN = 200          # V au-dessus -> blanc (si S faible) - évite de classer comme blanc avec des taches

S_ACHROMATIC_MAX = 40      # en dessous de cette saturation -> couleur "grise" (noir/gris/blanc)
S_ACHROMATIC_MAX_DARK = 60  # tolerance de saturation pour les tres sombres

# Reflets/eclats a exclure du calcul (trop brillants pour porter une info de
# teinte fiable) et zones trop sombres a exclure (ombre profonde / trous noirs
# du crop, pas la carrosserie elle-meme).
V_REFLECT_MIN = 245
V_TOO_DARK_MIN = 12  # sous ce seuil : quasi-noir pur / bord du crop, on garde
                       # quand meme (utile pour detecter le noir) mais on
                       # l'exclut du calcul si presque tout le crop est comme ca

# Plages de teinte (H, 0-179 dans OpenCV) pour les couleurs chromatiques.
HUE_RANGES = [
    ("rouge", [(0, 8), (170, 179)]),
    ("orange", [(9, 20)]),
    ("jaune", [(21, 34)]),
    ("vert", [(35, 85)]),
    ("cyan", [(86, 100)]),
    ("bleu", [(101, 130)]),
    ("violet", [(131, 155)]),
    ("rose / rouge", [(156, 169)]),
]


def _hue_to_name(hue: float) -> str:
    hue = hue % 180
    for name, ranges in HUE_RANGES:
        for lo, hi in ranges:
            if lo <= hue <= hi:
                return name
    return "indefini"


def _achromatic_name(v: float) -> str:
    # Luminosite (V) en HSV: 0-255
    # Noir: V tres bas (0-120)
    # Gris fonce: V bas-moyen (120-165)
    # Gris clair: V moyen-haut (165-220)
    # Blanc: V tres haut (200-255)
    if v < V_BLACK_MAX:
        return "noir"
    if v < V_DARK_GRAY_MAX:
        return "gris fonce"
    if v < V_LIGHT_GRAY_MAX:
        return "gris clair / argent"
    if v >= V_WHITE_MIN:
        return "blanc"
    # Entre V_LIGHT_GRAY_MAX et V_WHITE_MIN -> gris clair
    return "gris clair / argent"



class ColorClassifier:
    def __init__(self, n_clusters: int = 3, sample_max_pixels: int = 4000):
        self.n_clusters = n_clusters
        self.sample_max_pixels = sample_max_pixels

    # ------------------------------------------------------------------ #
    @staticmethod
    def _central_body_crop(bgr: np.ndarray) -> np.ndarray:
        """Isole la carrosserie centrale : exclut le toit/pare-brise (reflets
        du ciel), le bas (ombre/route/pare-chocs sombre) et les bords lateraux
        (fond, vehicules voisins).
        AMÉLIORATION: Zone centrale plus restreinte pour éviter les taches blanches."""
        h, w = bgr.shape[:2]
        if h < 10 or w < 10:
            return bgr
        # Zone plus centrale pour éviter les reflets et taches
        y1, y2 = int(h * 0.40), int(h * 0.75)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        crop = bgr[y1:y2, x1:x2]
        return crop if crop.size > 0 else bgr

    # ------------------------------------------------------------------ #
    def _dominant_hsv(self, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """K-means sur les pixels HSV valides -> (hsv_moyen_du_cluster_dominant,
        proportion_de_pixels_dans_ce_cluster)."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

        v = hsv[:, 2]
        s = hsv[:, 1]
        # exclut les reflets purs (trop brillants pour porter une teinte fiable)
        valid_mask = v < V_REFLECT_MIN
        hsv_valid = hsv[valid_mask]
        if hsv_valid.shape[0] < 20:
            hsv_valid = hsv  # crop degenere -> on garde tout plutot que rien

        # sous-echantillonnage pour la vitesse (K-means sur 4000 pixels max)
        if hsv_valid.shape[0] > self.sample_max_pixels:
            idx = np.random.choice(hsv_valid.shape[0], self.sample_max_pixels, replace=False)
            hsv_valid = hsv_valid[idx]

        k = min(self.n_clusters, max(1, hsv_valid.shape[0] // 10))
        if k <= 1:
            return hsv_valid.mean(axis=0), 1.0

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
        _compactness, labels, centers = cv2.kmeans(
            hsv_valid, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=k)
        dominant_idx = int(np.argmax(counts))
        proportion = counts[dominant_idx] / labels.shape[0]
        return centers[dominant_idx], float(proportion)

    # ------------------------------------------------------------------ #
    def get_dominant_color(self, bgr_crop: np.ndarray) -> ColorResult:
        if bgr_crop is None or bgr_crop.size == 0:
            return ColorResult(name="?", confidence=0.0, hsv_mean=(0, 0, 0))

        body = self._central_body_crop(bgr_crop)
        if body.size == 0:
            body = bgr_crop

        h_mean, s_mean, v_mean = self._dominant_hsv(body)[0]
        proportion = self._dominant_hsv(body)[1]  # recalcule pour clarte (cout negligeable)

        # tolerance de saturation plus large quand c'est tres sombre : le
        # bruit capteur/JPEG gonfle artificiellement S a bas V, ce qui faisait
        # auparavant classer des noirs en couleur chromatique (ex. "bleu").
        s_threshold = S_ACHROMATIC_MAX_DARK if v_mean < V_DARK_GRAY_MAX else S_ACHROMATIC_MAX

        if s_mean < s_threshold:
            name = _achromatic_name(v_mean)
        else:
            name = _hue_to_name(h_mean)
            # une teinte tres sombre (V bas) meme saturee est le plus souvent
            # un noir avec reflet colore parasite -> on retombe sur noir
            if v_mean < V_BLACK_MAX:
                name = "noir"
            elif name == "gris fonce" and 101 <= h_mean <= 130 and s_mean > 35:
                name = "bleu"

        # CORRECTIONS FINALES STRICTES pour le noir
        # 1. Si luminosité très basse (< 115), c'est forcément noir
        if v_mean < 115:
            name = "noir"
        # 2. Si classifié comme gris foncé mais luminosité < 130, c'est noir
        elif name == "gris fonce" and v_mean < 130:
            name = "noir"
        # 3. Si blanc mais luminosité moyenne (pas vraiment brillant), revoir
        elif name == "blanc" and v_mean < V_WHITE_MIN:
            # Probablement des taches blanches sur carrosserie plus foncée
            # Recalculer avec un seuil plus strict
            if v_mean < 150:
                name = "noir"
            elif v_mean < 180:
                name = "gris fonce"
            else:
                name = "gris clair / argent"

        return ColorResult(
            name=name,
            confidence=round(float(proportion), 2),
            hsv_mean=(round(float(h_mean), 1), round(float(s_mean), 1), round(float(v_mean), 1)),
        )
