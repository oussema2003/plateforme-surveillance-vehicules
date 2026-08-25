"""Nettoie les duplications dans un fichier JSON de véhicules."""

import json
import sys
from pathlib import Path

def plate_similarity(plate1: str, plate2: str) -> float:
    """Calcule la similarité entre deux plaques."""
    if not plate1 or not plate2:
        return 0.0
    
    if plate1 == plate2:
        return 1.0
    
    digits1 = ''.join(c for c in plate1 if c.isdigit())
    digits2 = ''.join(c for c in plate2 if c.isdigit())
    
    if not digits1 or not digits2:
        return 0.0
    
    if abs(len(digits1) - len(digits2)) > 2:
        return 0.0
    
    min_len = min(len(digits1), len(digits2))
    max_len = max(len(digits1), len(digits2))
    exact_matches = sum(1 for i in range(min_len) if digits1[i] == digits2[i])
    
    common_digits = sum(min(digits1.count(d), digits2.count(d)) for d in set(digits1))
    
    if len(digits1) == len(digits2):
        edit_distance = sum(c1 != c2 for c1, c2 in zip(digits1, digits2))
    else:
        edit_distance = max_len
        for i in range(max_len - min_len + 1):
            if len(digits1) < len(digits2):
                substring = digits2[i:i+len(digits1)]
                dist = sum(c1 != c2 for c1, c2 in zip(digits1, substring))
            else:
                substring = digits1[i:i+len(digits2)]
                dist = sum(c1 != c2 for c1, c2 in zip(digits2, substring))
            edit_distance = min(edit_distance, dist + abs(len(digits1) - len(digits2)))
    
    position_score = exact_matches / max_len
    content_score = common_digits / max_len
    edit_score = 1.0 - (edit_distance / max_len)
    
    similarity = (position_score * 0.5 + content_score * 0.2 + edit_score * 0.3)
    
    return similarity


def clean_duplicates(json_path: str, output_path: str = None, threshold: float = 0.70):
    """Nettoie les duplications dans un fichier JSON."""
    
    # Charger le JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Fichier original: {len(data)} entrées")
    print("=" * 80)
    
    # Grouper les plaques similaires
    groups = []
    processed = set()
    
    for plate1 in data.keys():
        if plate1 in processed or plate1 == "INCONNU":
            continue
        
        # Créer un nouveau groupe
        group = {plate1: data[plate1]}
        processed.add(plate1)
        
        # Chercher toutes les plaques similaires
        for plate2 in data.keys():
            if plate2 in processed or plate2 == "INCONNU":
                continue
            
            similarity = plate_similarity(plate1, plate2)
            
            if similarity >= threshold:
                group[plate2] = data[plate2]
                processed.add(plate2)
                print(f"🔗 Groupe trouvé: {plate1} ↔ {plate2} (similarité: {similarity:.2f})")
        
        groups.append(group)
    
    print("=" * 80)
    print(f"📊 Groupes trouvés: {len(groups)}")
    print()
    
    # Fusionner chaque groupe
    cleaned_data = {}
    
    for group in groups:
        if len(group) == 1:
            # Pas de duplication
            plate = list(group.keys())[0]
            cleaned_data[plate] = group[plate]
        else:
            # Choisir la meilleure plaque (meilleure confiance)
            best_plate = None
            best_conf = 0.0
            
            for plate, vehicle in group.items():
                conf = vehicle.get('confidence_plate', 0)
                if conf > best_conf:
                    best_conf = conf
                    best_plate = plate
            
            # Utiliser la meilleure plaque
            cleaned_data[best_plate] = group[best_plate]
            
            print(f"✓ Fusionné: {list(group.keys())} → {best_plate} (conf: {best_conf:.3f})")
    
    print()
    print("=" * 80)
    print(f"✓ Résultat: {len(cleaned_data)} entrées uniques ({len(data) - len(cleaned_data)} dupliqués éliminés)")
    
    # Sauvegarder
    if output_path is None:
        output_path = json_path.replace('.json', '_cleaned.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Sauvegardé dans: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_duplicates.py <fichier.json> [sortie.json] [seuil]")
        print("Exemple: python clean_duplicates.py models/vehicules.json models/vehicules_clean.json 0.70")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.70
    
    clean_duplicates(input_file, output_file, threshold)
