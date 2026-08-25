"""
analyze_gbnlb.py - Vérifie le contenu du dataset téléchargé (sans pandas)
"""
from pathlib import Path
import yaml

dataset_path = "./dataset_gbnlb"

with open(f"{dataset_path}/data.yaml") as f:
    data_cfg = yaml.safe_load(f)

print(f"Nombre de classes : {data_cfg['nc']}")
print(f"Classes : {data_cfg['names']}")

for split in ["train", "valid", "test"]:
    labels_dir = Path(dataset_path) / split / "labels"
    if not labels_dir.exists():
        print(f"\n--- {split} : dossier introuvable, ignoré ---")
        continue

    class_counts = {}
    n_images = len(list(labels_dir.glob("*.txt")))
    total_instances = 0

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                if line.strip():
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    total_instances += 1

    print(f"\n--- {split} ({n_images} images, {total_instances} instances) ---")

    # Tri par nombre d'instances décroissant
    sorted_counts = sorted(class_counts.items(), key=lambda x: -x[1])
    for cls_id, count in sorted_counts:
        class_name = data_cfg["names"][cls_id]
        print(f"  {class_name:20s} : {count}")

    if class_counts:
        counts_values = list(class_counts.values())
        print(f"\n  Min: {min(counts_values)} | Max: {max(counts_values)} | Moyenne: {sum(counts_values)/len(counts_values):.1f}")