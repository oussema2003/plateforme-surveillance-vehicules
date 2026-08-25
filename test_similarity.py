"""Test de similarité de plaques."""

from clean_duplicates import plate_similarity

# Test des plaques du JSON
test_pairs = [
    ("11 428", "114 3876"),  # Probablement 114 8706
    ("114 7876", "114 3876"),  # Même voiture?
    ("255 4346", "255 2234"),  # Probablement 255 3466
    ("16 955", "16 9331"),     # Possiblement la même
    ("139 5323", "139 5231"),  # Si elle existait
]

print("=" * 80)
print("TEST DE SIMILARITÉ DE PLAQUES")
print("=" * 80)

for plate1, plate2 in test_pairs:
    sim = plate_similarity(plate1, plate2)
    match = "✓ SIMILAIRE" if sim >= 0.70 else "✗ DIFFÉRENT"
    print(f"{match} | {plate1:12} ↔ {plate2:12} | Similarité: {sim:.2%}")

print("=" * 80)
