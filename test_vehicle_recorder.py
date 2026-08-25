"""
test_vehicle_recorder.py
------------------------
Tests unitaires pour le système d'enregistrement des véhicules.
"""

import tempfile
from datetime import datetime
from pathlib import Path

from vehicle_recorder import VehicleData, VehicleRecorder


def test_vehicle_data_creation():
    """Test: créer une instance VehicleData."""
    vehicle = VehicleData(
        plaque="TEST-123",
        marque="Toyota",
        couleur="noir",
        confiance_plaque=0.95,
        confiance_marque=0.90,
        confiance_couleur=0.85,
        timestamp=datetime.now().isoformat(),
        hsv_mean=(0, 0, 50),
    )
    
    assert vehicle.plaque == "TEST-123"
    assert vehicle.marque == "Toyota"
    assert vehicle.couleur == "noir"
    print("✓ test_vehicle_data_creation réussi")


def test_recorder_initialization():
    """Test: initialiser le recorder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        recorder = VehicleRecorder(str(db_path))
        
        assert db_path.exists()
        del recorder  # Libérer la ressource
        print("✓ test_recorder_initialization réussi")


def test_save_and_retrieve():
    """Test: sauvegarder et récupérer un véhicule."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        recorder = VehicleRecorder(str(db_path))
        
        vehicle = VehicleData(
            plaque="TEST-001",
            marque="BMW",
            couleur="blanc",
            confiance_plaque=0.92,
            confiance_marque=0.88,
            confiance_couleur=0.90,
            timestamp=datetime.now().isoformat(),
        )
        
        # Sauvegarder
        vehicle_id = recorder.save_vehicle(vehicle)
        assert vehicle_id is not None
        
        # Récupérer
        retrieved = recorder.get_vehicle("TEST-001")
        assert retrieved is not None
        assert retrieved.plaque == "TEST-001"
        assert retrieved.marque == "BMW"
        assert retrieved.couleur == "blanc"
        
        del recorder  # Libérer la ressource
        print("✓ test_save_and_retrieve réussi")


def test_update_vehicle():
    """Test: mettre à jour un véhicule."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        recorder = VehicleRecorder(str(db_path))
        
        # Créer et sauvegarder
        vehicle1 = VehicleData(
            plaque="TEST-002",
            marque="Toyota",
            couleur="noir",
            confiance_plaque=0.80,
            confiance_marque=0.75,
            confiance_couleur=0.80,
            timestamp=datetime.now().isoformat(),
        )
        recorder.save_vehicle(vehicle1)
        
        # Mettre à jour (même plaque)
        vehicle2 = VehicleData(
            plaque="TEST-002",  # Même plaque
            marque="Mercedes",  # Marque différente
            couleur="gris",     # Couleur différente
            confiance_plaque=0.95,
            confiance_marque=0.93,
            confiance_couleur=0.91,
            timestamp=datetime.now().isoformat(),
        )
        recorder.save_vehicle(vehicle2)
        
        # Vérifier la mise à jour
        retrieved = recorder.get_vehicle("TEST-002")
        assert retrieved.marque == "Mercedes"
        assert retrieved.couleur == "gris"
        assert retrieved.confiance_plaque == 0.95
        
        print("✓ test_update_vehicle réussi")


def test_get_all_vehicles():
    """Test: récupérer tous les véhicules."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        recorder = VehicleRecorder(str(db_path))
        
        # Ajouter plusieurs véhicules
        for i in range(5):
            vehicle = VehicleData(
                plaque=f"TEST-{i:03d}",
                marque=f"Brand-{i}",
                couleur="noir",
                confiance_plaque=0.90,
                confiance_marque=0.85,
                confiance_couleur=0.80,
                timestamp=datetime.now().isoformat(),
            )
            recorder.save_vehicle(vehicle)
        
        # Récupérer tous
        vehicles = recorder.get_all_vehicles()
        assert len(vehicles) == 5
        
        print("✓ test_get_all_vehicles réussi")


def test_export_json():
    """Test: exporter en JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        export_path = Path(tmpdir) / "export.json"
        recorder = VehicleRecorder(str(db_path))
        
        # Ajouter des véhicules
        vehicle = VehicleData(
            plaque="TEST-JSON",
            marque="Audi",
            couleur="rouge",
            confiance_plaque=0.88,
            confiance_marque=0.86,
            confiance_couleur=0.84,
            timestamp=datetime.now().isoformat(),
        )
        recorder.save_vehicle(vehicle)
        
        # Exporter
        recorder.export_to_json(str(export_path))
        
        # Vérifier
        assert export_path.exists()
        
        import json
        with open(export_path) as f:
            data = json.load(f)
        
        assert 'vehicles' in data
        assert len(data['vehicles']) == 1
        assert data['vehicles'][0]['plaque'] == "TEST-JSON"
        
        print("✓ test_export_json réussi")


def test_export_csv():
    """Test: exporter en CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        export_path = Path(tmpdir) / "export.csv"
        recorder = VehicleRecorder(str(db_path))
        
        # Ajouter des véhicules
        vehicle = VehicleData(
            plaque="TEST-CSV",
            marque="Honda",
            couleur="bleu",
            confiance_plaque=0.87,
            confiance_marque=0.85,
            confiance_couleur=0.83,
            timestamp=datetime.now().isoformat(),
        )
        recorder.save_vehicle(vehicle)
        
        # Exporter
        recorder.export_to_csv(str(export_path))
        
        # Vérifier
        assert export_path.exists()
        
        with open(export_path) as f:
            content = f.read()
        
        assert "TEST-CSV" in content
        assert "Honda" in content
        
        print("✓ test_export_csv réussi")


def test_statistics():
    """Test: calculer les statistiques."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        recorder = VehicleRecorder(str(db_path))
        
        # Ajouter des véhicules variés
        vehicles_data = [
            ("TEST-001", "Toyota", "noir"),
            ("TEST-002", "Toyota", "blanc"),
            ("TEST-003", "BMW", "noir"),
            ("TEST-004", "BMW", "noir"),
            ("TEST-005", "Mercedes", "gris"),
        ]
        
        for plaque, marque, couleur in vehicles_data:
            vehicle = VehicleData(
                plaque=plaque,
                marque=marque,
                couleur=couleur,
                confiance_plaque=0.90,
                confiance_marque=0.85,
                confiance_couleur=0.80,
                timestamp=datetime.now().isoformat(),
            )
            recorder.save_vehicle(vehicle)
        
        # Obtenir stats
        stats = recorder.get_statistics()
        
        assert stats['total_vehicles'] == 5
        assert 'Toyota' in stats['top_marques']
        assert stats['top_marques']['Toyota'] == 2
        assert stats['top_marques']['BMW'] == 2
        assert 'noir' in stats['top_couleurs']
        assert stats['top_couleurs']['noir'] == 3
        
        print("✓ test_statistics réussi")


def run_all_tests():
    """Exécute tous les tests."""
    print("\n" + "="*60)
    print("🧪 TESTS DU SYSTÈME D'ENREGISTREMENT DE VÉHICULES")
    print("="*60 + "\n")
    
    tests = [
        test_vehicle_data_creation,
        test_recorder_initialization,
        test_save_and_retrieve,
        test_update_vehicle,
        test_get_all_vehicles,
        test_export_json,
        test_export_csv,
        test_statistics,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} échoué: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} erreur: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"📊 Résultats: {passed} réussi(s), {failed} échoué(s)")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
