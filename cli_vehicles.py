"""
cli_vehicles.py
---------------
Interface en ligne de commande pour gérer les données des véhicules.
Permet de :
  - Traiter des images individuelles
  - Traiter des vidéos
  - Consulter la base de données
  - Exporter les données
  - Afficher les statistiques
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import cv2

from vehicle_recorder import (
    VehicleDetectionPipeline,
    VehicleRecorder,
    VehicleData,
)


def cmd_process_image(args):
    """Traite une image individuelle."""
    recorder = VehicleRecorder(args.db)
    
    pipeline = VehicleDetectionPipeline(
        plate_model_path=args.plate_model,
        brand_model_path=args.brand_model,
        recorder=recorder,
    )
    
    print(f"📸 Traitement de l'image: {args.image}")
    
    vehicles = pipeline.process_image(
        args.image,
        confidence_threshold=args.confidence,
    )
    
    if not vehicles:
        print("❌ Aucun véhicule détecté")
        return 1
    
    print(f"\n✓ {len(vehicles)} véhicule(s) détecté(s):\n")
    
    for i, vehicle in enumerate(vehicles, 1):
        print(f"{i}. Plaque: {vehicle.plaque}")
        print(f"   Marque: {vehicle.marque}")
        print(f"   Couleur: {vehicle.couleur}")
        print(f"   Confiances: plaque={vehicle.confiance_plaque:.1%}, "
              f"marque={vehicle.confiance_marque:.1%}, "
              f"couleur={vehicle.confiance_couleur:.1%}")
        print()
    
    return 0


def cmd_process_video(args):
    """Traite une vidéo."""
    recorder = VehicleRecorder(args.db)
    
    pipeline = VehicleDetectionPipeline(
        plate_model_path=args.plate_model,
        brand_model_path=args.brand_model,
        recorder=recorder,
    )
    
    print(f"🎬 Traitement vidéo: {args.video}")
    print(f"   Mode: traiter 1 frame sur {args.frame_skip}\n")
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la vidéo: {args.video}")
        return 1
    
    frame_count = 0
    detected_count = 0
    vehicles_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % args.frame_skip == 0:
            vehicles = pipeline.process_frame(
                frame,
                confidence_threshold=args.confidence,
                source_image=f"{args.video}@{frame_count}",
            )
            
            if vehicles:
                detected_count += len(vehicles)
                vehicles_list.extend(vehicles)
                
                for v in vehicles:
                    print(f"Frame {frame_count:5d}: {v.plaque:12s} {v.marque:15s} {v.couleur:12s}")
    
    cap.release()
    
    output_json = args.output
    recorder.export_to_json(output_json)
    
    print(f"\n{'='*60}")
    print(f"✓ Traitement terminé:")
    print(f"   Frames traitées: {frame_count // args.frame_skip}")
    print(f"   Véhicules uniques détectés: {len(set(v.plaque for v in vehicles_list))}")
    print(f"   Détections totales: {detected_count}")
    print(f"   Fichier JSON généré: {output_json}")
    print(f"{'='*60}\n")
    
    return 0


def cmd_list(args):
    """Liste les véhicules enregistrés."""
    recorder = VehicleRecorder(args.db)
    vehicles = recorder.get_all_vehicles(limit=args.limit)
    
    if not vehicles:
        print("❌ Aucun véhicule enregistré")
        return 1
    
    print(f"\n{'Plaque':<15} {'Marque':<20} {'Couleur':<15} {'Détecté le':<20}")
    print("-" * 75)
    
    for v in vehicles:
        timestamp = v.timestamp.split('T')[0]  # Format court YYYY-MM-DD
        print(f"{v.plaque:<15} {v.marque:<20} {v.couleur:<15} {timestamp:<20}")
    
    print(f"\nTotal: {len(vehicles)} véhicule(s)")
    return 0


def cmd_get(args):
    """Affiche les détails d'un véhicule."""
    recorder = VehicleRecorder(args.db)
    vehicle = recorder.get_vehicle(args.plaque.upper())
    
    if not vehicle:
        print(f"❌ Véhicule introuvable: {args.plaque}")
        return 1
    
    print(f"\n{'='*50}")
    print(f"📋 Détails du véhicule: {vehicle.plaque}")
    print(f"{'='*50}")
    print(f"Marque:              {vehicle.marque}")
    print(f"Couleur:             {vehicle.couleur}")
    print(f"Confiance plaque:    {vehicle.confiance_plaque:.1%}")
    print(f"Confiance marque:    {vehicle.confiance_marque:.1%}")
    print(f"Confiance couleur:   {vehicle.confiance_couleur:.1%}")
    print(f"Détecté le:          {vehicle.timestamp}")
    if vehicle.source_image:
        print(f"Source:              {vehicle.source_image}")
    if vehicle.hsv_mean:
        print(f"Données couleur HSV: H={vehicle.hsv_mean[0]:.1f}, S={vehicle.hsv_mean[1]:.1f}, V={vehicle.hsv_mean[2]:.1f}")
    print(f"{'='*50}\n")
    
    return 0


def cmd_stats(args):
    """Affiche les statistiques."""
    recorder = VehicleRecorder(args.db)
    recorder.print_statistics()
    return 0


def cmd_export(args):
    """Exporte les données."""
    recorder = VehicleRecorder(args.db)
    
    if args.format == "json" or args.format == "all":
        output_json = args.output or "data/vehicles_export.json"
        recorder.export_to_json(output_json)
    
    if args.format == "csv" or args.format == "all":
        output_csv = args.output or "data/vehicles_export.csv"
        # Remplacer .json par .csv si nécessaire
        if output_csv.endswith(".json"):
            output_csv = output_csv.replace(".json", ".csv")
        recorder.export_to_csv(output_csv)
    
    return 0


def cmd_filter(args):
    """Filtre les véhicules par attribut."""
    recorder = VehicleRecorder(args.db)
    vehicles = recorder.get_all_vehicles()
    
    if not vehicles:
        print("❌ Aucun véhicule enregistré")
        return 1
    
    # Filtrer par critère
    if args.color:
        vehicles = [v for v in vehicles if args.color.lower() in v.couleur.lower()]
        print(f"\n🎨 Véhicules {args.color}:\n")
    
    elif args.brand:
        vehicles = [v for v in vehicles if args.brand.lower() in v.marque.lower()]
        print(f"\n🏷️ Véhicules {args.brand}:\n")
    
    elif args.confidence:
        vehicles = [v for v in vehicles if v.confiance_plaque >= args.confidence]
        print(f"\n🔍 Véhicules avec confiance >= {args.confidence:.1%}:\n")
    
    if not vehicles:
        print("❌ Aucun résultat")
        return 1
    
    for v in vehicles:
        print(f"{v.plaque:<15} {v.marque:<20} {v.couleur:<15} (conf: {v.confiance_plaque:.1%})")
    
    print(f"\nRésultats: {len(vehicles)} véhicule(s)")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Gestionnaire de données de véhicules détectés par YOLO"
    )
    
    parser.add_argument("--db", default="data/vehicles.db",
                        help="Chemin de la base de données (défaut: data/vehicles.db)")
    parser.add_argument("--plate-model", default="models/best.pt",
                        help="Chemin du modèle YOLO plaque (défaut: models/best.pt)")
    parser.add_argument("--brand-model", default="models/brand_yolo.pt",
                        help="Chemin du modèle YOLO marque (défaut: models/brand_yolo.pt)")
    
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Commande: image
    img_parser = subparsers.add_parser("image", help="Traiter une image")
    img_parser.add_argument("image", help="Chemin de l'image")
    img_parser.add_argument("--confidence", type=float, default=0.5,
                            help="Seuil de confiance (défaut: 0.5)")
    img_parser.set_defaults(func=cmd_process_image)
    
    # Commande: video
    video_parser = subparsers.add_parser("video", help="Traiter une vidéo")
    video_parser.add_argument("video", help="Chemin de la vidéo")
    video_parser.add_argument("--frame-skip", type=int, default=10,
                              help="Traiter 1 frame sur N (défaut: 10)")
    video_parser.add_argument("--confidence", type=float, default=0.5,
                              help="Seuil de confiance (défaut: 0.5)")
    video_parser.add_argument("--output", default="models/vehicules.json",
                              help="Chemin du fichier JSON de sortie (défaut: models/vehicules.json)")
    video_parser.set_defaults(func=cmd_process_video)
    
    # Commande: list
    list_parser = subparsers.add_parser("list", help="Lister les véhicules")
    list_parser.add_argument("--limit", type=int,
                             help="Limiter le nombre de résultats")
    list_parser.set_defaults(func=cmd_list)
    
    # Commande: get
    get_parser = subparsers.add_parser("get", help="Afficher les détails d'un véhicule")
    get_parser.add_argument("plaque", help="Numéro de plaque")
    get_parser.set_defaults(func=cmd_get)
    
    # Commande: stats
    stats_parser = subparsers.add_parser("stats", help="Afficher les statistiques")
    stats_parser.set_defaults(func=cmd_stats)
    
    # Commande: export
    export_parser = subparsers.add_parser("export", help="Exporter les données")
    export_parser.add_argument("--format", choices=["json", "csv", "all"],
                               default="all", help="Format d'export (défaut: all)")
    export_parser.add_argument("--output", help="Chemin de sortie")
    export_parser.set_defaults(func=cmd_export)
    
    # Commande: filter
    filter_parser = subparsers.add_parser("filter", help="Filtrer les véhicules")
    filter_group = filter_parser.add_mutually_exclusive_group(required=True)
    filter_group.add_argument("--color", help="Filtrer par couleur")
    filter_group.add_argument("--brand", help="Filtrer par marque")
    filter_group.add_argument("--confidence", type=float,
                              help="Filtrer par confiance minimum")
    filter_parser.set_defaults(func=cmd_filter)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interruption utilisateur")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
