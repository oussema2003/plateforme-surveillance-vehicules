"""
⚠️  ATTENTION : Ce script est INUTILE pour votre cas d'usage !

Utilisez directement le fichier vidéo dans le dashboard :
    NOM: Test Video
    FLUX: test.mp4
    
Le système supporte déjà les fichiers vidéo locaux !

---

Si vous DEVEZ vraiment utiliser RTSP (non recommandé), suivez ces étapes :

PRÉREQUIS :
1. Installer FFmpeg : https://ffmpeg.org/download.html
2. Installer MediaMTX : https://github.com/bluenviron/mediamtx/releases
3. Lancer MediaMTX dans un terminal séparé : mediamtx.exe
4. Puis lancer ce script

UTILISATION :
    python start_rtsp_server.py test.mp4

Ensuite dans le dashboard, entrez : rtsp://localhost:8554/test
"""

import subprocess
import sys
from pathlib import Path
import time

def check_ffmpeg():
    """Vérifie que FFmpeg est installé."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def start_rtsp_stream(video_file: str, rtsp_url: str = "rtsp://localhost:8554/test"):
    """
    Streame une vidéo vers un serveur RTSP.
    
    IMPORTANT: MediaMTX (ou serveur RTSP similaire) doit être lancé AVANT !
    
    Args:
        video_file: Chemin vers le fichier vidéo
        rtsp_url: URL RTSP de destination
    """
    video_path = Path(video_file)
    
    if not video_path.exists():
        print(f"❌ Erreur: Fichier vidéo introuvable: {video_file}")
        print(f"   Cherché dans: {video_path.absolute()}")
        sys.exit(1)
    
    if not check_ffmpeg():
        print("❌ Erreur: FFmpeg n'est pas installé ou pas dans le PATH")
        print()
        print("📥 Téléchargez FFmpeg:")
        print("   Windows: https://www.gyan.dev/ffmpeg/builds/")
        print("   Ajoutez FFmpeg au PATH système")
        sys.exit(1)
    
    print("=" * 80)
    print("⚠️  RAPPEL : Solution NON RECOMMANDÉE !")
    print("   → Utilisez directement 'test.mp4' dans le dashboard")
    print("=" * 80)
    print()
    print("🎬 Streaming RTSP")
    print("=" * 80)
    print(f"📹 Vidéo source : {video_path.absolute()}")
    print(f"🔗 URL RTSP    : {rtsp_url}")
    print()
    print("⚠️  IMPORTANT :")
    print("   1. MediaMTX doit être lancé (dans un autre terminal)")
    print("   2. Commande : mediamtx.exe")
    print("   3. Par défaut, MediaMTX écoute sur le port 8554")
    print()
    print("💡 Dans le dashboard, entrez :")
    print(f"   FLUX: {rtsp_url}")
    print()
    print("⏹️  Appuyez sur Ctrl+C pour arrêter")
    print("=" * 80)
    print()
    
    time.sleep(2)  # Pause pour lire les messages
    
    # Commande FFmpeg optimisée
    cmd = [
        "ffmpeg",
        "-re",                          # Vitesse réelle
        "-stream_loop", "-1",           # Boucle infinie
        "-i", str(video_path),          # Fichier source
        "-c:v", "libx264",              # Codec H.264
        "-preset", "ultrafast",         # Preset rapide
        "-tune", "zerolatency",         # Faible latence
        "-b:v", "2M",                   # Bitrate 2 Mbps
        "-maxrate", "2M",               # Bitrate max
        "-bufsize", "4M",               # Buffer
        "-g", "30",                     # Keyframe interval
        "-c:a", "aac",                  # Codec audio
        "-b:a", "128k",                 # Bitrate audio
        "-f", "rtsp",                   # Format RTSP
        "-rtsp_transport", "tcp",       # Transport TCP (plus fiable)
        rtsp_url                        # URL destination
    ]
    
    print("🚀 Lancement FFmpeg...")
    print()
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n⏹️  Streaming arrêté")
        print()
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur FFmpeg: {e}")
        print()
        print("💡 Vérifiez que MediaMTX est bien lancé !")
        sys.exit(1)


if __name__ == "__main__":
    print()
    print("⚠️" * 40)
    print()
    print("        CE SCRIPT N'EST PAS NÉCESSAIRE !")
    print()
    print("   Utilisez directement le fichier vidéo dans le dashboard")
    print("   NOM: Test Video")
    print("   FLUX: test.mp4")
    print()
    print("⚠️" * 40)
    print()
    
    response = input("Voulez-vous vraiment continuer avec RTSP ? (oui/non): ")
    
    if response.lower() not in ['oui', 'yes', 'o', 'y']:
        print("\n✅ Bonne décision ! Utilisez directement 'test.mp4'")
        sys.exit(0)
    
    print()
    
    video_file = "test.mp4"
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    
    start_rtsp_stream(video_file)
