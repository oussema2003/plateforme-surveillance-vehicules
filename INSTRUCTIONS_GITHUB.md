# 📝 Instructions pour pousser sur GitHub

## ✅ Ce qui a été fait

1. ✅ Dépôt Git initialisé localement
2. ✅ Fichier `.gitignore` créé (ignore les fichiers volumineux et temporaires)
3. ✅ Fichier `LICENSE` créé (Licence MIT)
4. ✅ README.md complet et détaillé créé
5. ✅ Premier commit effectué avec tous les fichiers

## 📋 Étapes suivantes

### 1. Créer un dépôt sur GitHub

1. Allez sur [GitHub](https://github.com)
2. Cliquez sur le bouton **"+"** en haut à droite → **"New repository"**
3. Remplissez les informations :
   - **Repository name**: `plateforme-surveillance-vehicules` (ou votre choix)
   - **Description**: `Système intelligent de reconnaissance de véhicules avec détection de plaques, marques et couleurs`
   - **Visibilité**: Choisissez **Public** ou **Private**
   - ⚠️ **NE COCHEZ PAS** "Initialize with README" (nous en avons déjà un)
   - ⚠️ **NE COCHEZ PAS** "Add .gitignore" (nous en avons déjà un)
   - ⚠️ **NE COCHEZ PAS** "Add license" (nous en avons déjà un)
4. Cliquez sur **"Create repository"**

### 2. Lier le dépôt local au dépôt GitHub

GitHub va vous montrer plusieurs options. Suivez celle pour **"push an existing repository from the command line"**.

Ouvrez un terminal/PowerShell dans le dossier du projet et exécutez :

```bash
# Remplacez VOTRE_USERNAME par votre nom d'utilisateur GitHub
git remote add origin https://github.com/VOTRE_USERNAME/plateforme-surveillance-vehicules.git

# Renommer la branche principale en "main" (optionnel mais recommandé)
git branch -M main

# Pousser le code sur GitHub
git push -u origin main
```

**Exemple concret :**
```bash
# Si votre username GitHub est "johndoe"
git remote add origin https://github.com/johndoe/plateforme-surveillance-vehicules.git
git branch -M main
git push -u origin main
```

### 3. Authentification

Lors du `git push`, GitHub vous demandera de vous authentifier.

#### Option A : Personal Access Token (Recommandé)

1. Allez sur GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Cliquez sur "Generate new token (classic)"
3. Donnez un nom au token (ex: "Plateforme Surveillance")
4. Cochez **`repo`** (accès complet au dépôt)
5. Cliquez sur "Generate token"
6. **COPIEZ le token** (vous ne pourrez plus le voir après)
7. Lors du `git push`, utilisez :
   - **Username**: votre username GitHub
   - **Password**: le token (PAS votre mot de passe GitHub)

#### Option B : GitHub CLI (Alternative)

```bash
# Installer GitHub CLI
# Windows : https://cli.github.com/
# Ensuite :
gh auth login
gh repo create plateforme-surveillance-vehicules --public --source=. --remote=origin
git push -u origin main
```

### 4. Vérification

Allez sur `https://github.com/VOTRE_USERNAME/plateforme-surveillance-vehicules` pour voir votre projet en ligne !

## 🎨 Améliorations post-publication

### Ajouter un badge "Star"

Ajoutez ce badge en haut de votre README.md :

```markdown
[![GitHub stars](https://img.shields.io/github/stars/VOTRE_USERNAME/plateforme-surveillance-vehicules.svg?style=social&label=Star)](https://github.com/VOTRE_USERNAME/plateforme-surveillance-vehicules)
```

### Ajouter des topics au dépôt

Sur GitHub, dans votre dépôt :
1. Cliquez sur ⚙️ (Settings) à côté de "About"
2. Ajoutez des topics : `computer-vision`, `yolo`, `ocr`, `fastapi`, `vehicle-detection`, `license-plate-recognition`, `docker`, `python`

### Créer une GitHub Page (optionnel)

Pour héberger la documentation :
1. Settings → Pages
2. Source : Deploy from a branch
3. Branch : `main` → `/docs` (ou créer une branche `gh-pages`)

## 📂 Structure des fichiers ignorés

Les fichiers suivants ne seront **PAS poussés** sur GitHub (trop volumineux ou sensibles) :

- ❌ Modèles YOLO (`.pt`) - Trop volumineux (100+ MB)
- ❌ Vidéos (`.mp4`, `.avi`) - Trop volumineuses
- ❌ Images de plaques sauvegardées - Données sensibles
- ❌ Fichiers JSON de sortie - Données en temps réel
- ❌ Environnements virtuels (`venv/`) - À recréer localement

## 🔄 Futures mises à jour

Pour pousser de nouvelles modifications :

```bash
# Voir les fichiers modifiés
git status

# Ajouter les modifications
git add .

# Créer un commit
git commit -m "fix: correction de la détection de couleur"

# Pousser sur GitHub
git push
```

## 🆘 Problèmes courants

### Erreur : "remote origin already exists"

```bash
git remote remove origin
git remote add origin https://github.com/VOTRE_USERNAME/plateforme-surveillance-vehicules.git
```

### Erreur : "Authentication failed"

- Vérifiez que vous utilisez un **Personal Access Token** et non votre mot de passe
- Le token doit avoir les permissions **`repo`**

### Erreur : "file is too large"

Si un fichier est trop gros (>100 MB), ajoutez-le au `.gitignore` :

```bash
# Retirer le fichier du staging
git rm --cached chemin/vers/fichier-volumineux.pt

# L'ajouter au .gitignore
echo "chemin/vers/fichier-volumineux.pt" >> .gitignore

# Recommiter
git add .gitignore
git commit -m "chore: ignore fichier volumineux"
```

## 📧 Support

Si vous rencontrez des problèmes, consultez :
- [GitHub Docs - Pushing commits](https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository)
- [GitHub Docs - Authentication](https://docs.github.com/en/authentication)

---

✅ **Tout est prêt ! Il ne reste plus qu'à créer le dépôt sur GitHub et exécuter les commandes ci-dessus.**
