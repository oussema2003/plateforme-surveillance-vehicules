@echo off
chcp 65001 >nul
color 0A

echo ═══════════════════════════════════════════════════════════════════════
echo   🚀 SCRIPT DE PUSH VERS GITHUB
echo ═══════════════════════════════════════════════════════════════════════
echo.

REM Demander le nom d'utilisateur GitHub
set /p USERNAME="Entrez votre nom d'utilisateur GitHub: "

echo.
echo ════════════════════════════════════════════════════════════════════════
echo   📋 COMMANDES QUI VONT ÊTRE EXÉCUTÉES :
echo ════════════════════════════════════════════════════════════════════════
echo.
echo   1. git remote add origin https://github.com/%USERNAME%/plateforme-surveillance-vehicules.git
echo   2. git branch -M main
echo   3. git push -u origin main
echo.
echo ════════════════════════════════════════════════════════════════════════
echo.

REM Demander confirmation
set /p CONFIRM="Voulez-vous continuer ? (O/N): "

if /i not "%CONFIRM%"=="O" (
    echo.
    echo ❌ Opération annulée.
    pause
    exit /b
)

echo.
echo ═══════════════════════════════════════════════════════════════════════
echo   🔗 Ajout du remote origin...
echo ═══════════════════════════════════════════════════════════════════════
git remote add origin https://github.com/%USERNAME%/plateforme-surveillance-vehicules.git

if errorlevel 1 (
    echo.
    echo ⚠️  Le remote existe déjà. Tentative de suppression et réajout...
    git remote remove origin
    git remote add origin https://github.com/%USERNAME%/plateforme-surveillance-vehicules.git
)

echo.
echo ═══════════════════════════════════════════════════════════════════════
echo   🌿 Renommage de la branche en 'main'...
echo ═══════════════════════════════════════════════════════════════════════
git branch -M main

echo.
echo ═══════════════════════════════════════════════════════════════════════
echo   ⬆️  Push vers GitHub...
echo ═══════════════════════════════════════════════════════════════════════
echo.
echo ⚠️  IMPORTANT : Lors de la demande d'authentification :
echo    - Username : %USERNAME%
echo    - Password : UTILISEZ VOTRE PERSONAL ACCESS TOKEN (pas votre mot de passe)
echo.
echo    Si vous n'avez pas de token, créez-en un sur :
echo    https://github.com/settings/tokens
echo.

git push -u origin main

if errorlevel 1 (
    echo.
    echo ═══════════════════════════════════════════════════════════════════════
    echo   ❌ ERREUR LORS DU PUSH
    echo ═══════════════════════════════════════════════════════════════════════
    echo.
    echo Vérifiez :
    echo   1. Que le dépôt existe sur GitHub
    echo   2. Que vous avez créé un Personal Access Token
    echo   3. Que le token a les permissions "repo"
    echo   4. Que vous utilisez le TOKEN et non votre mot de passe
    echo.
    echo Pour créer un token :
    echo   GitHub ^> Settings ^> Developer settings ^> Personal access tokens
    echo.
) else (
    echo.
    echo ═══════════════════════════════════════════════════════════════════════
    echo   ✅ SUCCÈS ! Votre projet est maintenant sur GitHub
    echo ═══════════════════════════════════════════════════════════════════════
    echo.
    echo 🌐 Visitez : https://github.com/%USERNAME%/plateforme-surveillance-vehicules
    echo.
    echo 📚 Votre README.md magnifique est maintenant visible par tous !
    echo.
)

echo.
pause
