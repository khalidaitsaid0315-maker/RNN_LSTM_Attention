@echo off
REM ╔═════════════════════════════════════════════════════════════╗
REM ║  RNN/LSTM Interface — Script de Lancement (Windows)         ║
REM ╚═════════════════════════════════════════════════════════════╝

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║          🚀 RNN/LSTM INTERFACE - LANCEMENT              ║
echo  ║          Prédiction de Séries Financières              ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo  ❌ ERREUR : Python n'est pas installé ou non accessible
    echo  📥 Téléchargez Python depuis https://www.python.org
    pause
    exit /b 1
)

echo  ✅ Python détecté
echo.

REM Vérifier si les dépendances sont installées
echo  📦 Vérification des dépendances...
python -c "import streamlit; import numpy" >nul 2>&1
if errorlevel 1 (
    echo  ⚠️  Dépendances manquantes, installation...
    echo.
    pip install streamlit numpy
    if errorlevel 1 (
        echo  ❌ ERREUR lors de l'installation des dépendances
        pause
        exit /b 1
    )
    echo.
)

echo  ✅ Toutes les dépendances sont installées
echo.

REM Lancer l'application
echo  🎯 Démarrage de l'application...
echo  💻 L'application s'ouvrira dans votre navigateur
echo  📍 URL: http://localhost:8501
echo  ⏹️  Appuyez sur CTRL+C pour arrêter l'application
echo.

streamlit run app.py

pause
