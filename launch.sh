#!/bin/bash

# ╔═════════════════════════════════════════════════════════════╗
# ║  RNN/LSTM Interface — Script de Lancement (macOS/Linux)    ║
# ╚═════════════════════════════════════════════════════════════╝

echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║                                                          ║"
echo "  ║          🚀 RNN/LSTM INTERFACE - LANCEMENT              ║"
echo "  ║          Prédiction de Séries Financières              ║"
echo "  ║                                                          ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "  ❌ ERREUR : Python 3 n'est pas installé"
    echo "  📥 Installez Python 3 via :"
    echo "     - Homebrew (macOS) : brew install python3"
    echo "     - APT (Ubuntu/Debian) : sudo apt-get install python3"
    echo "     - Ou téléchargez depuis https://www.python.org"
    exit 1
fi

echo "  ✅ Python 3 détecté : $(python3 --version)"
echo ""

# Vérifier si les dépendances sont installées
echo "  📦 Vérification des dépendances..."
python3 -c "import streamlit; import numpy; import sklearn; import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  ⚠️  Dépendances manquantes, installation..."
    echo ""
    
    # Créer environnement virtuel si nécessaire
    if [ ! -d "venv" ]; then
        echo "  🔧 Création d'un environnement virtuel..."
        python3 -m venv venv
        source venv/bin/activate
    else
        source venv/bin/activate
    fi
    
    pip install --upgrade pip
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "  ❌ ERREUR lors de l'installation des dépendances"
        exit 1
    fi
    echo ""
fi

echo "  ✅ Toutes les dépendances sont installées"
echo ""

# Lancer l'application
echo "  🎯 Démarrage de l'application..."
echo "  💻 L'application s'ouvrira dans votre navigateur"
echo "  📍 URL: http://localhost:8501"
echo "  ⏹️  Appuyez sur CTRL+C pour arrêter l'application"
echo ""

streamlit run app.py

