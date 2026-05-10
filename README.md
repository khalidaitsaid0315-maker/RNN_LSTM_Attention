# 📈 RNN/LSTM Interface — Prédiction de Séries Financières

Interface web interactive pour comparer et tester des modèles de prédiction de séries temporelles financières.

## 🎯 Description

Cette application permet de :
- **Comparer** 4 architectures de modèles : Ridge, MLP-RNN, MLP-LSTM, et RNN avec Attention
- **Configurer** facilement les hyperparamètres
- **Visualiser** les prédictions, erreurs et poids d'attention
- **Analyser** les performances avec des métriques détaillées

## 🏗️ Architecture des Modèles

### 1️⃣ Ridge Regression (Baseline)
- Modèle linéaire classique
- **Avantage** : Rapide, stable, bon baseline
- **Limitation** : Capture uniquement les tendances linéaires

### 2️⃣ MLP-RNN (tanh activation)
- Réseau multicouche 128×64 avec activation tanh
- **Simule** un RNN simplifié sans récurrence temporelle
- Couches : 1080 inputs → 128 → 64 → 1 output

### 3️⃣ MLP-LSTM (relu activation)  
- Réseau plus profond : 256×128×64 avec activation relu
- **Plus de capacité** pour apprendre des dépendances complexes
- Couches : 1080 inputs → 256 → 128 → 64 → 1 output

### 4️⃣ RNN + Attention (Manuel)
- Mécanisme d'attention additive avec softmax
- **Pondère** chaque pas de temps selon son importance
- Interprétabilité : visualisation des poids d'attention

## 📊 Métriques Calculées

- **MSE** (Mean Squared Error) : Erreur quadratique moyenne
- **RMSE** (Root Mean Squared Error) : Racine de MSE
- **MAE** (Mean Absolute Error) : Erreur absolue moyenne

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip ou conda

### Étapes

```bash
# 1. Cloner ou télécharger le projet
cd app_RNN_LSTM

# 2. Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
# Activer : 
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

## ▶️ Lancer l'Application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à : `http://localhost:8501`

## 🎮 Utilisation

### 1. Configuration (Barre latérale)

**Données :**
- Source des données : synthétiques ou fichiers `.npy`
- Nombre d'échantillons (100-1000)
- Longueur de séquence (10-100)
- Nombre de features (5-50)
- Taille du test (10-40%)

Pour utiliser des données réelles, choisissez **Fichiers .npy** dans la barre latérale et indiquez le dossier contenant :
- `X_train.npy` au format `(samples, timesteps, features)`
- `y_train.npy` au format `(samples,)`
- `X_test.npy` au format `(samples, timesteps, features)`
- `y_test.npy` au format `(samples,)`

**Hyperparamètres :**
- Ridge α (0.1-10.0)
- Couches cachées MLP
- Dimension d'attention

### 2. Exécution

Cliquez sur **"🚀 Exécuter l'analyse"** pour :
- Générer les données
- Entraîner tous les modèles
- Calculer les métriques
- Afficher les visualisations

### 3. Interprétation des Résultats

**Tableau de métriques** : Compare MSE, RMSE, MAE pour chaque modèle
- Les meilleures valeurs sont marquées avec un bord rouge
- Les métriques peuvent être téléchargées en CSV

**Graphiques :**
- Barplot : Comparaison des performances
- Time series : Prédictions vs valeurs réelles
- Histogrammes : Distribution des erreurs
- Scatter : Réel vs Prédit (alignement sur diagonale = bon modèle)

## 📈 Visualisations

### 1. Comparaison des Modèles
Trois graphiques montrant MSE, RMSE, MAE
- Permet de voir rapidement le meilleur modèle

### 2. Prédictions vs Réalité
4 graphiques (un par modèle) sur les 100 premiers points du test
- Noir = valeurs réelles
- Couleur = prédictions du modèle

### 3. Distribution des Résidus
Histogrammes des erreurs (Prédictions - Réalité)
- Vérifie que la distribution est centrée en 0
- Affiche moyenne (μ) et écart-type (σ)

### 4. Scatter Réel vs Prédit
- Points sur la diagonale rouge = prédictions parfaites
- Points loin = zones où le modèle a des difficultés

## 💡 Conseils d'Utilisation

✅ **À faire :**
- Tester avec différentes longueurs de séquence
- Augmenter la dimension d'attention pour plus d'expressivité
- Comparer Ridge (baseline) avec les autres modèles
- Observer comment les poids d'attention changent

❌ **À éviter :**
- Trop de features sans assez d'échantillons
- Couches trop profondes avec peu de données
- Ignorer les résidus (indicateur de surapprentissage)

## 📚 Théorie

### Séquences Aplatties (Flattened)
- Format original : (samples, 60, 18) = 60 pas de temps × 18 features
- Après aplatissement : (samples, 1080) = 60 × 18 concatenés

### Attention Mechanism
```
Pour chaque séquence :
  scores = tanh(X @ Wa) @ v        // (timesteps,)
  weights = softmax(scores)         // (timesteps,) - somme à 1
  context = Σ(weights[t] * X[t])   // (features,) - moyenne pondérée
  prediction = Ridge(context)
```

### Pourquoi l'Attention ?
- **Interprétabilité** : Voir quels pas de temps importent
- **Performance** : Pondération adaptative > moyenne simple
- **Robustesse** : Résilience aux données bruitées

## 🔄 Workflow Typique

1. **Configuration initiale**
   - Nombre d'échantillons : 500
   - Longueur séquence : 60
   - Features : 18

2. **Première exécution**
   - Observe les performances baselines
   - Note le meilleur modèle

3. **Expérimentation**
   - Augmente la dimension d'attention
   - Ajuste les couches MLP
   - Compare les résultats

4. **Analyse approfondie**
   - Examine les distributions d'erreurs
   - Vérifie les scatter plots
   - Identifie les points difficiles

## 🐛 Dépannage

**L'app ne démarre pas :**
```bash
# Vérifiez Streamlit
pip install --upgrade streamlit

# Essayez avec le port explicite
streamlit run app.py --server.port 8501
```

**Erreur de mémoire :**
- Réduisez le nombre d'échantillons
- Diminuez la dimension d'attention

**Modèles qui ne convergent pas :**
- Vérifiez que `early_stopping=True` fonctionne
- Ajustez `max_iter`

## 📝 Exemple de Sortie

```
📊 Résultats :

Ridge (Baseline)
  MSE:  0.000378
  RMSE: 0.019444
  MAE:  0.015278

MLP-RNN (tanh)
  MSE:  0.009348
  RMSE: 0.096685
  MAE:  0.078690

MLP-LSTM (relu)
  MSE:  0.001357
  RMSE: 0.036836
  MAE:  0.028652

RNN+Attention (Manuel)
  MSE:  0.002490 ✅
  RMSE: 0.049903
  MAE:  0.039161
```

## 🔗 Fichiers du Projet

```
app_RNN_LSTM/
├── app.py                 # Application Streamlit principale
├── requirements.txt       # Dépendances Python
├── README.md             # Ce fichier
├── launch.bat            # Script de lancement (Windows)
└── launch.sh             # Script de lancement (macOS/Linux)
```

## 📚 Ressources Complémentaires

- [Streamlit Documentation](https://docs.streamlit.io)
- [scikit-learn Neural Networks](https://scikit-learn.org/stable/modules/neural_networks.html)
- [Attention Mechanisms](https://arxiv.org/abs/1706.03762)

## 📄 License

Projet éducatif - Libre d'usage

## 👤 Auteur

Interface créée pour la comparaison de modèles RNN/LSTM

---

**Questions ou améliorations ?** N'hésitez pas à modifier et personnaliser !
"# RNN_LSTM_Attention" 
