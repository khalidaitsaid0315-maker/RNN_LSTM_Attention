"""
╔═══════════════════════════════════════════════════════════════════════════╗
║         Utilitaires pour Charger des Données Réelles (.npy)              ║
╚═══════════════════════════════════════════════════════════════════════════╝

Module helper pour charger et utiliser des données réelles dans l'interface.
Utilisez ce module si vous avez des fichiers .npy existants.

Exemple d'utilisation :
    from utils import load_npy_data
    X_train, y_train, X_test, y_test = load_npy_data('./data')
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional

def load_npy_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge les données d'entraînement et de test depuis des fichiers .npy.
    
    Fichiers attendus :
        - X_train.npy (samples, timesteps, features)
        - y_train.npy (samples,)
        - X_test.npy  (samples, timesteps, features)
        - y_test.npy  (samples,)
    
    Paramètres:
        data_dir : Répertoire contenant les fichiers .npy
    
    Retour:
        (X_train, y_train, X_test, y_test)
    
    Exemple:
        >>> X_train, y_train, X_test, y_test = load_npy_data('./data')
        >>> print(X_train.shape)
        (4714, 60, 18)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Répertoire '{data_dir}' non trouvé")
    
    required_files = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']
    missing_files = [f for f in required_files if not (data_path / f).exists()]
    
    if missing_files:
        raise FileNotFoundError(
            f"Fichiers manquants : {', '.join(missing_files)}\n"
            f"Cherchez dans : {data_path}"
        )
    
    print("📂 Chargement des données...")
    X_train = np.load(data_path / 'X_train.npy')
    y_train = np.load(data_path / 'y_train.npy')
    X_test = np.load(data_path / 'X_test.npy')
    y_test = np.load(data_path / 'y_test.npy')
    
    print(f"✅ Données chargées avec succès !")
    print(f"   X_train: {X_train.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_test:  {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def validate_data_shapes(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray
) -> bool:
    """
    Valide les shapes des données.
    
    Vérifie que :
        - X_train et X_test ont le même nombre de features et timesteps
        - y_train et y_test sont 1D
        - X et y ont le même nombre de samples respectifs
    """
    print("🔍 Validation des données...")
    
    # Vérifier que X sont 3D et y sont 1D
    assert X_train.ndim == 3, f"X_train doit être 3D, got {X_train.ndim}D"
    assert X_test.ndim == 3, f"X_test doit être 3D, got {X_test.ndim}D"
    assert y_train.ndim == 1, f"y_train doit être 1D, got {y_train.ndim}D"
    assert y_test.ndim == 1, f"y_test doit être 1D, got {y_test.ndim}D"
    
    # Vérifier les dimensions temporelles et features
    assert X_train.shape[1:] == X_test.shape[1:], \
        f"Dimensions features/timesteps incompatibles: {X_train.shape[1:]} vs {X_test.shape[1:]}"
    
    # Vérifier correspondance samples
    assert X_train.shape[0] == y_train.shape[0], \
        f"X_train et y_train n'ont pas le même nombre de samples"
    assert X_test.shape[0] == y_test.shape[0], \
        f"X_test et y_test n'ont pas le même nombre de samples"
    
    print("✅ Toutes les validations sont passées !")
    print(f"   Timesteps: {X_train.shape[1]}")
    print(f"   Features: {X_train.shape[2]}")
    print(f"   Train samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    return True


def get_data_statistics(X: np.ndarray, y: np.ndarray) -> dict:
    """Retourne les statistiques des données."""
    stats = {
        'X_min': X.min(),
        'X_max': X.max(),
        'X_mean': X.mean(),
        'X_std': X.std(),
        'y_min': y.min(),
        'y_max': y.max(),
        'y_mean': y.mean(),
        'y_std': y.std(),
        'samples': X.shape[0],
        'timesteps': X.shape[1],
        'features': X.shape[2],
    }
    return stats


def print_data_summary(X_train, y_train, X_test, y_test):
    """Affiche un résumé complet des données."""
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES DONNÉES")
    print("="*60)
    
    train_stats = get_data_statistics(X_train, y_train)
    test_stats = get_data_statistics(X_test, y_test)
    
    print("\n🏋️  DONNÉES D'ENTRAÎNEMENT")
    print(f"  Samples:    {train_stats['samples']}")
    print(f"  Timesteps:  {train_stats['timesteps']}")
    print(f"  Features:   {train_stats['features']}")
    print(f"  X - Range:  [{train_stats['X_min']:.4f}, {train_stats['X_max']:.4f}]")
    print(f"  X - Mean:   {train_stats['X_mean']:.4f} ± {train_stats['X_std']:.4f}")
    print(f"  y - Range:  [{train_stats['y_min']:.4f}, {train_stats['y_max']:.4f}]")
    print(f"  y - Mean:   {train_stats['y_mean']:.4f} ± {train_stats['y_std']:.4f}")
    
    print("\n🧪 DONNÉES DE TEST")
    print(f"  Samples:    {test_stats['samples']}")
    print(f"  Timesteps:  {test_stats['timesteps']}")
    print(f"  Features:   {test_stats['features']}")
    print(f"  X - Range:  [{test_stats['X_min']:.4f}, {test_stats['X_max']:.4f}]")
    print(f"  X - Mean:   {test_stats['X_mean']:.4f} ± {test_stats['X_std']:.4f}")
    print(f"  y - Range:  [{test_stats['y_min']:.4f}, {test_stats['y_max']:.4f}]")
    print(f"  y - Mean:   {test_stats['y_mean']:.4f} ± {test_stats['y_std']:.4f}")
    
    print("\n📈 STATISTIQUES GLOBALES")
    total_samples = train_stats['samples'] + test_stats['samples']
    train_ratio = train_stats['samples'] / total_samples * 100
    test_ratio = test_stats['samples'] / total_samples * 100
    print(f"  Total samples:     {total_samples}")
    print(f"  Train / Test ratio: {train_ratio:.1f}% / {test_ratio:.1f}%")
    print(f"  Data shape:        ({total_samples}, {train_stats['timesteps']}, {train_stats['features']})")
    
    print("="*60 + "\n")


# Exemple d'utilisation
if __name__ == "__main__":
    # Essayez de charger les données du répertoire courant
    try:
        data_dir = "./data"  # À adapter selon votre structure
        X_train, y_train, X_test, y_test = load_npy_data(data_dir)
        
        # Valider
        validate_data_shapes(X_train, y_train, X_test, y_test)
        
        # Afficher résumé
        print_data_summary(X_train, y_train, X_test, y_test)
        
    except FileNotFoundError as e:
        print(f"❌ Erreur : {e}")
        print("\n💡 Utilisation alternative :")
        print("   L'application génère automatiquement des données synthétiques")
        print("   Placez vos fichiers .npy dans un dossier 'data/' pour les utiliser")
