# =============================================================================
# SCRIPT PRINCIPAL POUR LA SEGMENTATION CÉRÉBRALE 3D
# Exécute le pipeline complet: prétraitement -> augmentation -> entraînement
# =============================================================================

import os
import sys
import argparse
from pathlib import Path

# Ajouter le répertoire courant au path pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from preprocessing import main as preprocess_main
from augmentation import main as augment_main
from training import main as train_main, load_trained_model, continue_training

def run_preprocessing():
    """Exécute le prétraitement des données"""
    print("=" * 60)
    print("ÉTAPE 1: PRÉTRAITEMENT DES DONNÉES")
    print("=" * 60)
    
    try:
        splits = preprocess_main()
        if splits is not None:
            print("✅ Prétraitement terminé avec succès")
            return True
        else:
            print("❌ Échec du prétraitement")
            return False
    except Exception as e:
        print(f"❌ Erreur lors du prétraitement: {e}")
        return False

def run_augmentation():
    """Exécute l'augmentation des données"""
    print("=" * 60)
    print("ÉTAPE 2: AUGMENTATION DES DONNÉES")
    print("=" * 60)
    
    try:
        augmented_splits = augment_main()
        if augmented_splits is not None:
            print("✅ Augmentation terminée avec succès")
            return True
        else:
            print("❌ Échec de l'augmentation")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de l'augmentation: {e}")
        return False

def run_training():
    """Exécute l'entraînement du modèle"""
    print("=" * 60)
    print("ÉTAPE 3: ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60)
    
    try:
        model, history = train_main()
        if model is not None:
            print("✅ Entraînement terminé avec succès")
            return True
        else:
            print("❌ Échec de l'entraînement")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        return False

def run_full_pipeline():
    """Exécute le pipeline complet"""
    print("=" * 80)
    print("DÉMARRAGE DU PIPELINE COMPLET DE SEGMENTATION CÉRÉBRALE 3D")
    print("=" * 80)
    
    # Afficher la configuration
    config.print_config()
    
    # Étape 1: Prétraitement
    if not run_preprocessing():
        print("❌ Arrêt du pipeline: échec du prétraitement")
        return False
    
    # Étape 2: Augmentation
    if not run_augmentation():
        print("❌ Arrêt du pipeline: échec de l'augmentation")
        return False
    
    # Étape 3: Entraînement
    if not run_training():
        print("❌ Arrêt du pipeline: échec de l'entraînement")
        return False
    
    print("=" * 80)
    print("✅ PIPELINE COMPLET TERMINÉ AVEC SUCCÈS!")
    print("=" * 80)
    return True

def run_specific_step(step):
    """Exécute une étape spécifique"""
    if step == "preprocessing":
        return run_preprocessing()
    elif step == "augmentation":
        return run_augmentation()
    elif step == "training":
        return run_training()
    else:
        print(f"❌ Étape inconnue: {step}")
        print("Étapes disponibles: preprocessing, augmentation, training")
        return False

def main():
    """Fonction principale avec gestion des arguments"""
    parser = argparse.ArgumentParser(description="Pipeline de segmentation cérébrale 3D")
    parser.add_argument(
        "--step", 
        choices=["preprocessing", "augmentation", "training", "all"],
        default="all",
        help="Étape à exécuter (défaut: all)"
    )
    parser.add_argument(
        "--continue-training",
        type=str,
        help="Chemin vers un modèle existant pour continuer l'entraînement"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Nombre d'époques supplémentaires pour la continuation d'entraînement"
    )
    
    args = parser.parse_args()
    
    # Vérifier que les dossiers de sortie existent
    for path in [config.OUTPUT_PATH, config.PREPROCESSED_PATH, 
                config.AUGMENTED_PATH, config.MODEL_PATH]:
        os.makedirs(path, exist_ok=True)
    
    if args.continue_training:
        # Mode continuation d'entraînement
        print("=" * 60)
        print("CONTINUATION DE L'ENTRAÎNEMENT")
        print("=" * 60)
        
        additional_epochs = args.epochs if args.epochs else 10
        model, history = continue_training(args.continue_training, additional_epochs)
        
        if model is not None:
            print("✅ Continuation d'entraînement terminée avec succès")
        else:
            print("❌ Échec de la continuation d'entraînement")
    else:
        # Mode normal
        if args.step == "all":
            success = run_full_pipeline()
        else:
            success = run_specific_step(args.step)
        
        if success:
            print("\n🎉 Opération terminée avec succès!")
        else:
            print("\n💥 Opération échouée!")
            sys.exit(1)

if __name__ == "__main__":
    main()