# =============================================================================
# EXEMPLE D'UTILISATION DU PIPELINE DE SEGMENTATION CÉRÉBRALE 3D
# =============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from preprocessing import load_preprocessed_data, debug_dataset_structure
from augmentation import load_augmented_data, visualize_augmentations
from training import load_trained_model, evaluate_model
from utils import plot_training_history, visualize_predictions, print_metrics

def example_preprocessing():
    """Exemple d'utilisation du prétraitement"""
    print("=== EXEMPLE: PRÉTRAITEMENT ===")
    
    # Vérifier la structure du dataset
    debug_dataset_structure()
    
    # Charger les données prétraitées (si elles existent)
    splits = load_preprocessed_data()
    
    if any(len(split_data['X']) > 0 for split_data in splits.values()):
        print("✅ Données prétraitées trouvées")
        
        # Afficher un résumé
        for split_name, split_data in splits.items():
            if len(split_data['X']) > 0:
                print(f"{split_name.upper()}:")
                print(f"  Images: {split_data['X'].shape}")
                print(f"  Masks: {split_data['y'].shape}")
                print(f"  Classes uniques: {np.unique(split_data['y'])}")
    else:
        print("❌ Aucune donnée prétraitée trouvée")
        print("Exécutez d'abord: python main.py --step preprocessing")

def example_augmentation():
    """Exemple d'utilisation de l'augmentation"""
    print("\n=== EXEMPLE: AUGMENTATION ===")
    
    # Charger les données augmentées
    splits = load_augmented_data()
    
    if any(len(split_data['X']) > 0 for split_data in splits.values()):
        print("✅ Données augmentées trouvées")
        
        # Visualiser quelques exemples d'augmentation
        if len(splits['train']['X']) > 0:
            print("Visualisation des augmentations...")
            
            original_img = splits['train']['X'][0]
            original_mask = splits['train']['y'][0]
            
            # Prendre quelques augmentations
            aug_imgs = splits['train']['X'][1:5]
            aug_masks = splits['train']['y'][1:5]
            
            visualize_augmentations(
                original_img, original_mask, aug_imgs, aug_masks, 
                config, n_samples=4
            )
    else:
        print("❌ Aucune donnée augmentée trouvée")
        print("Exécutez d'abord: python main.py --step augmentation")

def example_training():
    """Exemple d'utilisation de l'entraînement"""
    print("\n=== EXEMPLE: ENTRAÎNEMENT ===")
    
    # Vérifier si un modèle existe
    model_path = os.path.join(config.MODEL_PATH, 'best_model.keras')
    
    if os.path.exists(model_path):
        print("✅ Modèle entraîné trouvé")
        
        # Charger le modèle
        model = load_trained_model(model_path, config)
        
        if model is not None:
            print("Modèle chargé avec succès")
            print(f"Architecture: {model.input_shape} -> {model.output_shape}")
            
            # Charger les données de test
            splits = load_augmented_data()
            
            if len(splits['test']['X']) > 0:
                print("Évaluation sur les données de test...")
                
                # Évaluer le modèle
                metrics, predictions, pred_classes = evaluate_model(
                    model, splits['test']['X'], splits['test']['y'], config
                )
                
                if metrics is not None:
                    print_metrics(metrics)
                    
                    # Visualiser quelques prédictions
                    print("Visualisation des prédictions...")
                    visualize_predictions(
                        model, splits['test']['X'][:3], splits['test']['y'][:3], 
                        config, n_samples=3
                    )
            else:
                print("❌ Aucune donnée de test disponible")
        else:
            print("❌ Erreur lors du chargement du modèle")
    else:
        print("❌ Aucun modèle entraîné trouvé")
        print("Exécutez d'abord: python main.py --step training")

def example_custom_pipeline():
    """Exemple de pipeline personnalisé"""
    print("\n=== EXEMPLE: PIPELINE PERSONNALISÉ ===")
    
    # Configuration personnalisée
    print("Configuration actuelle:")
    config.print_config()
    
    # Modifier quelques paramètres pour l'exemple
    original_batch_size = config.BATCH_SIZE
    config.BATCH_SIZE = 1  # Plus petit pour l'exemple
    
    print(f"\nBatch size modifié: {original_batch_size} -> {config.BATCH_SIZE}")
    
    # Restaurer la configuration originale
    config.BATCH_SIZE = original_batch_size
    
    print("Configuration restaurée")

def main():
    """Fonction principale d'exemple"""
    print("=" * 80)
    print("EXEMPLES D'UTILISATION DU PIPELINE DE SEGMENTATION CÉRÉBRALE 3D")
    print("=" * 80)
    
    # Vérifier que les dossiers existent
    for path in [config.OUTPUT_PATH, config.PREPROCESSED_PATH, 
                config.AUGMENTED_PATH, config.MODEL_PATH]:
        os.makedirs(path, exist_ok=True)
    
    # Exécuter les exemples
    example_preprocessing()
    example_augmentation()
    example_training()
    example_custom_pipeline()
    
    print("\n" + "=" * 80)
    print("EXEMPLES TERMINÉS")
    print("=" * 80)
    
    print("\nPour exécuter le pipeline complet:")
    print("python main.py")
    
    print("\nPour exécuter une étape spécifique:")
    print("python main.py --step preprocessing")
    print("python main.py --step augmentation")
    print("python main.py --step training")
    
    print("\nPour continuer l'entraînement:")
    print("python main.py --continue-training ./models/best_model.keras --epochs 10")

if __name__ == "__main__":
    main()