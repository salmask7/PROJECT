# =============================================================================
# UTILITAIRES POUR LA SEGMENTATION CÉRÉBRALE 3D
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras

# Medical Imaging
import nibabel as nib
from scipy import ndimage
from skimage import measure, morphology
from sklearn.model_selection import train_test_split

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Coefficient de Dice pour l'évaluation"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Loss basée sur le coefficient de Dice"""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Loss combinée: Cross-entropy + Dice"""
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice_loss_val = dice_loss(y_true, y_pred)
    return ce_loss + dice_loss_val

def plot_training_history(history, save_path=None):
    """Visualise l'historique d'entraînement"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Dice Coefficient
    axes[2].plot(history.history['dice_coefficient'], label='Train Dice')
    axes[2].plot(history.history['val_dice_coefficient'], label='Val Dice')
    axes[2].set_title('Dice Coefficient')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Dice Score')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {save_path}")
    
    plt.show()

def visualize_predictions(model, X_val, y_val, config, n_samples=3, save_path=None):
    """Visualise les prédictions du modèle"""
    predictions = model.predict(X_val[:n_samples][..., np.newaxis])
    pred_classes = np.argmax(predictions, axis=-1)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    for i in range(n_samples):
        # Image originale
        axes[i, 0].imshow(X_val[i][:, :, 64], cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Ground Truth
        im1 = axes[i, 1].imshow(y_val[i][:, :, 64], cmap='tab10', vmin=0, vmax=3)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Prédiction
        im2 = axes[i, 2].imshow(pred_classes[i][:, :, 64], cmap='tab10', vmin=0, vmax=3)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

        # Différence
        diff = np.abs(y_val[i][:, :, 64] - pred_classes[i][:, :, 64])
        im3 = axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title('Difference')
        axes[i, 3].axis('off')

    # Ajouter une colorbar pour les masks
    plt.colorbar(im1, ax=axes[:, 1:3].ravel().tolist(), 
                ticks=[0, 1, 2, 3], label='Classes')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualisation sauvegardée: {save_path}")
    
    plt.show()

def calculate_metrics(y_true, y_pred, config):
    """Calcule les métriques de segmentation"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Convertir en format 1D pour les métriques
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Métriques globales
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    
    # Métriques par classe
    class_metrics = {}
    for i, class_name in enumerate(config.CLASS_NAMES):
        if i in y_true_flat:  # Vérifier que la classe existe
            class_mask_true = (y_true_flat == i)
            class_mask_pred = (y_pred_flat == i)
            
            if np.sum(class_mask_true) > 0:  # Éviter la division par zéro
                precision_class = precision_score(class_mask_true, class_mask_pred, zero_division=0)
                recall_class = recall_score(class_mask_true, class_mask_pred, zero_division=0)
                f1_class = f1_score(class_mask_true, class_mask_pred, zero_division=0)
                
                class_metrics[class_name] = {
                    'precision': precision_class,
                    'recall': recall_class,
                    'f1': f1_class
                }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics
    }

def print_metrics(metrics):
    """Affiche les métriques de manière formatée"""
    print("\n=== MÉTRIQUES DE SEGMENTATION ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    print("\n=== MÉTRIQUES PAR CLASSE ===")
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"{class_name}:")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  Recall: {class_metrics['recall']:.4f}")
        print(f"  F1-Score: {class_metrics['f1']:.4f}")
    print("================================")

def save_predictions(predictions, y_true, config, save_path):
    """Sauvegarde les prédictions et métriques"""
    import json
    
    # Calculer les métriques
    pred_classes = np.argmax(predictions, axis=-1)
    metrics = calculate_metrics(y_true, pred_classes, config)
    
    # Sauvegarder les métriques
    metrics_path = save_path.replace('.npy', '_metrics.json')
    with open(metrics_path, 'w') as f:
        # Convertir les numpy types en types Python pour JSON
        json_metrics = {}
        for key, value in metrics.items():
            if key == 'class_metrics':
                json_metrics[key] = {k: {kk: float(vv) for kk, vv in v.items()} 
                                   for k, v in value.items()}
            else:
                json_metrics[key] = float(value)
        json.dump(json_metrics, f, indent=2)
    
    # Sauvegarder les prédictions
    np.save(save_path, predictions)
    
    print(f"Prédictions sauvegardées: {save_path}")
    print(f"Métriques sauvegardées: {metrics_path}")
    
    return metrics