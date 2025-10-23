# =============================================================================
# BRAIN ANATOMICAL SEGMENTATION - VERSION ROBUSTE
# Dataset: 3D Brain Tissue Segmentation (726 sujets)
# Modèle: U-Net 3D pour segmentation CSF, Gray Matter, White Matter
# =============================================================================

# =============================================================================
# 1. SETUP ET IMPORTS
# =============================================================================

!pip install nibabel
!pip install tensorflow-addons
!pip install scikit-image

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import glob
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

# Medical Imaging
import nibabel as nib
from scipy import ndimage
from skimage import measure, morphology
from sklearn.model_selection import train_test_split

# =============================================================================
# 2. CONFIGURATION
# =============================================================================

class Config:
    def __init__(self):
        # Dataset paths
        self.DATA_PATH = '/kaggle/input/3dbraintissuesegmentation'
        
        # Model parameters
        self.IMG_SIZE = (128, 128, 128)
        self.BATCH_SIZE = 2
        self.EPOCHS = 30  # Réduit pour test
        self.LEARNING_RATE = 1e-4
        
        # Classes
        self.N_CLASSES = 4
        self.CLASS_NAMES = ['Background', 'CSF', 'Gray Matter', 'White Matter']
        
        # Debug
        self.DEBUG = True

config = Config()

# =============================================================================
# 3. FONCTIONS DE CHARGEMENT ROBUSTES
# =============================================================================

def find_all_data_files():
    """Trouve tous les fichiers de données de manière robuste"""
    print("=== RECHERCHE DE TOUS LES FICHIERS ===")
    
    # Chercher tous les fichiers .nii.gz
    all_files = glob.glob(os.path.join(config.DATA_PATH, "**/*.nii.gz"), recursive=True)
    print(f"Total fichiers .nii.gz trouvés: {len(all_files)}")
    
    if len(all_files) == 0:
        print("❌ Aucun fichier .nii.gz trouvé!")
        return [], []
    
    # Séparer images et masks
    img_files = [f for f in all_files if "_img.nii.gz" in f]
    csf_files = [f for f in all_files if "probmask_csf" in f]
    gm_files = [f for f in all_files if "probmask_graymatter" in f]
    wm_files = [f for f in all_files if "probmask_whitematter" in f]
    
    print(f"Images: {len(img_files)}")
    print(f"CSF masks: {len(csf_files)}")
    print(f"GM masks: {len(gm_files)}")
    print(f"WM masks: {len(wm_files)}")
    
    # Trouver les correspondances
    matching_pairs = []
    
    for img_file in img_files:
        # Extraire le nom de base
        base_name = os.path.basename(img_file).replace("_img.nii.gz", "")
        
        # Chercher les masks correspondants
        csf_match = None
        gm_match = None
        wm_match = None
        
        for csf_file in csf_files:
            if base_name in csf_file:
                csf_match = csf_file
                break
        
        for gm_file in gm_files:
            if base_name in gm_file:
                gm_match = gm_file
                break
        
        for wm_file in wm_files:
            if base_name in wm_file:
                wm_match = wm_file
                break
        
        # Si tous les masks sont trouvés
        if all([csf_match, gm_match, wm_match]):
            matching_pairs.append({
                'image': img_file,
                'csf': csf_match,
                'gm': gm_match,
                'wm': wm_match,
                'base_name': base_name
            })
    
    print(f"✅ Paires complètes trouvées: {len(matching_pairs)}")
    
    if len(matching_pairs) > 0:
        print("Exemples de paires:")
        for i, pair in enumerate(matching_pairs[:3]):
            print(f"  {i+1}. {pair['base_name']}")
    
    return matching_pairs, all_files

def load_nifti_file(filepath):
    """Charge un fichier NIFTI de manière robuste"""
    try:
        nifti = nib.load(filepath)
        data = nifti.get_fdata()
        return data.astype(np.float32)
    except Exception as e:
        print(f"Erreur chargement {os.path.basename(filepath)}: {e}")
        return None

def normalize_image(image):
    """Normalise l'image entre 0 et 1"""
    if image is None:
        return None
    
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image)
    
    return image

def resize_volume(volume, target_shape):
    """Redimensionne le volume 3D"""
    if volume is None:
        return None
    
    current_shape = volume.shape
    factors = [target_shape[i] / current_shape[i] for i in range(3)]
    resized = ndimage.zoom(volume, factors, order=1)
    return resized

def create_multiclass_mask(csf_mask, gm_mask, wm_mask):
    """Combine les 3 masks en un mask multiclasse"""
    if any(mask is None for mask in [csf_mask, gm_mask, wm_mask]):
        return None
    
    # Convertir les probabilités en classes hard
    csf_hard = (csf_mask > 0.5).astype(np.uint8)
    gm_hard = (gm_mask > 0.5).astype(np.uint8)
    wm_hard = (wm_mask > 0.5).astype(np.uint8)
    
    # Créer le mask multiclasse
    multiclass_mask = np.zeros_like(csf_hard, dtype=np.uint8)
    multiclass_mask[wm_hard == 1] = 3  # White Matter
    multiclass_mask[gm_hard == 1] = 2  # Gray Matter  
    multiclass_mask[csf_hard == 1] = 1  # CSF
    
    return multiclass_mask

def load_dataset_robust(max_samples=None):
    """Charge le dataset de manière robuste"""
    print("=== CHARGEMENT ROBUSTE DU DATASET ===")
    
    # Trouver tous les fichiers
    matching_pairs, all_files = find_all_data_files()
    
    if len(matching_pairs) == 0:
        print("❌ Aucune paire image-mask trouvée!")
        return np.array([]), np.array([])
    
    # Limiter le nombre d'échantillons
    if max_samples:
        matching_pairs = matching_pairs[:max_samples]
    
    print(f"Chargement de {len(matching_pairs)} échantillons...")
    
    images = []
    masks = []
    
    for i, pair in enumerate(matching_pairs):
        if i % 5 == 0:
            print(f"Progression: {i+1}/{len(matching_pairs)}")
        
        try:
            # Charger l'image
            image = load_nifti_file(pair['image'])
            if image is None:
                continue
            
            # Charger les masks
            csf_mask = load_nifti_file(pair['csf'])
            gm_mask = load_nifti_file(pair['gm'])
            wm_mask = load_nifti_file(pair['wm'])
            
            if any(mask is None for mask in [csf_mask, gm_mask, wm_mask]):
                continue
            
            # Créer le mask multiclasse
            multiclass_mask = create_multiclass_mask(csf_mask, gm_mask, wm_mask)
            if multiclass_mask is None:
                continue
            
            # Redimensionner
            image_resized = resize_volume(image, config.IMG_SIZE)
            mask_resized = resize_volume(multiclass_mask, config.IMG_SIZE)
            
            if image_resized is None or mask_resized is None:
                continue
            
            # Normaliser l'image
            image_normalized = normalize_image(image_resized)
            if image_normalized is None:
                continue
            
            images.append(image_normalized)
            masks.append(mask_resized)
            
        except Exception as e:
            print(f"Erreur avec {pair['base_name']}: {e}")
            continue
    
    print(f"✅ Chargé {len(images)} échantillons avec succès")
    
    if len(images) == 0:
        print("❌ Aucun échantillon chargé!")
        return np.array([]), np.array([])
    
    return np.array(images), np.array(masks)

# =============================================================================
# 4. MODÈLE U-NET 3D SIMPLIFIÉ
# =============================================================================

def simple_unet_3d(input_shape, n_classes):
    """U-Net 3D simplifié pour test"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv3D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling3D(2)(x)
    
    x = layers.Conv3D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling3D(2)(x)
    
    x = layers.Conv3D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(128, 3, padding='same', activation='relu')(x)
    
    # Decoder
    x = layers.UpSampling3D(2)(x)
    x = layers.Conv3D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(64, 3, padding='same', activation='relu')(x)
    
    x = layers.UpSampling3D(2)(x)
    x = layers.Conv3D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(32, 3, padding='same', activation='relu')(x)
    
    # Output
    outputs = layers.Conv3D(n_classes, 1, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# =============================================================================
# 5. MÉTRIQUES
# =============================================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Coefficient de Dice"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# =============================================================================
# 6. ENTRÂINEMENT PRINCIPAL
# =============================================================================

def train_model_robust():
    """Entraînement robuste"""
    print("=== DÉMARRAGE DE L'ENTRÂINEMENT ROBUSTE ===")
    
    # Charger les données
    X_data, y_data = load_dataset_robust(max_samples=20)  # Petit échantillon pour test
    
    if len(X_data) == 0:
        print("❌ Impossible de charger les données!")
        return None, None
    
    print(f"✅ Données chargées: {X_data.shape}, {y_data.shape}")
    
    # Diviser en train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # One-hot encoding
    y_train_oh = tf.keras.utils.to_categorical(y_train, config.N_CLASSES)
    y_val_oh = tf.keras.utils.to_categorical(y_val, config.N_CLASSES)
    
    # Créer le modèle
    print("=== CRÉATION DU MODÈLE ===")
    model = simple_unet_3d(input_shape=config.IMG_SIZE + (1,), n_classes=config.N_CLASSES)
    
    # Compiler
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', dice_coefficient]
    )
    
    print("=== DÉMARRAGE DE L'ENTRÂINEMENT ===")
    history = model.fit(
        X_train[..., np.newaxis], y_train_oh,
        validation_data=(X_val[..., np.newaxis], y_val_oh),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        verbose=1
    )
    
    return model, history

# =============================================================================
# 7. EXÉCUTION
# =============================================================================

if __name__ == "__main__":
    print("=== DÉMARRAGE ===")
    
    # Vérifier GPU
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")
    
    # Entraîner
    model, history = train_model_robust()
    
    if model is not None:
        print("=== SUCCÈS ===")
        model.save('brain_segmentation_model.h5')
        print("Modèle sauvegardé!")
    else:
        print("=== ÉCHEC ===")