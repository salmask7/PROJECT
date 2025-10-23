# =============================================================================
# BRAIN ANATOMICAL SEGMENTATION - VERSION CORRIGÉE POUR KAGGLE
# Dataset: 3D Brain Tissue Segmentation (726 sujets)
# Modèle: U-Net 3D pour segmentation CSF, Gray Matter, White Matter
# =============================================================================

# =============================================================================
# 1. SETUP ET IMPORTS
# =============================================================================

# Installation des packages nécessaires
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
# 2. CONFIGURATION DYNAMIQUE
# =============================================================================

class Config:
    def __init__(self):
        # Dataset paths - Détection automatique
        self.DATA_PATH = '/kaggle/input/3dbraintissuesegmentation'
        
        # Détecter automatiquement la structure
        self._detect_structure()
        
        # Model parameters
        self.IMG_SIZE = (128, 128, 128)  # Réduit pour Kaggle
        self.BATCH_SIZE = 2  # Petit batch pour mémoire limitée
        self.EPOCHS = 50
        self.LEARNING_RATE = 1e-4
        
        # Classes
        self.N_CLASSES = 4  # Background + 3 tissus
        self.CLASS_NAMES = ['Background', 'CSF', 'Gray Matter', 'White Matter']
        
        # Augmentation
        self.ROTATION_RANGE = 15
        self.ZOOM_RANGE = 0.1
    
    def _detect_structure(self):
        """Détecte automatiquement la structure du dataset"""
        print("=== DÉTECTION DE LA STRUCTURE DU DATASET ===")
        
        if not os.path.exists(self.DATA_PATH):
            print(f"❌ Dataset non trouvé à {self.DATA_PATH}")
            return
        
        # Explorer la structure
        print(f"📁 Structure trouvée:")
        for root, dirs, files in os.walk(self.DATA_PATH):
            level = root.replace(self.DATA_PATH, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
        
        # Détecter les chemins d'images et masks
        self.TRAIN_IMG_PATH = self._find_path('train', 'image')
        self.TRAIN_MASK_PATH = self._find_path('train', 'mask')
        self.VALID_IMG_PATH = self._find_path('valid', 'image')
        self.VALID_MASK_PATH = self._find_path('valid', 'mask')
        
        print(f"\n✅ Chemins détectés:")
        print(f"Train Images: {self.TRAIN_IMG_PATH}")
        print(f"Train Masks: {self.TRAIN_MASK_PATH}")
        print(f"Valid Images: {self.VALID_IMG_PATH}")
        print(f"Valid Masks: {self.VALID_MASK_PATH}")
    
    def _find_path(self, split, data_type):
        """Trouve le chemin pour un split et type de données donné"""
        possible_paths = [
            os.path.join(self.DATA_PATH, split, data_type),
            os.path.join(self.DATA_PATH, split, f"{data_type}s"),
            os.path.join(self.DATA_PATH, split, f"{data_type}_data"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        print(f"⚠️  Chemin non trouvé pour {split}/{data_type}")
        return None

config = Config()

# =============================================================================
# 3. DATA LOADING ET PREPROCESSING
# =============================================================================

def load_nifti_file(filepath):
    """Charge un fichier NIFTI et retourne les données"""
    try:
        nifti = nib.load(filepath)
        data = nifti.get_fdata()
        return data.astype(np.float32)
    except Exception as e:
        print(f"Erreur lors du chargement de {filepath}: {e}")
        return None

def normalize_image(image):
    """Normalise l'image entre 0 et 1"""
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    return image

def resize_volume(volume, target_shape):
    """Redimensionne le volume 3D"""
    current_shape = volume.shape
    factors = [target_shape[i] / current_shape[i] for i in range(3)]
    resized = ndimage.zoom(volume, factors, order=1)
    return resized

def create_multiclass_mask(csf_mask, gm_mask, wm_mask):
    """Combine les 3 masks de probabilité en un mask multiclasse"""
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

def find_matching_files(img_path, mask_path):
    """Trouve les fichiers d'images et leurs masks correspondants"""
    if not img_path or not mask_path:
        return [], []
    
    # Lister les fichiers d'images
    img_files = glob.glob(os.path.join(img_path, "*.nii.gz"))
    img_files = [f for f in img_files if "_img.nii.gz" in f]
    
    matching_pairs = []
    
    for img_file in img_files:
        # Extraire le nom de base
        base_name = os.path.basename(img_file).replace("_img.nii.gz", "")
        
        # Chercher les 3 masks correspondants
        csf_path = os.path.join(mask_path, f"{base_name}_probmask_csf.nii.gz")
        gm_path = os.path.join(mask_path, f"{base_name}_probmask_graymatter.nii.gz")
        wm_path = os.path.join(mask_path, f"{base_name}_probmask_whitematter.nii.gz")
        
        # Vérifier que tous les masks existent
        if all(os.path.exists(p) for p in [csf_path, gm_path, wm_path]):
            matching_pairs.append({
                'image': img_file,
                'csf': csf_path,
                'gm': gm_path,
                'wm': wm_path,
                'base_name': base_name
            })
    
    return matching_pairs

def load_dataset(img_path, mask_path, max_samples=None):
    """Charge le dataset avec détection automatique des fichiers"""
    if not img_path or not mask_path:
        print(f"❌ Chemins invalides: img={img_path}, mask={mask_path}")
        return np.array([]), np.array([])
    
    # Trouver les paires de fichiers correspondants
    matching_pairs = find_matching_files(img_path, mask_path)
    
    if not matching_pairs:
        print(f"❌ Aucune paire image-mask trouvée dans {img_path}")
        return np.array([]), np.array([])
    
    print(f"✅ Trouvé {len(matching_pairs)} paires de fichiers")
    
    if max_samples:
        matching_pairs = matching_pairs[:max_samples]
        print(f"📊 Utilisation de {len(matching_pairs)} échantillons")
    
    images = []
    masks = []
    
    for i, pair in enumerate(matching_pairs):
        if i % 10 == 0:
            print(f"Chargement: {i+1}/{len(matching_pairs)}")
        
        try:
            # Charger l'image
            image = load_nifti_file(pair['image'])
            if image is None:
                continue
            
            # Charger les 3 masks
            csf_mask = load_nifti_file(pair['csf'])
            gm_mask = load_nifti_file(pair['gm'])
            wm_mask = load_nifti_file(pair['wm'])
            
            if any(mask is None for mask in [csf_mask, gm_mask, wm_mask]):
                continue
            
            # Créer le mask multiclasse
            multiclass_mask = create_multiclass_mask(csf_mask, gm_mask, wm_mask)
            
            # Redimensionner
            image_resized = resize_volume(image, config.IMG_SIZE)
            mask_resized = resize_volume(multiclass_mask, config.IMG_SIZE)
            
            # Normaliser l'image
            image_normalized = normalize_image(image_resized)
            
            images.append(image_normalized)
            masks.append(mask_resized)
            
        except Exception as e:
            print(f"Erreur avec {pair['base_name']}: {e}")
            continue
    
    print(f"✅ Chargé {len(images)} échantillons avec succès")
    return np.array(images), np.array(masks)

# =============================================================================
# 4. MODÈLE U-NET 3D (même que précédemment)
# =============================================================================

def conv_block_3d(inputs, filters, kernel_size=3, padding='same', activation='relu'):
    """Bloc de convolution 3D"""
    x = layers.Conv3D(filters, kernel_size, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def unet_3d(input_shape, n_classes):
    """Architecture U-Net 3D"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = conv_block_3d(inputs, 32)
    c1 = conv_block_3d(c1, 32)
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)
    
    c2 = conv_block_3d(p1, 64)
    c2 = conv_block_3d(c2, 64)
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)
    
    c3 = conv_block_3d(p2, 128)
    c3 = conv_block_3d(c3, 128)
    p3 = layers.MaxPooling3D((2, 2, 2))(c3)
    
    # Bottleneck
    c4 = conv_block_3d(p3, 256)
    c4 = conv_block_3d(c4, 256)
    
    # Decoder
    u5 = layers.UpSampling3D((2, 2, 2))(c4)
    u5 = layers.Concatenate()([u5, c3])
    c5 = conv_block_3d(u5, 128)
    c5 = conv_block_3d(c5, 128)
    
    u6 = layers.UpSampling3D((2, 2, 2))(c5)
    u6 = layers.Concatenate()([u6, c2])
    c6 = conv_block_3d(u6, 64)
    c6 = conv_block_3d(c6, 64)
    
    u7 = layers.UpSampling3D((2, 2, 2))(c6)
    u7 = layers.Concatenate()([u7, c1])
    c7 = conv_block_3d(u7, 32)
    c7 = conv_block_3d(c7, 32)
    
    # Output
    outputs = layers.Conv3D(n_classes, 1, activation='softmax')(c7)
    
    model = keras.Model(inputs, outputs)
    return model

# =============================================================================
# 5. MÉTRIQUES ET LOSS (même que précédemment)
# =============================================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Coefficient de Dice"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Loss basée sur Dice"""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Combinaison de crossentropy et dice loss"""
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice_loss_val = dice_loss(y_true, y_pred)
    return ce_loss + dice_loss_val

# =============================================================================
# 6. ENTRÂINEMENT PRINCIPAL
# =============================================================================

def train_model():
    """Fonction principale d'entraînement"""
    
    print("=== CHARGEMENT DU DATASET ===")
    
    # Charger les données
    X_train, y_train = load_dataset(config.TRAIN_IMG_PATH, config.TRAIN_MASK_PATH, max_samples=30)
    X_val, y_val = load_dataset(config.VALID_IMG_PATH, config.VALID_MASK_PATH, max_samples=10)
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("❌ Impossible de charger les données. Vérifiez la structure du dataset.")
        return None, None
    
    print(f"✅ Train shapes: {X_train.shape}, {y_train.shape}")
    print(f"✅ Val shapes: {X_val.shape}, {y_val.shape}")
    
    # One-hot encoding des masks
    y_train_oh = tf.keras.utils.to_categorical(y_train, config.N_CLASSES)
    y_val_oh = tf.keras.utils.to_categorical(y_val, config.N_CLASSES)
    
    print("=== CRÉATION DU MODÈLE ===")
    model = unet_3d(input_shape=config.IMG_SIZE + (1,), n_classes=config.N_CLASSES)
    
    # Compiler le modèle
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=combined_loss,
        metrics=['accuracy', dice_coefficient]
    )
    
    print("=== CONFIGURATION DES CALLBACKS ===")
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_dice_coefficient',
            mode='max'
        )
    ]
    
    print("=== DÉMARRAGE DE L'ENTRÂINEMENT ===")
    history = model.fit(
        X_train[..., np.newaxis], y_train_oh,
        validation_data=(X_val[..., np.newaxis], y_val_oh),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# =============================================================================
# 7. EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    print("=== DÉMARRAGE DE L'ENTRÂINEMENT ===")
    
    # Vérifier la disponibilité du GPU
    print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
    
    # Entraîner le modèle
    model, history = train_model()
    
    if model is not None:
        print("=== ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ===")
        
        # Sauvegarder le modèle
        model.save('brain_segmentation_model.h5')
        print("✅ Modèle sauvegardé: brain_segmentation_model.h5")
        
        # Afficher les métriques finales
        print(f"✅ Accuracy finale: {history.history['accuracy'][-1]:.4f}")
        print(f"✅ Dice coefficient final: {history.history['dice_coefficient'][-1]:.4f}")
    else:
        print("❌ Échec de l'entraînement")