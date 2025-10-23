# =============================================================================
# BRAIN ANATOMICAL SEGMENTATION - KAGGLE NOTEBOOK
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
    # Dataset paths - Correction des chemins Kaggle
    DATA_PATH = '/kaggle/input/3dbraintissuesegmentation'
    TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train', 'image')
    TRAIN_MASK_PATH = os.path.join(DATA_PATH, 'train', 'mask')
    VALID_IMG_PATH = os.path.join(DATA_PATH, 'valid', 'image')
    VALID_MASK_PATH = os.path.join(DATA_PATH, 'valid', 'mask')
    
    # Vérification des chemins
    def check_paths(self):
        print("=== VÉRIFICATION DES CHEMINS ===")
        print(f"DATA_PATH existe: {os.path.exists(self.DATA_PATH)}")
        print(f"TRAIN_IMG_PATH existe: {os.path.exists(self.TRAIN_IMG_PATH)}")
        print(f"TRAIN_MASK_PATH existe: {os.path.exists(self.TRAIN_MASK_PATH)}")
        print(f"VALID_IMG_PATH existe: {os.path.exists(self.VALID_IMG_PATH)}")
        print(f"VALID_MASK_PATH existe: {os.path.exists(self.VALID_MASK_PATH)}")
        
        # Lister le contenu du répertoire principal
        if os.path.exists(self.DATA_PATH):
            print(f"\nContenu de {self.DATA_PATH}:")
            print(os.listdir(self.DATA_PATH))
            
        # Lister le contenu du train si il existe
        train_path = os.path.join(self.DATA_PATH, 'train')
        if os.path.exists(train_path):
            print(f"\nContenu de {train_path}:")
            print(os.listdir(train_path))
    
    # Model parameters
    IMG_SIZE = (128, 128, 128)  # Réduit pour Kaggle
    BATCH_SIZE = 2  # Petit batch pour mémoire limitée
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    # Classes
    N_CLASSES = 4  # Background + 3 tissus
    CLASS_NAMES = ['Background', 'CSF', 'Gray Matter', 'White Matter']
    
    # Augmentation
    ROTATION_RANGE = 15
    ZOOM_RANGE = 0.1

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

def load_dataset(data_path, img_path, mask_path, max_samples=None):
    """Charge le dataset complet"""
    images = []
    masks = []
    
    # Lister tous les fichiers d'images
    img_files = sorted([f for f in os.listdir(img_path) if f.endswith('.nii.gz')])
    
    if max_samples:
        img_files = img_files[:max_samples]
    
    print(f"Chargement de {len(img_files)} échantillons...")
    
    for i, img_file in enumerate(img_files):
        if i % 50 == 0:
            print(f"Progression: {i}/{len(img_files)}")
            
        # Charger l'image
        img_path_full = os.path.join(img_path, img_file)
        image = load_nifti_file(img_path_full)
        if image is None:
            continue
            
        # Charger les 3 masks
        base_name = img_file.replace('_img.nii.gz', '')
        csf_path = os.path.join(mask_path, f"{base_name}_probmask_csf.nii.gz")
        gm_path = os.path.join(mask_path, f"{base_name}_probmask_graymatter.nii.gz")
        wm_path = os.path.join(mask_path, f"{base_name}_probmask_whitematter.nii.gz")
        
        csf_mask = load_nifti_file(csf_path)
        gm_mask = load_nifti_file(gm_path)
        wm_mask = load_nifti_file(wm_path)
        
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
    
    return np.array(images), np.array(masks)

# =============================================================================
# 4. MODÈLE U-NET 3D
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
# 5. DATA AUGMENTATION
# =============================================================================

def random_rotation_3d(image, mask, max_angle=15):
    """Rotation aléatoire 3D"""
    angle = np.random.uniform(-max_angle, max_angle)
    
    # Rotation autour de l'axe Z
    image_rot = ndimage.rotate(image, angle, axes=(0, 1), reshape=False, order=1)
    mask_rot = ndimage.rotate(mask, angle, axes=(0, 1), reshape=False, order=0)
    
    return image_rot, mask_rot

def random_zoom_3d(image, mask, zoom_range=0.1):
    """Zoom aléatoire 3D"""
    zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    
    image_zoom = ndimage.zoom(image, zoom_factor, order=1)
    mask_zoom = ndimage.zoom(mask, zoom_factor, order=0)
    
    # Recadrer à la taille originale
    current_shape = image_zoom.shape
    target_shape = image.shape
    
    start = [(current_shape[i] - target_shape[i]) // 2 for i in range(3)]
    end = [start[i] + target_shape[i] for i in range(3)]
    
    image_crop = image_zoom[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    mask_crop = mask_zoom[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    
    return image_crop, mask_crop

def augment_data(images, masks, num_augmentations=5):
    """Augmente le dataset"""
    augmented_images = []
    augmented_masks = []
    
    for i in range(len(images)):
        # Original
        augmented_images.append(images[i])
        augmented_masks.append(masks[i])
        
        # Augmentations
        for _ in range(num_augmentations):
            img_aug, mask_aug = random_rotation_3d(images[i], masks[i])
            img_aug, mask_aug = random_zoom_3d(img_aug, mask_aug)
            
            augmented_images.append(img_aug)
            augmented_masks.append(mask_aug)
    
    return np.array(augmented_images), np.array(augmented_masks)

# =============================================================================
# 6. MÉTRIQUES ET LOSS
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
# 7. ENTRÂINEMENT
# =============================================================================

def train_model():
    """Fonction principale d'entraînement"""
    
    print("=== CHARGEMENT DU DATASET ===")
    
    # Vérifier les chemins d'abord
    config.check_paths()
    
    # Essayer de trouver les bons chemins
    data_path = '/kaggle/input/3dbraintissuesegmentation'
    
    # Vérifier la structure du dataset
    if os.path.exists(data_path):
        print(f"\nStructure du dataset:")
        for root, dirs, files in os.walk(data_path):
            level = root.replace(data_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Afficher seulement les 5 premiers fichiers
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... et {len(files) - 5} autres fichiers")
    
    # Charger seulement un échantillon pour commencer (Kaggle limitation)
    X_train, y_train = load_dataset(config.DATA_PATH, config.TRAIN_IMG_PATH, config.TRAIN_MASK_PATH, max_samples=50)
    X_val, y_val = load_dataset(config.DATA_PATH, config.VALID_IMG_PATH, config.VALID_MASK_PATH, max_samples=20)
    
    print(f"Train shapes: {X_train.shape}, {y_train.shape}")
    print(f"Val shapes: {X_val.shape}, {y_val.shape}")
    
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
# 8. VISUALISATION ET ÉVALUATION
# =============================================================================

def plot_training_history(history):
    """Affiche l'historique d'entraînement"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    # Dice Coefficient
    axes[2].plot(history.history['dice_coefficient'], label='Train Dice')
    axes[2].plot(history.history['val_dice_coefficient'], label='Val Dice')
    axes[2].set_title('Dice Coefficient')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Dice')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, X_val, y_val, n_samples=3):
    """Visualise les prédictions"""
    predictions = model.predict(X_val[:n_samples])
    pred_classes = np.argmax(predictions, axis=-1)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    for i in range(n_samples):
        # Image originale
        axes[i, 0].imshow(X_val[i][:, :, 64], cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(y_val[i][:, :, 64], cmap='tab10')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prédiction
        axes[i, 2].imshow(pred_classes[i][:, :, 64], cmap='tab10')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Différence
        diff = np.abs(y_val[i][:, :, 64] - pred_classes[i][:, :, 64])
        axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title('Difference')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 9. SAUVEGARDE ET EXPORT
# =============================================================================

def save_model_for_download(model, history):
    """Sauvegarde le modèle pour téléchargement"""
    
    # Sauvegarder le modèle complet
    model.save('brain_segmentation_model.h5')
    
    # Sauvegarder les poids seulement
    model.save_weights('brain_segmentation_weights.h5')
    
    # Sauvegarder l'historique
    import json
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
    
    # Créer un fichier de configuration
    config_dict = {
        'img_size': config.IMG_SIZE,
        'n_classes': config.N_CLASSES,
        'class_names': config.CLASS_NAMES,
        'input_shape': config.IMG_SIZE + (1,)
    }
    
    with open('model_config.json', 'w') as f:
        json.dump(config_dict, f)
    
    print("=== MODÈLE SAUVEGARDÉ ===")
    print("Fichiers créés:")
    print("- brain_segmentation_model.h5 (modèle complet)")
    print("- brain_segmentation_weights.h5 (poids seulement)")
    print("- training_history.json (historique)")
    print("- model_config.json (configuration)")
    
    # Créer un fichier ZIP pour téléchargement facile
    import zipfile
    with zipfile.ZipFile('brain_segmentation_model.zip', 'w') as zipf:
        zipf.write('brain_segmentation_model.h5')
        zipf.write('brain_segmentation_weights.h5')
        zipf.write('training_history.json')
        zipf.write('model_config.json')
    
    print("- brain_segmentation_model.zip (tous les fichiers)")

# =============================================================================
# 10. EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    print("=== DÉMARRAGE DE L'ENTRÂINEMENT ===")
    
    # Vérifier la disponibilité du GPU
    print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
    
    # Entraîner le modèle
    model, history = train_model()
    
    # Visualiser les résultats
    plot_training_history(history)
    
    # Charger quelques données de validation pour visualisation
    X_val, y_val = load_dataset(config.DATA_PATH, config.VALID_IMG_PATH, config.VALID_MASK_PATH, max_samples=5)
    visualize_predictions(model, X_val, y_val)
    
    # Sauvegarder le modèle
    save_model_for_download(model, history)
    
    print("=== ENTRAÎNEMENT TERMINÉ ===")
    print("Téléchargez le fichier 'brain_segmentation_model.zip' pour utiliser le modèle localement!")