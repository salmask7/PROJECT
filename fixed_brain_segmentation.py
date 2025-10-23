# =============================================================================
# BRAIN ANATOMICAL SEGMENTATION - VERSION OPTIMISÉE ET ROBUSTE
# Dataset: 3D Brain Tissue Segmentation (726 sujets)
# Modèle: U-Net 3D pour segmentation CSF, Gray Matter, White Matter
# Améliorations: Recherche récursive des fichiers, augmentation des données,
#                modèle avec dropout pour réduire overfitting, sauvegarde corrigée, debug étendu
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
        self.NUM_AUGMENTATIONS = 3  # Nombre d'augmentations par sample

config = Config()

# =============================================================================
# DEBUG: VÉRIFIER LA STRUCTURE DU DATASET
# =============================================================================

def debug_dataset_structure():
    print("=== DEBUG STRUCTURE DU DATASET ===")
    if not os.path.exists(config.DATA_PATH):
        print(f"❌ Chemin DATA_PATH non trouvé: {config.DATA_PATH}")
        print("Vérifiez que le dataset est ajouté au notebook via 'Add Data' sur Kaggle.")
        return

    print(f"Contenu de {config.DATA_PATH}:")
    print(os.listdir(config.DATA_PATH))

    print("\nArborescence complète:")
    for root, dirs, files in os.walk(config.DATA_PATH):
        level = root.replace(config.DATA_PATH, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in sorted(files)[:5]:  # Afficher seulement 5 fichiers par dossier
            print(f"{subindent}- {f}")
        if len(files) > 5:
            print(f"{subindent}- ... ({len(files)} fichiers au total)")

debug_dataset_structure()

# =============================================================================
# 3. DATA LOADING ET PREPROCESSING ROBUSTE
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

def find_all_matching_pairs():
    """Trouve tous les fichiers de manière récursive et matche les paires"""
    print("=== RECHERCHE RÉCURSIVE DES FICHIERS ===")

    # Méthode 1: Recherche récursive avec glob
    all_files = []
    for root, dirs, files in os.walk(config.DATA_PATH):
        for file in files:
            if file.endswith('.nii'):
                all_files.append(os.path.join(root, file))
    
    print(f"Total fichiers trouvés: {len(all_files)}")

    # Méthode 2: Recherche spécifique par dossier
    img_files = []
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(config.DATA_PATH, split)
        if os.path.exists(split_path):
            image_dir = os.path.join(split_path, 'image')
            if os.path.exists(image_dir):
                for file in os.listdir(image_dir):
                    if file.endswith('_img.nii'):
                        img_files.append(os.path.join(image_dir, file))

    print(f"Images trouvées: {len(img_files)}")

    matching_pairs = []

    for img_file in img_files:
        base_name = os.path.basename(img_file).replace('_img.nii', '')
        
        # Déterminer le split (train/valid/test)
        split = None
        if '/train/' in img_file:
            split = 'train'
        elif '/valid/' in img_file:
            split = 'valid'
        elif '/test/' in img_file:
            split = 'test'
        
        if split is None:
            continue
            
        # Chercher les masks dans le bon dossier
        mask_dir = os.path.join(config.DATA_PATH, split, 'mask')
        if not os.path.exists(mask_dir):
            continue
            
        csf_path = os.path.join(mask_dir, f"{base_name}_probmask_csf.nii")
        gm_path = os.path.join(mask_dir, f"{base_name}_probmask_graymatter.nii")
        wm_path = os.path.join(mask_dir, f"{base_name}_probmask_whitematter.nii")

        if all(os.path.exists(p) for p in [csf_path, gm_path, wm_path]):
            matching_pairs.append({
                'image': img_file,
                'csf': csf_path,
                'gm': gm_path,
                'wm': wm_path,
                'base_name': base_name,
                'split': split
            })

    print(f"✅ Paires complètes: {len(matching_pairs)}")
    if len(matching_pairs) > 0:
        print("Exemple de paire:")
        print(matching_pairs[0])
        print(f"Split distribution:")
        splits = [p['split'] for p in matching_pairs]
        from collections import Counter
        print(Counter(splits))
    return matching_pairs

def load_dataset(max_samples=None):
    """Charge le dataset avec les paires trouvées"""
    matching_pairs = find_all_matching_pairs()

    if not matching_pairs:
        print("❌ Aucune paire trouvée! Vérifiez le debug ci-dessus pour la structure.")
        return np.array([]), np.array([])

    if max_samples:
        matching_pairs = matching_pairs[:max_samples]

    print(f"Chargement de {len(matching_pairs)} échantillons...")

    images = []
    masks = []

    for i, pair in enumerate(matching_pairs):
        if i % 50 == 0:
            print(f"Progression: {i}/{len(matching_pairs)}")

        image = load_nifti_file(pair['image'])
        if image is None: continue

        csf_mask = load_nifti_file(pair['csf'])
        gm_mask = load_nifti_file(pair['gm'])
        wm_mask = load_nifti_file(pair['wm'])
        if any(mask is None for mask in [csf_mask, gm_mask, wm_mask]): continue

        multiclass_mask = create_multiclass_mask(csf_mask, gm_mask, wm_mask)

        image_resized = resize_volume(image, config.IMG_SIZE)
        mask_resized = resize_volume(multiclass_mask, config.IMG_SIZE)

        image_normalized = normalize_image(image_resized)

        images.append(image_normalized)
        masks.append(mask_resized)

    if len(images) == 0:
        print("❌ Aucun échantillon chargé avec succès! Vérifiez si les fichiers NIFTI sont valides.")
    return np.array(images), np.array(masks)

# =============================================================================
# 4. AUGMENTATION DES DONNÉES - VERSION CORRIGÉE
# =============================================================================

def random_rotation_3d(image, mask, max_angle=config.ROTATION_RANGE):
    """Rotation 3D avec préservation de la forme"""
    angle = np.random.uniform(-max_angle, max_angle)
    image_rot = ndimage.rotate(image, angle, axes=(0, 1), reshape=False, order=1)
    mask_rot = ndimage.rotate(mask, angle, axes=(0, 1), reshape=False, order=0)
    return image_rot, mask_rot

def random_zoom_3d(image, mask, zoom_range=config.ZOOM_RANGE):
    """Zoom 3D avec garantie de forme finale cohérente"""
    zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    
    # Appliquer le zoom
    image_zoom = ndimage.zoom(image, zoom_factor, order=1)
    mask_zoom = ndimage.zoom(mask, zoom_factor, order=0)

    # S'assurer que la forme finale est exactement la même que l'original
    target_shape = image.shape
    
    # Si la forme est déjà correcte, retourner directement
    if image_zoom.shape == target_shape:
        return image_zoom, mask_zoom
    
    # Sinon, redimensionner pour avoir exactement la forme cible
    image_final = resize_volume(image_zoom, target_shape)
    mask_final = resize_volume(mask_zoom, target_shape)
    
    return image_final, mask_final

def random_flip_3d(image, mask):
    """Flip aléatoire 3D"""
    axes = np.random.choice([0, 1, 2], size=np.random.randint(1, 4), replace=False)
    
    image_flipped = image.copy()
    mask_flipped = mask.copy()
    
    for axis in axes:
        image_flipped = np.flip(image_flipped, axis=axis)
        mask_flipped = np.flip(mask_flipped, axis=axis)
    
    return image_flipped, mask_flipped

def augment_data(images, masks, num_augmentations=config.NUM_AUGMENTATIONS):
    """Augmentation des données avec vérification de forme"""
    print(f"Augmentation de {len(images)} échantillons avec {num_augmentations} augmentations chacun...")
    
    augmented_images = []
    augmented_masks = []
    
    # Ajouter les données originales
    for img, mask in zip(images, masks):
        augmented_images.append(img)
        augmented_masks.append(mask)
    
    # Générer les augmentations
    for i in range(len(images)):
        if i % 10 == 0:
            print(f"Augmentation: {i}/{len(images)}")
            
        for _ in range(num_augmentations):
            # Choisir une augmentation aléatoire
            augmentation_type = np.random.choice(['rotation', 'zoom', 'flip', 'combined'])
            
            try:
                if augmentation_type == 'rotation':
                    img_aug, mask_aug = random_rotation_3d(images[i], masks[i])
                elif augmentation_type == 'zoom':
                    img_aug, mask_aug = random_zoom_3d(images[i], masks[i])
                elif augmentation_type == 'flip':
                    img_aug, mask_aug = random_flip_3d(images[i], masks[i])
                else:  # combined
                    img_aug, mask_aug = random_rotation_3d(images[i], masks[i])
                    img_aug, mask_aug = random_zoom_3d(img_aug, mask_aug)
                    if np.random.random() > 0.5:
                        img_aug, mask_aug = random_flip_3d(img_aug, mask_aug)
                
                # Vérifier que les formes sont correctes
                if img_aug.shape == images[i].shape and mask_aug.shape == masks[i].shape:
                    augmented_images.append(img_aug)
                    augmented_masks.append(mask_aug)
                else:
                    print(f"Forme incorrecte après augmentation: {img_aug.shape} vs {images[i].shape}")
                    
            except Exception as e:
                print(f"Erreur lors de l'augmentation {augmentation_type}: {e}")
                continue
    
    print(f"Total après augmentation: {len(augmented_images)} échantillons")
    
    # Vérifier que toutes les formes sont identiques
    if len(augmented_images) > 0:
        target_shape = augmented_images[0].shape
        print(f"Forme cible: {target_shape}")
        
        # Filtrer les échantillons avec des formes incorrectes
        valid_images = []
        valid_masks = []
        
        for img, mask in zip(augmented_images, augmented_masks):
            if img.shape == target_shape and mask.shape == target_shape:
                valid_images.append(img)
                valid_masks.append(mask)
            else:
                print(f"Forme incorrecte détectée: {img.shape} vs {target_shape}")
        
        print(f"Échantillons valides après filtrage: {len(valid_images)}")
        return np.array(valid_images), np.array(valid_masks)
    
    return np.array(augmented_images), np.array(augmented_masks)

# =============================================================================
# 5. MODÈLE U-Net 3D AVEC DROPOUT
# =============================================================================

def conv_block_3d(inputs, filters, kernel_size=3, padding='same', activation='relu', dropout=0.1):
    x = layers.Conv3D(filters, kernel_size, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x

def unet_3d(input_shape, n_classes):
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
# 6. MÉTRIQUES ET LOSS
# =============================================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice_loss_val = dice_loss(y_true, y_pred)
    return ce_loss + dice_loss_val

# =============================================================================
# 7. ENTRÂINEMENT
# =============================================================================

def train_model():
    print("=== CHARGEMENT DU DATASET ===")
    X_data, y_data = load_dataset(max_samples=50)  # Augmente si possible

    if len(X_data) < 10:
        print("❌ Pas assez de données chargées! Vérifiez le debug ci-dessus.")
        return None, None

    # Augmentation
    print("=== AUGMENTATION DES DONNÉES ===")
    X_data_aug, y_data_aug = augment_data(X_data, y_data)
    print(f"Après augmentation: {X_data_aug.shape}")

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_data_aug, y_data_aug, test_size=0.2, random_state=42
    )

    print(f"Train shapes: {X_train.shape}, {y_train.shape}")
    print(f"Val shapes: {X_val.shape}, {y_val.shape}")

    # One-hot
    y_train_oh = tf.keras.utils.to_categorical(y_train, config.N_CLASSES)
    y_val_oh = tf.keras.utils.to_categorical(y_val, config.N_CLASSES)

    print("=== CRÉATION DU MODÈLE ===")
    model = unet_3d(input_shape=config.IMG_SIZE + (1,), n_classes=config.N_CLASSES)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=combined_loss,
        metrics=['accuracy', dice_coefficient]
    )

    print("=== CONFIGURATION DES CALLBACKS ===")
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_dice_coefficient', mode='max'),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss'),
        keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_dice_coefficient', mode='max')
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title('Accuracy')
    axes[1].legend()

    axes[2].plot(history.history['dice_coefficient'], label='Train Dice')
    axes[2].plot(history.history['val_dice_coefficient'], label='Val Dice')
    axes[2].set_title('Dice Coefficient')
    axes[2].legend()

    plt.show()

def visualize_predictions(model, X_val, y_val, n_samples=3):
    predictions = model.predict(X_val[:n_samples][..., np.newaxis])
    pred_classes = np.argmax(predictions, axis=-1)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    for i in range(n_samples):
        axes[i, 0].imshow(X_val[i][:, :, 64], cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(y_val[i][:, :, 64], cmap='tab10')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_classes[i][:, :, 64], cmap='tab10')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

        diff = np.abs(y_val[i][:, :, 64] - pred_classes[i][:, :, 64])
        axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title('Difference')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()

# =============================================================================
# 9. EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    print("=== DÉMARRAGE DE L'ENTRÂINEMENT ===")

    # Vérifier GPU
    print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")

    # Entraîner
    model, history = train_model()

    if model is not None:
        # Visualiser history
        plot_training_history(history)

        # Charger quelques val pour visu (réutilise load mais petit)
        X_val_vis, y_val_vis = load_dataset(max_samples=5)
        visualize_predictions(model, X_val_vis, y_val_vis)

        # Sauvegarde
        model.save('brain_segmentation_model.keras')
        model.save_weights('brain_segmentation.weights.h5')
        print("=== MODÈLE SAUVEGARDÉ ===")

        print("=== ENTRAÎNEMENT TERMINÉ ===")
    else:
        print("=== ÉCHEC DE L'ENTRAÎNEMENT ===")