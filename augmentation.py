# =============================================================================
# AUGMENTATION DES DONNÉES POUR LA SEGMENTATION CÉRÉBRALE 3D
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy import ndimage
from sklearn.model_selection import train_test_split

from config import config
from preprocessing import load_preprocessed_data, save_preprocessed_data

def random_rotation_3d(image, mask, max_angle=config.ROTATION_RANGE):
    """Rotation 3D avec préservation de la forme"""
    angle = np.random.uniform(-max_angle, max_angle)
    
    # Rotation autour de l'axe Z (axial)
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
    from preprocessing import resize_volume
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

def random_noise_3d(image, noise_factor=0.1):
    """Ajoute du bruit gaussien à l'image"""
    noise = np.random.normal(0, noise_factor, image.shape)
    return np.clip(image + noise, 0, 1)

def random_brightness_3d(image, brightness_range=0.2):
    """Modifie la luminosité de l'image"""
    brightness_factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
    return np.clip(image * brightness_factor, 0, 1)

def random_contrast_3d(image, contrast_range=0.2):
    """Modifie le contraste de l'image"""
    contrast_factor = np.random.uniform(1 - contrast_range, 1 + contrast_range)
    mean = np.mean(image)
    return np.clip((image - mean) * contrast_factor + mean, 0, 1)

def elastic_deformation_3d(image, mask, alpha=1000, sigma=30):
    """Déformation élastique 3D"""
    # Générer un champ de déformation aléatoire
    shape = image.shape
    dx = np.random.randn(*shape) * alpha
    dy = np.random.randn(*shape) * alpha
    dz = np.random.randn(*shape) * alpha
    
    # Lisser le champ de déformation
    dx = ndimage.gaussian_filter(dx, sigma)
    dy = ndimage.gaussian_filter(dy, sigma)
    dz = ndimage.gaussian_filter(dz, sigma)
    
    # Créer les grilles de coordonnées
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    
    # Appliquer la déformation
    indices = (y + dy, x + dx, z + dz)
    
    # Interpoler l'image et le mask
    image_deformed = ndimage.map_coordinates(image, indices, order=1, mode='nearest')
    mask_deformed = ndimage.map_coordinates(mask, indices, order=0, mode='nearest')
    
    return image_deformed, mask_deformed

def augment_single_sample(image, mask, num_augmentations=config.NUM_AUGMENTATIONS):
    """Augmente un seul échantillon"""
    augmented_images = [image]
    augmented_masks = [mask]
    
    for _ in range(num_augmentations):
        # Choisir une augmentation aléatoire
        augmentation_type = np.random.choice([
            'rotation', 'zoom', 'flip', 'noise', 'brightness', 
            'contrast', 'elastic', 'combined'
        ])
        
        try:
            if augmentation_type == 'rotation':
                img_aug, mask_aug = random_rotation_3d(image, mask)
            elif augmentation_type == 'zoom':
                img_aug, mask_aug = random_zoom_3d(image, mask)
            elif augmentation_type == 'flip':
                img_aug, mask_aug = random_flip_3d(image, mask)
            elif augmentation_type == 'noise':
                img_aug = random_noise_3d(image)
                mask_aug = mask.copy()
            elif augmentation_type == 'brightness':
                img_aug = random_brightness_3d(image)
                mask_aug = mask.copy()
            elif augmentation_type == 'contrast':
                img_aug = random_contrast_3d(image)
                mask_aug = mask.copy()
            elif augmentation_type == 'elastic':
                img_aug, mask_aug = elastic_deformation_3d(image, mask)
            else:  # combined
                img_aug, mask_aug = random_rotation_3d(image, mask)
                img_aug, mask_aug = random_zoom_3d(img_aug, mask_aug)
                if np.random.random() > 0.5:
                    img_aug, mask_aug = random_flip_3d(img_aug, mask_aug)
                if np.random.random() > 0.5:
                    img_aug = random_brightness_3d(img_aug)
                if np.random.random() > 0.5:
                    img_aug = random_contrast_3d(img_aug)

            # Vérifier que les formes sont correctes
            if img_aug.shape == image.shape and mask_aug.shape == mask.shape:
                augmented_images.append(img_aug)
                augmented_masks.append(mask_aug)
            else:
                print(f"Forme incorrecte après augmentation {augmentation_type}: {img_aug.shape} vs {image.shape}")

        except Exception as e:
            print(f"Erreur lors de l'augmentation {augmentation_type}: {e}")
            continue
    
    return augmented_images, augmented_masks

def augment_dataset(splits, num_augmentations=config.NUM_AUGMENTATIONS):
    """Augmente le dataset complet"""
    print("=== AUGMENTATION DU DATASET ===")
    
    augmented_splits = {}
    
    for split_name, split_data in splits.items():
        if len(split_data['X']) == 0:
            print(f"Split {split_name} vide, ignoré")
            augmented_splits[split_name] = split_data
            continue
        
        print(f"Augmentation du split {split_name} ({len(split_data['X'])} échantillons)...")
        
        augmented_images = []
        augmented_masks = []
        augmented_metadata = []
        
        # Ajouter les données originales
        for i in range(len(split_data['X'])):
            augmented_images.append(split_data['X'][i])
            augmented_masks.append(split_data['y'][i])
            augmented_metadata.append(split_data['metadata'][i])
        
        # Générer les augmentations
        for i in range(len(split_data['X'])):
            if i % 10 == 0:
                print(f"  Progression: {i}/{len(split_data['X'])}")
            
            img_aug_list, mask_aug_list = augment_single_sample(
                split_data['X'][i], 
                split_data['y'][i], 
                num_augmentations
            )
            
            # Ajouter les augmentations (sans l'original qui est déjà ajouté)
            for j in range(1, len(img_aug_list)):
                augmented_images.append(img_aug_list[j])
                augmented_masks.append(mask_aug_list[j])
                # Dupliquer les métadonnées pour les augmentations
                augmented_metadata.append(split_data['metadata'][i].copy())
        
        print(f"  Total après augmentation: {len(augmented_images)} échantillons")
        
        # Vérifier que toutes les formes sont identiques
        if len(augmented_images) > 0:
            target_shape = augmented_images[0].shape
            print(f"  Forme cible: {target_shape}")
            
            # Filtrer les échantillons avec des formes incorrectes
            valid_images = []
            valid_masks = []
            valid_metadata = []
            
            for img, mask, meta in zip(augmented_images, augmented_masks, augmented_metadata):
                if img.shape == target_shape and mask.shape == target_shape:
                    valid_images.append(img)
                    valid_masks.append(mask)
                    valid_metadata.append(meta)
                else:
                    print(f"  Forme incorrecte détectée: {img.shape} vs {target_shape}")
            
            print(f"  Échantillons valides après filtrage: {len(valid_images)}")
            
            augmented_splits[split_name] = {
                'X': np.array(valid_images),
                'y': np.array(valid_masks),
                'metadata': valid_metadata
            }
        else:
            print(f"  Aucun échantillon valide pour {split_name}")
            augmented_splits[split_name] = split_data
    
    return augmented_splits

def visualize_augmentations(original_image, original_mask, augmented_images, augmented_masks, 
                          config, n_samples=4, save_path=None):
    """Visualise les augmentations générées"""
    fig, axes = plt.subplots(2, n_samples + 1, figsize=(4 * (n_samples + 1), 8))
    
    # Image originale
    axes[0, 0].imshow(original_image[:, :, 64], cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original_mask[:, :, 64], cmap='tab10', vmin=0, vmax=3)
    axes[1, 0].set_title('Original Mask')
    axes[1, 0].axis('off')
    
    # Augmentations
    for i in range(min(n_samples, len(augmented_images))):
        axes[0, i + 1].imshow(augmented_images[i][:, :, 64], cmap='gray')
        axes[0, i + 1].set_title(f'Augmented {i + 1}')
        axes[0, i + 1].axis('off')
        
        axes[1, i + 1].imshow(augmented_masks[i][:, :, 64], cmap='tab10', vmin=0, vmax=3)
        axes[1, i + 1].set_title(f'Augmented {i + 1}')
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualisation sauvegardée: {save_path}")
    
    plt.show()

def analyze_augmentation_quality(splits, config):
    """Analyse la qualité des augmentations"""
    print("=== ANALYSE DE LA QUALITÉ DES AUGMENTATIONS ===")
    
    for split_name, split_data in splits.items():
        if len(split_data['X']) == 0:
            continue
        
        print(f"\nSplit {split_name}:")
        print(f"  Nombre d'échantillons: {len(split_data['X'])}")
        print(f"  Forme des images: {split_data['X'].shape}")
        print(f"  Forme des masks: {split_data['y'].shape}")
        
        # Analyser la distribution des classes
        unique_classes, counts = np.unique(split_data['y'], return_counts=True)
        print(f"  Classes présentes: {unique_classes}")
        print(f"  Distribution des classes:")
        for class_id, count in zip(unique_classes, counts):
            percentage = (count / split_data['y'].size) * 100
            print(f"    Classe {class_id}: {count} pixels ({percentage:.2f}%)")
        
        # Vérifier l'intégrité des données
        nan_count = np.isnan(split_data['X']).sum()
        inf_count = np.isinf(split_data['X']).sum()
        print(f"  Valeurs NaN: {nan_count}")
        print(f"  Valeurs Inf: {inf_count}")
        
        # Vérifier la plage des valeurs
        print(f"  Plage des images: [{np.min(split_data['X']):.4f}, {np.max(split_data['X']):.4f}]")
        print(f"  Plage des masks: [{np.min(split_data['y'])}, {np.max(split_data['y'])}]")

def save_augmented_data(splits, save_path=None):
    """Sauvegarde les données augmentées"""
    if save_path is None:
        save_path = config.AUGMENTED_PATH
    
    print(f"=== SAUVEGARDE DES DONNÉES AUGMENTÉES ===")
    print(f"Sauvegarde dans: {save_path}")
    
    for split_name, split_data in splits.items():
        if len(split_data['X']) > 0:
            # Sauvegarder les données
            np.save(os.path.join(save_path, f'X_{split_name}_augmented.npy'), split_data['X'])
            np.save(os.path.join(save_path, f'y_{split_name}_augmented.npy'), split_data['y'])
            
            # Sauvegarder les métadonnées
            import json
            with open(os.path.join(save_path, f'metadata_{split_name}_augmented.json'), 'w') as f:
                json.dump(split_data['metadata'], f, indent=2)
            
            print(f"✅ {split_name}: {len(split_data['X'])} échantillons sauvegardés")
    
    print("=== SAUVEGARDE TERMINÉE ===")

def load_augmented_data(load_path=None):
    """Charge les données augmentées"""
    if load_path is None:
        load_path = config.AUGMENTED_PATH
    
    print(f"=== CHARGEMENT DES DONNÉES AUGMENTÉES ===")
    print(f"Chargement depuis: {load_path}")
    
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        X_path = os.path.join(load_path, f'X_{split_name}_augmented.npy')
        y_path = os.path.join(load_path, f'y_{split_name}_augmented.npy')
        metadata_path = os.path.join(load_path, f'metadata_{split_name}_augmented.json')
        
        if os.path.exists(X_path) and os.path.exists(y_path):
            splits[split_name] = {
                'X': np.load(X_path),
                'y': np.load(y_path),
                'metadata': []
            }
            
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    splits[split_name]['metadata'] = json.load(f)
            
            print(f"✅ {split_name}: {len(splits[split_name]['X'])} échantillons chargés")
        else:
            print(f"❌ {split_name}: fichiers non trouvés")
            splits[split_name] = {'X': np.array([]), 'y': np.array([]), 'metadata': []}
    
    return splits

def main():
    """Fonction principale d'augmentation"""
    print("=== DÉMARRAGE DE L'AUGMENTATION ===")
    
    # Charger les données prétraitées
    splits = load_preprocessed_data()
    
    if not any(len(split_data['X']) > 0 for split_data in splits.values()):
        print("❌ Aucune donnée prétraitée trouvée. Exécutez d'abord preprocessing.py")
        return None
    
    # Augmenter le dataset
    augmented_splits = augment_dataset(splits)
    
    # Analyser la qualité des augmentations
    analyze_augmentation_quality(augmented_splits, config)
    
    # Sauvegarder les données augmentées
    save_augmented_data(augmented_splits)
    
    # Visualiser quelques exemples d'augmentation
    if len(augmented_splits['train']['X']) > 0:
        print("\n=== VISUALISATION DES AUGMENTATIONS ===")
        original_img = augmented_splits['train']['X'][0]
        original_mask = augmented_splits['train']['y'][0]
        
        # Prendre quelques augmentations du premier échantillon
        aug_imgs = augmented_splits['train']['X'][1:5]  # Supposant que les 4 premiers sont des augmentations
        aug_masks = augmented_splits['train']['y'][1:5]
        
        visualize_augmentations(
            original_img, original_mask, aug_imgs, aug_masks, 
            config, save_path=os.path.join(config.OUTPUT_PATH, 'augmentation_examples.png')
        )
    
    print("=== AUGMENTATION TERMINÉE ===")
    return augmented_splits

if __name__ == "__main__":
    augmented_splits = main()