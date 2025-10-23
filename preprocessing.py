# =============================================================================
# PRÉTRAITEMENT DES DONNÉES POUR LA SEGMENTATION CÉRÉBRALE 3D
# =============================================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Medical Imaging
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import train_test_split

from config import config
from utils import calculate_metrics, print_metrics

def debug_dataset_structure():
    """Debug la structure du dataset"""
    print("=== DEBUG STRUCTURE DU DATASET ===")
    if not os.path.exists(config.DATA_PATH):
        print(f"❌ Chemin DATA_PATH non trouvé: {config.DATA_PATH}")
        print("Vérifiez que le dataset est ajouté au notebook via 'Add Data' sur Kaggle.")
        return False

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
    
    return True

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
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image)
    
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

def preprocess_single_sample(pair):
    """Prétraite un seul échantillon"""
    try:
        # Charger l'image
        image = load_nifti_file(pair['image'])
        if image is None:
            return None, None

        # Charger les masks
        csf_mask = load_nifti_file(pair['csf'])
        gm_mask = load_nifti_file(pair['gm'])
        wm_mask = load_nifti_file(pair['wm'])
        
        if any(mask is None for mask in [csf_mask, gm_mask, wm_mask]):
            return None, None

        # Créer le mask multiclasse
        multiclass_mask = create_multiclass_mask(csf_mask, gm_mask, wm_mask)

        # Redimensionner
        image_resized = resize_volume(image, config.IMG_SIZE)
        mask_resized = resize_volume(multiclass_mask, config.IMG_SIZE)

        # Normaliser l'image
        image_normalized = normalize_image(image_resized)

        return image_normalized, mask_resized

    except Exception as e:
        print(f"Erreur lors du prétraitement de {pair['base_name']}: {e}")
        return None, None

def load_and_preprocess_dataset(max_samples=None):
    """Charge et prétraite le dataset complet"""
    print("=== CHARGEMENT ET PRÉTRAITEMENT DU DATASET ===")
    
    # Vérifier la structure du dataset
    if not debug_dataset_structure():
        return np.array([]), np.array([]), []

    # Trouver toutes les paires
    matching_pairs = find_all_matching_pairs()
    if not matching_pairs:
        print("❌ Aucune paire trouvée! Vérifiez le debug ci-dessus pour la structure.")
        return np.array([]), np.array([]), []

    # Limiter le nombre d'échantillons si spécifié
    if max_samples:
        matching_pairs = matching_pairs[:max_samples]
        print(f"Limitation à {max_samples} échantillons")

    print(f"Prétraitement de {len(matching_pairs)} échantillons...")

    images = []
    masks = []
    metadata = []

    for i, pair in enumerate(matching_pairs):
        if i % 50 == 0:
            print(f"Progression: {i}/{len(matching_pairs)}")

        image, mask = preprocess_single_sample(pair)
        
        if image is not None and mask is not None:
            images.append(image)
            masks.append(mask)
            metadata.append({
                'base_name': pair['base_name'],
                'split': pair['split'],
                'original_shape': pair.get('original_shape', 'unknown')
            })

    if len(images) == 0:
        print("❌ Aucun échantillon chargé avec succès!")
        return np.array([]), np.array([]), []

    print(f"✅ {len(images)} échantillons prétraités avec succès")
    
    # Vérifier les formes
    print(f"Forme des images: {np.array(images).shape}")
    print(f"Forme des masks: {np.array(masks).shape}")
    
    return np.array(images), np.array(masks), metadata

def split_dataset(images, masks, metadata):
    """Divise le dataset en train/val/test"""
    print("=== DIVISION DU DATASET ===")
    
    # Séparer par split original si disponible
    train_indices = [i for i, meta in enumerate(metadata) if meta['split'] == 'train']
    val_indices = [i for i, meta in enumerate(metadata) if meta['split'] == 'valid']
    test_indices = [i for i, meta in enumerate(metadata) if meta['split'] == 'test']
    
    # Si pas de split original, diviser aléatoirement
    if not train_indices and not val_indices and not test_indices:
        print("Pas de split original détecté, division aléatoire...")
        train_indices, temp_indices = train_test_split(
            range(len(images)), 
            test_size=1-config.TRAIN_SPLIT, 
            random_state=42
        )
        val_indices, test_indices = train_test_split(
            temp_indices, 
            test_size=config.TEST_SPLIT/(config.VAL_SPLIT + config.TEST_SPLIT), 
            random_state=42
        )
    
    # Créer les splits
    X_train = images[train_indices] if train_indices else np.array([])
    y_train = masks[train_indices] if train_indices else np.array([])
    train_metadata = [metadata[i] for i in train_indices] if train_indices else []
    
    X_val = images[val_indices] if val_indices else np.array([])
    y_val = masks[val_indices] if val_indices else np.array([])
    val_metadata = [metadata[i] for i in val_indices] if val_indices else []
    
    X_test = images[test_indices] if test_indices else np.array([])
    y_test = masks[test_indices] if test_indices else np.array([])
    test_metadata = [metadata[i] for i in test_indices] if test_indices else []
    
    print(f"Train: {len(X_train)} échantillons")
    print(f"Validation: {len(X_val)} échantillons")
    print(f"Test: {len(X_test)} échantillons")
    
    return {
        'train': {'X': X_train, 'y': y_train, 'metadata': train_metadata},
        'val': {'X': X_val, 'y': y_val, 'metadata': val_metadata},
        'test': {'X': X_test, 'y': y_test, 'metadata': test_metadata}
    }

def save_preprocessed_data(splits, save_path=None):
    """Sauvegarde les données prétraitées"""
    if save_path is None:
        save_path = config.PREPROCESSED_PATH
    
    print(f"=== SAUVEGARDE DES DONNÉES PRÉTRAITÉES ===")
    print(f"Sauvegarde dans: {save_path}")
    
    for split_name, split_data in splits.items():
        if len(split_data['X']) > 0:
            # Sauvegarder les données
            np.save(os.path.join(save_path, f'X_{split_name}.npy'), split_data['X'])
            np.save(os.path.join(save_path, f'y_{split_name}.npy'), split_data['y'])
            
            # Sauvegarder les métadonnées
            import json
            with open(os.path.join(save_path, f'metadata_{split_name}.json'), 'w') as f:
                json.dump(split_data['metadata'], f, indent=2)
            
            print(f"✅ {split_name}: {len(split_data['X'])} échantillons sauvegardés")
    
    print("=== SAUVEGARDE TERMINÉE ===")

def load_preprocessed_data(load_path=None):
    """Charge les données prétraitées"""
    if load_path is None:
        load_path = config.PREPROCESSED_PATH
    
    print(f"=== CHARGEMENT DES DONNÉES PRÉTRAITÉES ===")
    print(f"Chargement depuis: {load_path}")
    
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        X_path = os.path.join(load_path, f'X_{split_name}.npy')
        y_path = os.path.join(load_path, f'y_{split_name}.npy')
        metadata_path = os.path.join(load_path, f'metadata_{split_name}.json')
        
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
    """Fonction principale de prétraitement"""
    print("=== DÉMARRAGE DU PRÉTRAITEMENT ===")
    
    # Charger et prétraiter le dataset
    images, masks, metadata = load_and_preprocess_dataset(config.MAX_SAMPLES)
    
    if len(images) == 0:
        print("❌ Échec du prétraitement")
        return None
    
    # Diviser le dataset
    splits = split_dataset(images, masks, metadata)
    
    # Sauvegarder les données prétraitées
    save_preprocessed_data(splits)
    
    # Afficher un résumé
    print("\n=== RÉSUMÉ DU PRÉTRAITEMENT ===")
    for split_name, split_data in splits.items():
        if len(split_data['X']) > 0:
            print(f"{split_name.upper()}:")
            print(f"  Images: {split_data['X'].shape}")
            print(f"  Masks: {split_data['y'].shape}")
            print(f"  Classes uniques: {np.unique(split_data['y'])}")
    
    print("=== PRÉTRAITEMENT TERMINÉ ===")
    return splits

if __name__ == "__main__":
    splits = main()