# =============================================================================
# DIAGNOSTIC DU DATASET KAGGLE - 3D BRAIN TISSUE SEGMENTATION
# =============================================================================

import os
import glob
from pathlib import Path

def diagnose_dataset():
    """Diagnostique la structure du dataset Kaggle"""
    
    print("=== DIAGNOSTIC DU DATASET KAGGLE ===")
    
    # Chemin principal
    data_path = '/kaggle/input/3dbraintissuesegmentation'
    
    if not os.path.exists(data_path):
        print(f"❌ Le chemin {data_path} n'existe pas!")
        print("Vérifiez que le dataset est bien attaché au notebook Kaggle")
        return
    
    print(f"✅ Dataset trouvé à: {data_path}")
    
    # Explorer la structure
    print(f"\n📁 Structure du dataset:")
    for root, dirs, files in os.walk(data_path):
        level = root.replace(data_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Afficher seulement les 3 premiers fichiers
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... et {len(files) - 3} autres fichiers")
    
    # Compter les fichiers par type
    print(f"\n📊 Statistiques des fichiers:")
    
    # Images
    img_patterns = [
        '/kaggle/input/3dbraintissuesegmentation/train/image/*.nii.gz',
        '/kaggle/input/3dbraintissuesegmentation/valid/image/*.nii.gz',
        '/kaggle/input/3dbraintissuesegmentation/test/image/*.nii.gz'
    ]
    
    for pattern in img_patterns:
        files = glob.glob(pattern)
        print(f"Images {pattern.split('/')[-3]}: {len(files)} fichiers")
        if files:
            print(f"  Exemple: {os.path.basename(files[0])}")
    
    # Masks
    mask_patterns = [
        '/kaggle/input/3dbraintissuesegmentation/train/mask/*.nii.gz',
        '/kaggle/input/3dbraintissuesegmentation/valid/mask/*.nii.gz',
        '/kaggle/input/3dbraintissuesegmentation/test/mask/*.nii.gz'
    ]
    
    for pattern in mask_patterns:
        files = glob.glob(pattern)
        print(f"Masks {pattern.split('/')[-3]}: {len(files)} fichiers")
        if files:
            print(f"  Exemple: {os.path.basename(files[0])}")
    
    # Vérifier les correspondances
    print(f"\n🔍 Vérification des correspondances:")
    
    # Train
    train_imgs = glob.glob('/kaggle/input/3dbraintissuesegmentation/train/image/*.nii.gz')
    train_masks_csf = glob.glob('/kaggle/input/3dbraintissuesegmentation/train/mask/*csf*.nii.gz')
    train_masks_gm = glob.glob('/kaggle/input/3dbraintissuesegmentation/train/mask/*graymatter*.nii.gz')
    train_masks_wm = glob.glob('/kaggle/input/3dbraintissuesegmentation/train/mask/*whitematter*.nii.gz')
    
    print(f"Train - Images: {len(train_imgs)}")
    print(f"Train - Masks CSF: {len(train_masks_csf)}")
    print(f"Train - Masks GM: {len(train_masks_gm)}")
    print(f"Train - Masks WM: {len(train_masks_wm)}")
    
    # Afficher quelques exemples de noms de fichiers
    if train_imgs:
        print(f"\n📝 Exemples de noms de fichiers:")
        print(f"Image: {os.path.basename(train_imgs[0])}")
        if train_masks_csf:
            print(f"CSF: {os.path.basename(train_masks_csf[0])}")
        if train_masks_gm:
            print(f"GM: {os.path.basename(train_masks_gm[0])}")
        if train_masks_wm:
            print(f"WM: {os.path.basename(train_masks_wm[0])}")
    
    # Vérifier la cohérence des noms
    if train_imgs and train_masks_csf:
        print(f"\n✅ Vérification de cohérence:")
        img_base = os.path.basename(train_imgs[0]).replace('_img.nii.gz', '')
        csf_base = os.path.basename(train_masks_csf[0]).replace('_probmask_csf.nii.gz', '')
        print(f"Base image: {img_base}")
        print(f"Base mask: {csf_base}")
        print(f"Cohérent: {img_base == csf_base}")

if __name__ == "__main__":
    diagnose_dataset()