# =============================================================================
# DEBUG COMPLET DU DATASET KAGGLE
# =============================================================================

import os
import glob
import numpy as np
from pathlib import Path

def debug_dataset_complete():
    """Debug complet du dataset pour identifier le problème"""
    
    print("=== DEBUG COMPLET DU DATASET ===")
    
    # 1. Vérifier l'existence du dataset
    data_path = '/kaggle/input/3dbraintissuesegmentation'
    print(f"1. Chemin dataset: {data_path}")
    print(f"   Existe: {os.path.exists(data_path)}")
    
    if not os.path.exists(data_path):
        print("❌ Dataset non trouvé! Vérifiez l'attachement du dataset.")
        return
    
    # 2. Explorer la structure complète
    print(f"\n2. Structure complète du dataset:")
    for root, dirs, files in os.walk(data_path):
        level = root.replace(data_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... et {len(files) - 5} autres fichiers")
    
    # 3. Chercher tous les fichiers .nii.gz
    print(f"\n3. Recherche de tous les fichiers .nii.gz:")
    all_nii_files = glob.glob(os.path.join(data_path, "**/*.nii.gz"), recursive=True)
    print(f"   Total fichiers .nii.gz trouvés: {len(all_nii_files)}")
    
    if all_nii_files:
        print(f"   Exemples:")
        for i, file in enumerate(all_nii_files[:10]):
            print(f"     {i+1}. {file}")
        if len(all_nii_files) > 10:
            print(f"     ... et {len(all_nii_files) - 10} autres")
    
    # 4. Analyser les patterns de noms
    print(f"\n4. Analyse des patterns de noms:")
    
    # Images
    img_files = [f for f in all_nii_files if "_img.nii.gz" in f]
    print(f"   Fichiers d'images (_img.nii.gz): {len(img_files)}")
    
    # Masks CSF
    csf_files = [f for f in all_nii_files if "probmask_csf" in f]
    print(f"   Fichiers masks CSF: {len(csf_files)}")
    
    # Masks Gray Matter
    gm_files = [f for f in all_nii_files if "probmask_graymatter" in f]
    print(f"   Fichiers masks Gray Matter: {len(gm_files)}")
    
    # Masks White Matter
    wm_files = [f for f in all_nii_files if "probmask_whitematter" in f]
    print(f"   Fichiers masks White Matter: {len(wm_files)}")
    
    # 5. Vérifier les correspondances
    print(f"\n5. Vérification des correspondances:")
    
    if img_files and csf_files and gm_files and wm_files:
        # Extraire les noms de base
        img_bases = set()
        for img_file in img_files:
            base = os.path.basename(img_file).replace("_img.nii.gz", "")
            img_bases.add(base)
        
        csf_bases = set()
        for csf_file in csf_files:
            base = os.path.basename(csf_file).replace("_probmask_csf.nii.gz", "")
            csf_bases.add(base)
        
        gm_bases = set()
        for gm_file in gm_files:
            base = os.path.basename(gm_file).replace("_probmask_graymatter.nii.gz", "")
            gm_bases.add(base)
        
        wm_bases = set()
        for wm_file in wm_files:
            base = os.path.basename(wm_file).replace("_probmask_whitematter.nii.gz", "")
            wm_bases.add(base)
        
        # Trouver les intersections
        common_bases = img_bases.intersection(csf_bases).intersection(gm_bases).intersection(wm_bases)
        print(f"   Noms de base communs: {len(common_bases)}")
        
        if common_bases:
            print(f"   Exemples de noms communs:")
            for i, base in enumerate(list(common_bases)[:5]):
                print(f"     {i+1}. {base}")
        
        # Vérifier les différences
        print(f"   Images sans masks: {len(img_bases - common_bases)}")
        print(f"   Masks sans images: {len((csf_bases | gm_bases | wm_bases) - common_bases)}")
    
    # 6. Tester le chargement d'un fichier
    print(f"\n6. Test de chargement d'un fichier:")
    
    if img_files:
        test_file = img_files[0]
        print(f"   Test avec: {test_file}")
        
        try:
            import nibabel as nib
            nifti = nib.load(test_file)
            data = nifti.get_fdata()
            print(f"   ✅ Chargement réussi!")
            print(f"   Shape: {data.shape}")
            print(f"   Type: {data.dtype}")
            print(f"   Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
        except Exception as e:
            print(f"   ❌ Erreur de chargement: {e}")
    
    # 7. Suggestions de correction
    print(f"\n7. Suggestions de correction:")
    
    if len(all_nii_files) == 0:
        print("   ❌ Aucun fichier .nii.gz trouvé")
        print("   → Vérifiez que le dataset est bien attaché")
        print("   → Vérifiez le nom exact du dataset")
    elif len(common_bases) == 0:
        print("   ❌ Aucune correspondance image-mask trouvée")
        print("   → Vérifiez les patterns de noms de fichiers")
        print("   → Vérifiez la structure des dossiers")
    else:
        print(f"   ✅ {len(common_bases)} paires image-mask trouvées")
        print("   → Le dataset semble correct, problème dans le code de chargement")

if __name__ == "__main__":
    debug_dataset_complete()