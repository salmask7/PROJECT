# =============================================================================
# CONFIGURATION POUR LA SEGMENTATION CÉRÉBRALE 3D
# =============================================================================

import os

class Config:
    def __init__(self):
        # Dataset paths
        self.DATA_PATH = '/kaggle/input/3dbraintissuesegmentation'
        self.OUTPUT_PATH = './output'
        self.PREPROCESSED_PATH = './preprocessed_data'
        self.AUGMENTED_PATH = './augmented_data'
        self.MODEL_PATH = './models'
        
        # Créer les dossiers de sortie
        for path in [self.OUTPUT_PATH, self.PREPROCESSED_PATH, 
                    self.AUGMENTED_PATH, self.MODEL_PATH]:
            os.makedirs(path, exist_ok=True)

        # Model parameters
        self.IMG_SIZE = (128, 128, 128)  # Taille réduite pour Kaggle
        self.BATCH_SIZE = 2  # Petit batch pour mémoire limitée
        self.EPOCHS = 50
        self.LEARNING_RATE = 1e-4

        # Classes
        self.N_CLASSES = 4  # Background + 3 tissus
        self.CLASS_NAMES = ['Background', 'CSF', 'Gray Matter', 'White Matter']
        self.CLASS_COLORS = ['black', 'red', 'green', 'blue']

        # Augmentation parameters
        self.ROTATION_RANGE = 15
        self.ZOOM_RANGE = 0.1
        self.NUM_AUGMENTATIONS = 3  # Nombre d'augmentations par sample

        # Preprocessing parameters
        self.MAX_SAMPLES = None  # None pour charger tous les échantillons
        self.TRAIN_SPLIT = 0.7
        self.VAL_SPLIT = 0.15
        self.TEST_SPLIT = 0.15

        # Training parameters
        self.PATIENCE = 10
        self.MIN_DELTA = 1e-4
        self.REDUCE_LR_PATIENCE = 5
        self.REDUCE_LR_FACTOR = 0.5

    def get_file_paths(self, split='train'):
        """Retourne les chemins des fichiers pour un split donné"""
        split_path = os.path.join(self.DATA_PATH, split)
        image_dir = os.path.join(split_path, 'image')
        mask_dir = os.path.join(split_path, 'mask')
        return split_path, image_dir, mask_dir

    def print_config(self):
        """Affiche la configuration"""
        print("=== CONFIGURATION ===")
        print(f"Data path: {self.DATA_PATH}")
        print(f"Image size: {self.IMG_SIZE}")
        print(f"Batch size: {self.BATCH_SIZE}")
        print(f"Epochs: {self.EPOCHS}")
        print(f"Learning rate: {self.LEARNING_RATE}")
        print(f"Number of classes: {self.N_CLASSES}")
        print(f"Class names: {self.CLASS_NAMES}")
        print(f"Augmentation per sample: {self.NUM_AUGMENTATIONS}")
        print("=====================")

# Instance globale de configuration
config = Config()