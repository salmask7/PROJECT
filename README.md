# Segmentation Cérébrale 3D avec U-Net

Ce projet implémente un pipeline complet pour la segmentation automatique des tissus cérébraux (CSF, Gray Matter, White Matter) à partir d'images IRM 3D en utilisant un modèle U-Net 3D.

## 🏗️ Architecture du Projet

```
├── config.py              # Configuration globale
├── utils.py               # Utilitaires communs (métriques, visualisation)
├── preprocessing.py       # Prétraitement des données
├── augmentation.py        # Augmentation des données
├── training.py           # Entraînement du modèle
├── main.py               # Script principal
└── README.md             # Documentation
```

## 🚀 Installation

```bash
# Installer les dépendances
pip install nibabel tensorflow-addons scikit-image scikit-learn matplotlib seaborn

# Ou utiliser requirements.txt (à créer)
pip install -r requirements.txt
```

## 📊 Dataset

Le projet est conçu pour fonctionner avec le dataset "3D Brain Tissue Segmentation" de Kaggle. Structure attendue:

```
/kaggle/input/3dbraintissuesegmentation/
├── train/
│   ├── image/
│   │   ├── subject001_img.nii
│   │   └── ...
│   └── mask/
│       ├── subject001_probmask_csf.nii
│       ├── subject001_probmask_graymatter.nii
│       └── subject001_probmask_whitematter.nii
├── valid/
│   └── ...
└── test/
    └── ...
```

## 🔧 Configuration

Modifiez `config.py` pour ajuster les paramètres:

```python
class Config:
    # Chemins
    DATA_PATH = '/kaggle/input/3dbraintissuesegmentation'
    
    # Paramètres du modèle
    IMG_SIZE = (128, 128, 128)
    BATCH_SIZE = 2
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    # Classes
    N_CLASSES = 4  # Background + 3 tissus
    CLASS_NAMES = ['Background', 'CSF', 'Gray Matter', 'White Matter']
    
    # Augmentation
    NUM_AUGMENTATIONS = 3
    ROTATION_RANGE = 15
    ZOOM_RANGE = 0.1
```

## 🎯 Utilisation

### Pipeline Complet

```bash
# Exécuter tout le pipeline
python main.py

# Ou spécifier une étape
python main.py --step preprocessing
python main.py --step augmentation
python main.py --step training
```

### Étapes Individuelles

#### 1. Prétraitement

```python
from preprocessing import main as preprocess_main
splits = preprocess_main()
```

**Fonctionnalités:**
- Chargement des fichiers NIFTI
- Normalisation des images
- Redimensionnement à (128, 128, 128)
- Création des masks multiclasses
- Division train/val/test

#### 2. Augmentation

```python
from augmentation import main as augment_main
augmented_splits = augment_main()
```

**Techniques d'augmentation:**
- Rotation 3D
- Zoom 3D
- Flip 3D
- Bruit gaussien
- Modification de luminosité/contraste
- Déformation élastique
- Combinaisons aléatoires

#### 3. Entraînement

```python
from training import main as train_main
model, history = train_main()
```

**Architecture U-Net 3D:**
- Encoder avec 4 niveaux
- Decoder avec skip connections
- Dropout pour réduire l'overfitting
- Batch normalization
- Loss combinée: Cross-entropy + Dice

### Continuation d'Entraînement

```bash
# Continuer l'entraînement d'un modèle existant
python main.py --continue-training ./models/best_model.keras --epochs 20
```

## 📈 Métriques

Le modèle utilise plusieurs métriques:

- **Accuracy**: Précision globale
- **Dice Coefficient**: Métrique principale pour la segmentation
- **Precision/Recall/F1**: Par classe et globale
- **Loss combinée**: Cross-entropy + Dice loss

## 📁 Structure des Sorties

```
output/
├── training_history.png      # Graphiques d'entraînement
├── test_predictions.png      # Visualisations des prédictions
├── training_log.csv          # Log d'entraînement
└── test_predictions.npy      # Prédictions sur test

preprocessed_data/
├── X_train.npy              # Images d'entraînement
├── y_train.npy              # Masks d'entraînement
├── metadata_train.json      # Métadonnées
└── ...

augmented_data/
├── X_train_augmented.npy    # Images augmentées
├── y_train_augmented.npy    # Masks augmentés
└── ...

models/
├── best_model.keras         # Meilleur modèle
├── final_model.keras        # Modèle final
└── model_weights.h5         # Poids du modèle
```

## 🔍 Visualisation

Le projet inclut des fonctions de visualisation:

```python
from utils import plot_training_history, visualize_predictions

# Historique d'entraînement
plot_training_history(history)

# Prédictions
visualize_predictions(model, X_test, y_test, config)
```

## ⚙️ Personnalisation

### Ajouter de Nouvelles Techniques d'Augmentation

```python
def custom_augmentation(image, mask):
    # Votre technique personnalisée
    return augmented_image, augmented_mask
```

### Modifier l'Architecture du Modèle

```python
def custom_unet_3d(input_shape, n_classes):
    # Votre architecture personnalisée
    return model
```

### Ajuster les Métriques

```python
def custom_loss(y_true, y_pred):
    # Votre fonction de loss personnalisée
    return loss_value
```

## 🐛 Dépannage

### Problèmes Courants

1. **Mémoire insuffisante**: Réduisez `BATCH_SIZE` et `IMG_SIZE`
2. **Données non trouvées**: Vérifiez `DATA_PATH` dans `config.py`
3. **Formats de fichiers**: Assurez-vous que les fichiers sont en format NIFTI (.nii)

### Debug

```python
# Activer le debug pour la structure du dataset
from preprocessing import debug_dataset_structure
debug_dataset_structure()
```

## 📚 Références

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650)
- [Kaggle 3D Brain Tissue Segmentation Dataset](https://www.kaggle.com/datasets/...)

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à:
- Signaler des bugs
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation
- Optimiser les performances

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.