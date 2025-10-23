# =============================================================================
# ENTRAÎNEMENT DU MODÈLE U-Net 3D POUR LA SEGMENTATION CÉRÉBRALE
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from config import config
from utils import (
    dice_coefficient, dice_loss, combined_loss, 
    plot_training_history, visualize_predictions, 
    calculate_metrics, print_metrics, save_predictions
)
from augmentation import load_augmented_data

def conv_block_3d(inputs, filters, kernel_size=3, padding='same', activation='relu', dropout=0.1):
    """Bloc de convolution 3D avec normalisation et dropout"""
    x = layers.Conv3D(filters, kernel_size, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x

def unet_3d(input_shape, n_classes):
    """Architecture U-Net 3D pour la segmentation"""
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block_3d(inputs, 32, dropout=0.1)
    c1 = conv_block_3d(c1, 32, dropout=0.1)
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)

    c2 = conv_block_3d(p1, 64, dropout=0.1)
    c2 = conv_block_3d(c2, 64, dropout=0.1)
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)

    c3 = conv_block_3d(p2, 128, dropout=0.1)
    c3 = conv_block_3d(c3, 128, dropout=0.1)
    p3 = layers.MaxPooling3D((2, 2, 2))(c3)

    # Bottleneck
    c4 = conv_block_3d(p3, 256, dropout=0.2)
    c4 = conv_block_3d(c4, 256, dropout=0.2)

    # Decoder
    u5 = layers.UpSampling3D((2, 2, 2))(c4)
    u5 = layers.Concatenate()([u5, c3])
    c5 = conv_block_3d(u5, 128, dropout=0.1)
    c5 = conv_block_3d(c5, 128, dropout=0.1)

    u6 = layers.UpSampling3D((2, 2, 2))(c5)
    u6 = layers.Concatenate()([u6, c2])
    c6 = conv_block_3d(u6, 64, dropout=0.1)
    c6 = conv_block_3d(c6, 64, dropout=0.1)

    u7 = layers.UpSampling3D((2, 2, 2))(c6)
    u7 = layers.Concatenate()([u7, c1])
    c7 = conv_block_3d(u7, 32, dropout=0.1)
    c7 = conv_block_3d(c7, 32, dropout=0.1)

    # Output
    outputs = layers.Conv3D(n_classes, 1, activation='softmax')(c7)

    model = keras.Model(inputs, outputs)
    return model

def create_callbacks(model_save_path, config):
    """Crée les callbacks pour l'entraînement"""
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=config.PATIENCE, 
            restore_best_weights=True, 
            monitor='val_dice_coefficient', 
            mode='max',
            min_delta=config.MIN_DELTA
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=config.REDUCE_LR_FACTOR, 
            patience=config.REDUCE_LR_PATIENCE, 
            monitor='val_loss',
            min_delta=config.MIN_DELTA
        ),
        keras.callbacks.ModelCheckpoint(
            model_save_path, 
            save_best_only=True, 
            monitor='val_dice_coefficient', 
            mode='max',
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(config.OUTPUT_PATH, 'training_log.csv'),
            append=False
        )
    ]
    return callbacks

def prepare_data_for_training(splits):
    """Prépare les données pour l'entraînement"""
    print("=== PRÉPARATION DES DONNÉES POUR L'ENTRAÎNEMENT ===")
    
    # Vérifier que nous avons des données d'entraînement
    if len(splits['train']['X']) == 0:
        print("❌ Aucune donnée d'entraînement trouvée!")
        return None, None, None, None
    
    X_train = splits['train']['X']
    y_train = splits['train']['y']
    X_val = splits['val']['X'] if len(splits['val']['X']) > 0 else None
    y_val = splits['val']['y'] if len(splits['val']['y']) > 0 else None
    
    print(f"Données d'entraînement: {X_train.shape}")
    if X_val is not None:
        print(f"Données de validation: {X_val.shape}")
    
    # Ajouter une dimension de canal pour les images
    X_train = X_train[..., np.newaxis]
    if X_val is not None:
        X_val = X_val[..., np.newaxis]
    
    # Convertir les masks en one-hot encoding
    y_train_oh = tf.keras.utils.to_categorical(y_train, config.N_CLASSES)
    if y_val is not None:
        y_val_oh = tf.keras.utils.to_categorical(y_val, config.N_CLASSES)
    else:
        y_val_oh = None
    
    print(f"Forme finale X_train: {X_train.shape}")
    print(f"Forme finale y_train: {y_train_oh.shape}")
    if X_val is not None:
        print(f"Forme finale X_val: {X_val.shape}")
        print(f"Forme finale y_val: {y_val_oh.shape}")
    
    return X_train, y_train_oh, X_val, y_val_oh

def create_model(input_shape, n_classes):
    """Crée et compile le modèle"""
    print("=== CRÉATION DU MODÈLE ===")
    
    model = unet_3d(input_shape, n_classes)
    
    # Compiler le modèle
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=combined_loss,
        metrics=['accuracy', dice_coefficient]
    )
    
    # Afficher l'architecture
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, config):
    """Entraîne le modèle"""
    print("=== DÉMARRAGE DE L'ENTRAÎNEMENT ===")
    
    # Créer les callbacks
    model_save_path = os.path.join(config.MODEL_PATH, 'best_model.keras')
    callbacks = create_callbacks(model_save_path, config)
    
    # Préparer les données de validation
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
        print("Utilisation des données de validation")
    else:
        print("Pas de données de validation, utilisation d'un split automatique")
    
    # Entraîner le modèle
    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        validation_split=0.2 if validation_data is None else None,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("=== ENTRAÎNEMENT TERMINÉ ===")
    return history

def evaluate_model(model, X_test, y_test, config):
    """Évalue le modèle sur les données de test"""
    if len(X_test) == 0:
        print("❌ Aucune donnée de test disponible")
        return None
    
    print("=== ÉVALUATION DU MODÈLE ===")
    
    # Prédictions
    X_test_expanded = X_test[..., np.newaxis]
    predictions = model.predict(X_test_expanded, batch_size=config.BATCH_SIZE)
    pred_classes = np.argmax(predictions, axis=-1)
    
    # Calculer les métriques
    metrics = calculate_metrics(y_test, pred_classes, config)
    print_metrics(metrics)
    
    # Sauvegarder les prédictions
    predictions_path = os.path.join(config.OUTPUT_PATH, 'test_predictions.npy')
    save_predictions(predictions, y_test, config, predictions_path)
    
    return metrics, predictions, pred_classes

def save_model_and_history(model, history, config):
    """Sauvegarde le modèle et l'historique"""
    print("=== SAUVEGARDE DU MODÈLE ET DE L'HISTORIQUE ===")
    
    # Sauvegarder le modèle complet
    model_path = os.path.join(config.MODEL_PATH, 'final_model.keras')
    model.save(model_path)
    print(f"Modèle sauvegardé: {model_path}")
    
    # Sauvegarder les poids
    weights_path = os.path.join(config.MODEL_PATH, 'model_weights.h5')
    model.save_weights(weights_path)
    print(f"Poids sauvegardés: {weights_path}")
    
    # Sauvegarder l'historique
    history_path = os.path.join(config.OUTPUT_PATH, 'training_history.npy')
    np.save(history_path, history.history)
    print(f"Historique sauvegardé: {history_path}")
    
    # Sauvegarder les graphiques
    plot_path = os.path.join(config.OUTPUT_PATH, 'training_history.png')
    plot_training_history(history, save_path=plot_path)

def load_trained_model(model_path, config):
    """Charge un modèle entraîné"""
    print(f"=== CHARGEMENT DU MODÈLE: {model_path} ===")
    
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={
                'dice_coefficient': dice_coefficient,
                'dice_loss': dice_loss,
                'combined_loss': combined_loss
            }
        )
        print("✅ Modèle chargé avec succès")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

def main():
    """Fonction principale d'entraînement"""
    print("=== DÉMARRAGE DE L'ENTRAÎNEMENT ===")
    
    # Vérifier la disponibilité du GPU
    print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Charger les données augmentées
    splits = load_augmented_data()
    
    if not any(len(split_data['X']) > 0 for split_data in splits.values()):
        print("❌ Aucune donnée augmentée trouvée. Exécutez d'abord augmentation.py")
        return None
    
    # Préparer les données
    X_train, y_train, X_val, y_val = prepare_data_for_training(splits)
    
    if X_train is None:
        print("❌ Échec de la préparation des données")
        return None
    
    # Créer le modèle
    input_shape = config.IMG_SIZE + (1,)
    model = create_model(input_shape, config.N_CLASSES)
    
    # Entraîner le modèle
    history = train_model(model, X_train, y_train, X_val, y_val, config)
    
    # Sauvegarder le modèle et l'historique
    save_model_and_history(model, history, config)
    
    # Évaluer sur les données de test
    if len(splits['test']['X']) > 0:
        metrics, predictions, pred_classes = evaluate_model(
            model, splits['test']['X'], splits['test']['y'], config
        )
        
        # Visualiser quelques prédictions
        if predictions is not None:
            print("\n=== VISUALISATION DES PRÉDICTIONS ===")
            visualize_predictions(
                model, splits['test']['X'][:3], splits['test']['y'][:3], 
                config, save_path=os.path.join(config.OUTPUT_PATH, 'test_predictions.png')
            )
    
    print("=== ENTRAÎNEMENT TERMINÉ ===")
    return model, history

def continue_training(model_path, additional_epochs=10):
    """Continue l'entraînement d'un modèle existant"""
    print(f"=== CONTINUATION DE L'ENTRAÎNEMENT ===")
    
    # Charger le modèle
    model = load_trained_model(model_path, config)
    if model is None:
        return None, None
    
    # Charger les données
    splits = load_augmented_data()
    X_train, y_train, X_val, y_val = prepare_data_for_training(splits)
    
    if X_train is None:
        return None, None
    
    # Continuer l'entraînement
    print(f"Continuation pour {additional_epochs} époques supplémentaires...")
    
    # Créer de nouveaux callbacks
    model_save_path = os.path.join(config.MODEL_PATH, 'continued_model.keras')
    callbacks = create_callbacks(model_save_path, config)
    
    # Entraîner
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        validation_split=0.2 if X_val is None else None,
        batch_size=config.BATCH_SIZE,
        epochs=additional_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Sauvegarder
    save_model_and_history(model, history, config)
    
    print("=== CONTINUATION TERMINÉE ===")
    return model, history

if __name__ == "__main__":
    model, history = main()