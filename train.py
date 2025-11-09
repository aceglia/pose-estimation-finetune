"""
Script d'entraînement du modèle de pose estimation
"""
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)
import config


def create_callbacks(model_name="pose_model", model_dir=None):
    """
    Crée les callbacks pour l'entraînement
    
    Args:
        model_name: Nom du modèle pour sauvegarder les fichiers
        model_dir: Dossier racine du modèle (si None, utilise config)
    
    Returns:
        callbacks: Liste des callbacks
    """
    # Déterminer les dossiers
    models_dir = config.MODELS_DIR if model_dir is None else os.path.join(model_dir, "models")
    logs_dir = config.LOGS_DIR if model_dir is None else os.path.join(model_dir, "logs")
    
    callbacks = []
    
    # 1. ModelCheckpoint - Sauvegarde le meilleur modèle
    checkpoint_path = os.path.join(models_dir, f"{model_name}_best.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # 2. EarlyStopping - Arrête l'entraînement si pas d'amélioration
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    callbacks.append(early_stopping)
    
    # 3. ReduceLROnPlateau - Réduit le learning rate si plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    )
    callbacks.append(reduce_lr)
    
    # 4. TensorBoard - Visualisation de l'entraînement
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(logs_dir, f"{model_name}_{timestamp}")
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )
    callbacks.append(tensorboard)
    
    # 5. CSVLogger - Sauvegarde les métriques dans un CSV
    csv_path = os.path.join(logs_dir, f"{model_name}_training_log.csv")
    csv_logger = CSVLogger(csv_path, append=True)
    callbacks.append(csv_logger)
    
    print(f"\n📋 Callbacks configurés:")
    print(f"   - Meilleur modèle sauvegardé dans: {checkpoint_path}")
    print(f"   - Logs TensorBoard dans: {log_dir}")
    print(f"   - Logs CSV dans: {csv_path}")
    
    return callbacks


def create_data_augmentation():
    """
    Crée un pipeline d'augmentation de données (optionnel)
    
    Returns:
        augmentation: ImageDataGenerator pour l'augmentation
    """
    if not config.USE_AUGMENTATION:
        return None
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=config.AUGMENTATION_CONFIG['rotation_range'],
        width_shift_range=config.AUGMENTATION_CONFIG['width_shift_range'],
        height_shift_range=config.AUGMENTATION_CONFIG['height_shift_range'],
        zoom_range=config.AUGMENTATION_CONFIG['zoom_range'],
        horizontal_flip=config.AUGMENTATION_CONFIG['horizontal_flip'],
        fill_mode=config.AUGMENTATION_CONFIG['fill_mode']
    )
    
    print("\n🔄 Augmentation de données activée")
    return datagen


def train_model(model, X_train, y_train, X_val, y_val, model_name="pose_model", model_dir=None):
    """
    Entraîne le modèle avec les callbacks appropriés
    
    Args:
        model: Modèle Keras à entraîner
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        model_name: Nom du modèle
        model_dir: Dossier racine du modèle
    
    Returns:
        history: Historique de l'entraînement
    """
    print("=" * 60)
    print("🏋️  ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60)
    
    # Créer les callbacks
    callbacks = create_callbacks(model_name, model_dir)
    
    # Compiler le modèle
    model.compile(
        optimizer=config.OPTIMIZER,
        loss='mse',
        metrics=['mae']
    )
    
    # Créer l'augmentation de données si activée
    if config.USE_AUGMENTATION:
        augmentation = create_data_augmentation()
        if augmentation is not None:
            # Entraîner avec augmentation
            train_generator = augmentation.flow(X_train, y_train, batch_size=config.BATCH_SIZE)
            history = model.fit(
                train_generator,
                validation_data=(X_val, y_val),
                epochs=config.EPOCHS,
                callbacks=callbacks,
                verbose=config.VERBOSE,
                steps_per_epoch=len(X_train) // config.BATCH_SIZE
            )
        else:
            # Entraîner sans augmentation
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=config.BATCH_SIZE,
                epochs=config.EPOCHS,
                callbacks=callbacks,
                verbose=config.VERBOSE
            )
    else:
        # Entraîner sans augmentation
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            callbacks=callbacks,
            verbose=config.VERBOSE
        )
    
    return history


def save_final_model(model, model_name="pose_model", model_dir=None):
    """
    Sauvegarde le modèle final
    
    Args:
        model: Modèle Keras entraîné
        model_name: Nom du modèle
        model_dir: Dossier racine du modèle
    
    Returns:
        tuple: (final_model_path, saved_model_dir)
    """
    # Déterminer le dossier des modèles
    models_dir = config.MODELS_DIR if model_dir is None else os.path.join(model_dir, "models")
    
    # Sauvegarder le modèle complet (architecture + poids)
    final_model_path = os.path.join(models_dir, f"{model_name}_final.h5")
    model.save(final_model_path)
    print(f"\n💾 Modèle final sauvegardé: {final_model_path}")
    
    # Sauvegarder aussi au format SavedModel (pour TFLite)
    saved_model_dir = os.path.join(models_dir, f"{model_name}_saved_model")
    model.save(saved_model_dir, save_format='tf')
    print(f"💾 SavedModel sauvegardé: {saved_model_dir}")
    
    return final_model_path, saved_model_dir


def evaluate_model(model, X_val, y_val):
    """
    Évalue le modèle sur le set de validation
    
    Args:
        model: Modèle Keras entraîné
        X_val: Images de validation
        y_val: Heatmaps de validation
    
    Returns:
        metrics: Dictionnaire des métriques
    """
    print("\n📊 Évaluation du modèle sur le set de validation...")
    
    results = model.evaluate(X_val, y_val, verbose=0)
    
    metrics = {
        'loss': results[0],
        'mae': results[1]
    }
    
    print(f"   - Loss (MSE): {metrics['loss']:.6f}")
    print(f"   - MAE: {metrics['mae']:.6f}")
    
    return metrics


def plot_training_history(history, save_path=None):
    """
    Trace les courbes d'apprentissage
    
    Args:
        history: Historique de l'entraînement
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Courbe de Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE
        axes[1].plot(history.history['mae'], label='Train MAE')
        axes[1].plot(history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Courbe de MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Courbes d'apprentissage sauvegardées: {save_path}")
        
        plt.show()
        
    except ImportError:
        print("\n⚠️  Matplotlib non installé, impossible de tracer les courbes")


if __name__ == "__main__":
    # Ce script est importé par main.py, mais peut aussi être testé indépendamment
    print("✅ Module train.py chargé avec succès")
    print("📝 Utilisez main.py pour lancer l'entraînement complet")
