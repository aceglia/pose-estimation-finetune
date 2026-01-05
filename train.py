"""
Script d'entraînement du modèle de pose estimation
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import config


def create_callbacks(model_name="pose_model", model_dir=None, backbone=True):
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
    suff = "backbone" if backbone else "head"
    
    callbacks = []
    # 1. ModelCheckpoint - Sauvegarde le meilleur modèle
    # checkpoint_path = os.path.join(models_dir, f"{model_name}_{suff}_best.h5")
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     checkpoint_path,
    #     monitor='val_loss',
    #     save_best_only=True,
    #     save_weights_only=False,
    #     mode='min',
    #     verbose=1, 
    #     save_freq=
    # )
    # callbacks.append(checkpoint)

    # # 2. EarlyStopping - Arrête l'entraînement si pas d'amélioration
    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     patience=config.EARLY_STOPPING_PATIENCE,
    #     restore_best_weights=True,
    #     verbose=1,
    #     mode='min'
    # )
    # callbacks.append(early_stopping)
    
    # 3. ReduceLROnPlateau - Réduit le learning rate si plateau
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=config.REDUCE_LR_FACTOR,
    #     patience=config.REDUCE_LR_PATIENCE,
    #     min_lr=1e-6,
    #     verbose=1,
    #     mode='min'
    # )
    # callbacks.append(reduce_lr)

    # 4. TensorBoard - Visualisation de l'entraînement
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = os.path.join(logs_dir, f"{model_name}_{suff}_{timestamp}")
    # tensorboard = tf.keras.callbacks.TensorBoard(
    #     log_dir=log_dir,
    #     histogram_freq=1,
    #     write_graph=True,
    #     write_images=False
    # )
    # callbacks.append(tensorboard)
    
    # 5. CSVLogger - Sauvegarde les métriques dans un CSV
    csv_path = os.path.join(logs_dir, f"{model_name}_{suff}_training_log.csv")
    csv_logger = tf.keras.callbacks.CSVLogger(csv_path, append=True)
    callbacks.append(csv_logger)
    return callbacks


def train_model(model, tf_data_set=None, model_name="pose_model", model_dir=None):
    """
    Entraîne le modèle avec les callbacks appropriés
    
    Args:
        model: Modèle Keras à entraîner
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        model_name: Nom du modèle
        tf_data_set: Dataset Tensorflow pour l'entraînement (optionnel)
        
    """
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
        from_logits=True, 
        gamma=1.0,
        apply_class_balancing=True 
    )    

    train, val = tf_data_set
    history_head = {"loss":[], 
                    "val_loss":[],
                    "lr":[]}
    for learning_rate in config.HEAD_LEARNING_RATE[0:1]: 
        print("Training the head layers first...")
        if config.OPTIMIZER == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)   
        elif config.OPTIMIZER == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=config.MOMENTUM)
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[]
        )
        model.summary()
        history = model.fit(train.repeat(), epochs=400, verbose=1,
                                validation_data=val.repeat(),
                                validation_steps=val.cardinality().numpy(),
                                steps_per_epoch=train.cardinality().numpy(),
                                    callbacks=create_callbacks(model_name, model_dir, backbone=False))
        history_head["loss"].extend(history.history["loss"])
        history_head["val_loss"].extend(history.history["val_loss"])

        history_head["lr"].extend([learning_rate]* len(history.history["loss"]))


    
    # history_head = model.fit(train, steps_per_epoch=1, epochs=100, verbose=1)
    history = history_head
    # Evaluate the model on the validation set
    # create new model with trained data
    # print("Head layers trained. Now fine tuning the backbone...")
    # # fine tune backbone
    model.get_layer('MobileNetV3Small').trainable = True
    if config.OPTIMIZER == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=config.BACKBONE_LEARNING_RATE)   
    elif config.OPTIMIZER == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=config.BACKBONE_LEARNING_RATE, momentum=config.MOMENTUM)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["mae"]
    )
    model.summary()
    history_back = {"loss":[], 
                    "val_loss":[],
                    "lr":[]}
    history = model.fit(train.repeat(), epochs=1000, verbose=1,
                                validation_data=val.repeat(),
                                validation_steps=val.cardinality().numpy(),
                                steps_per_epoch=train.cardinality().numpy(),
                         callbacks=create_callbacks(model_name, model_dir, backbone=True))
    history_back["loss"].extend(history.history["loss"])
    history_back["val_loss"].extend(history.history["val_loss"])

    history_back["lr"].extend([learning_rate]* len(history.history["loss"]))
    # history_head = model.fit(train, steps_per_epoch=1, epochs=100, verbose=1)
    # print(history.history)
    return history_back, history_head


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
    models_dir = config.MODELS_DIR if model_dir is None else os.path.join(model_dir, "models")
    
    final_model_path = os.path.join(models_dir, f"{model_name}_final.keras")
    model.save(final_model_path)
    print(f"\nModel save in keras format: {final_model_path}")
    
    saved_model_dir = os.path.join(models_dir, f"{model_name}_saved_model")
    tf.saved_model.save(model, saved_model_dir)
    print(f"Model save in SavedModel format: {saved_model_dir}")
    
    return final_model_path, saved_model_dir

def plot_training_history(history, save_path=None):
    """
    Trace les courbes d'apprentissage
    
    Args:
        history: Historique de l'entraînement
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 1, figsize=(15, 5))
    
    # Loss
    axes.plot(history['loss'], label='Train Loss')
    axes.plot(history['val_loss'], label='Val Loss')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss (MSE)')
    axes.set_title('Courbe de Loss')
    axes.legend()
    axes.grid(True)
    
    # MAE
    # axes[1].plot(history.history['mae'], label='Train MAE')
    # axes[1].plot(history.history['val_mae'], label='Val MAE')
    # axes[1].set_xlabel('Epoch')
    # axes[1].set_ylabel('MAE')
    # axes[1].set_title('Courbe de MAE')
    # axes[1].legend()
    # axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Courbes d'apprentissage sauvegardées: {save_path}")
        plt.close()
    


if __name__ == "__main__":
    # Ce script est importé par main.py, mais peut aussi être testé indépendamment
    print("✅ Module train.py chargé avec succès")
    print("📝 Utilisez main.py pour lancer l'entraînement complet")
