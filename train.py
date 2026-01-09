"""
Script d'entraînement du modèle de pose estimation
"""
import os
import json

import tensorflow as tf
from tensorflow import keras
import config
import numpy as np


def create_callbacks(model_name="pose_model", model_dir=None, backbone=True, freq_save=1, checkpoint_suffix=""):
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
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
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
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(checkpoints_dir, f"model_{checkpoint_suffix}" + "_checkpoint_{epoch:03d}.keras"),
    save_freq=int(freq_save),
    )
    callbacks.append(checkpoint)

    # 5. CSVLogger - Sauvegarde les métriques dans un CSV
    csv_path = os.path.join(logs_dir, f"{model_name}_{suff}_training_log.csv")
    csv_logger = tf.keras.callbacks.CSVLogger(csv_path, append=True)
    callbacks.append(csv_logger)
    return callbacks

def check_previous_checkpoints(model_dir, model_name):
    model_name = "model_pose"
    checkpoints_path = os.path.join(model_dir, "checkpoints")
    if not os.path.exists(checkpoints_path):
        return None, 0
    checkpoint_files = os.listdir(checkpoints_path)
    # find last checkpoint file of the format model_name_checkpoint_*.keras using re
    import re
    pattern = re.compile(f"{model_name}_head_checkpoint_(\d+).keras")
    checkpoint_nums = [int(pattern.search(f).group(1)) for f in checkpoint_files if pattern.search(f)]
    if not checkpoint_nums:
        return None, 0
    model = None
    for check in -np.sort(-np.array(checkpoint_nums)):
        try:
            model = tf.keras.models.load_model(os.path.join(checkpoints_path, f"{model_name}_head_checkpoint_{check}.keras"))
            break
        except:
            print("Existing checkpoints found but unable to load the model." \
            "Trying the previous one...")
    if model is None:
        return None, 0
    
    return model, check
    
def from_heatmaps_to_coord(heatmaps):
    pass

# def loss_fn(y_true, y_pred):
#     # y_pred = tf.sigmoid(y_pred)
#     # coords_pred = from_heatmaps_to_coord(y_pred)
#     # coords_true = from_heatmaps_to_coord(y_true)

#     return (y_true, y_pred) 


def return_model_backbone(model):
    nb_layers = config.BACKBONE_TRAINING.TRAINABLE_LAYERS if config.BACKBONE_TRAINING.TRAINABLE_LAYERS else len(model.get_layer(config.BACKBONE).layers)
    nb_layers = min(nb_layers, len(model.get_layer(config.BACKBONE).layers))
    start_layer = len(model.get_layer(config.BACKBONE).layers) - nb_layers
    for l, layer in enumerate(model.get_layer(config.BACKBONE).layers):
        if l >= start_layer:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
    return model

def fit_model(model, train, val, epochs,model_name, model_dir, training):
    history = model.fit(train.repeat(), epochs=epochs, verbose=1,
                        validation_data=val.repeat(),
                        validation_steps=val.cardinality().numpy(),
                        steps_per_epoch=train.cardinality().numpy(),
                            callbacks=create_callbacks(model_name, model_dir, backbone=False, freq_save=50*train.cardinality().numpy(),
                                                        checkpoint_suffix=training))
    final_model_path = os.path.join(model_dir, "models", f"{model_name}_{training}_final.keras")
    model.save(final_model_path)

    plot_path = os.path.join(os.path.join(model_dir, "logs"), f"{model_name}_{training}_loss.png")
    if training == "head":
        history.history["loss"] = history.history["loss"][5:]
        history.history["val_loss"] = history.history["val_loss"][5:]
    plot_training_history(history.history, save_path=plot_path)

def save_config(model_dir):
    dict_to_save = {key: items for key, items in config.__dict__.items() if isinstance(items, (str, dict, list, tuple, int, float)) and not key.startswith("_")}
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(dict_to_save, f, indent=4)

def train_model(model, 
                tf_data_set=None,
                model_name="pose_model",
                model_dir=None,
                      ):
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
            gamma=2.0,
            apply_class_balancing=True 
        )
    
    save_config(model_dir)
    train, val = tf_data_set
    for training in ["head", "backbone"]:
        config_tmp = getattr(config, training.upper() + "_TRAINING")
        model_prev, prev_epochs = check_previous_checkpoints(model_dir, model_name + f"_{training}")
        if model_prev is None:
            model = return_model_backbone(model) if training == "backbone" else model
            if config_tmp.OPTIMIZER == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=config_tmp.LR)   
            elif config_tmp.OPTIMIZER == 'sgd':
                optimizer = keras.optimizers.SGD(learning_rate=config_tmp.LR, momentum=config_tmp.MOMENTUM)
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=[]
            )
        else:
            model = model_prev
        model.summary()
        epochs = config_tmp.EPOCHS - prev_epochs
        fit_model(model, train, val, epochs, model_name, model_dir, training)


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
