"""
Script d'entraînement du modèle de pose estimation
"""
import os

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
    os.path.join(model_dir, f"model_{checkpoint_suffix}" + "_checkpoint_{epoch:03d}.keras"),
    save_freq=freq_save,
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
    train, val = tf_data_set
    config_dict = config.__dict__
    histories = []
    for training in ["head", "backbone"]:
        model_prev, prev_epochs = check_previous_checkpoints(model_dir, model_name + f"_{training}")
        if model_prev is None:
            if training == "backbone":
                nb_layers = config.BACKBONE_TRAINABLE_LAYERS if config.BACKBONE_TRAINABLE_LAYERS else len(model.get_layer('MobileNetV3Small').layers)
                start_layer = len(model.get_layer('MobileNetV3Small').layers) - nb_layers
                for l, layer in enumerate(model.get_layer('MobileNetV3Small').layers):
                    if l >= start_layer:
                        if not isinstance(layer, tf.keras.layers.BatchNormalization):
                            layer.trainable = True

            history_dict = {"loss":[], 
                        "val_loss":[],
                        "lr":[]}
            if config.OPTIMIZER == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=config_dict[f"{training.upper()}_LEARNING_RATE"])   
            elif config.OPTIMIZER == 'sgd':
                optimizer = keras.optimizers.SGD(learning_rate=config_dict[f"{training.upper()}_LEARNING_RATE"], momentum=config.MOMENTUM)
            
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=[]
            )
        else:
            model = model_prev
        model.summary()
        epochs = config_dict[f"{training.upper()}_EPOCHS"] - prev_epochs
        history = model.fit(train.repeat(), epochs=epochs, verbose=1,
                                validation_data=val.repeat(),
                                validation_steps=val.cardinality().numpy(),
                                steps_per_epoch=train.cardinality().numpy(),
                                    callbacks=create_callbacks(model_name, model_dir, backbone=False, freq_save=50*train.cardinality().numpy(),
                                                                checkpoint_suffix=training))
        final_model_path = os.path.join(model_dir, "models", f"{model_name}_{training}_final.keras")
        model.save(final_model_path)
        history_dict["loss"].extend(history.history["loss"])
        history_dict["val_loss"].extend(history.history["val_loss"])
        plot_path = os.path.join(os.path.join(model_dir, "logs"), f"{model_name}_{training}_loss.png")
        plot_training_history(history_dict, save_path=plot_path)
        history_dict["lr"].extend([config.HEAD_LEARNING_RATE] * len(history.history["loss"]))
        histories.append(history_dict)
    return histories[0], histories[1]


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
