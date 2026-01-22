"""
Script d'entraînement du modèle de pose estimation
"""
import os
import json

import tensorflow as tf
from tensorflow import keras
import config
import numpy as np
from train_utils import LandmarkHuberLoss, HeatmapToCoordinates, LandmarkMetrics
import csv

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
    checkpoints_path = os.path.join(model_dir, "models", "checkpoints")
    if not os.path.exists(checkpoints_path):
        return None, 0
    checkpoint_files = os.listdir(checkpoints_path)
    # find last checkpoint file of the format model_name_checkpoint_*.keras using re
    import re
    pattern = re.compile(f"model_backbone_checkpoint_(\d+).keras")
    checkpoint_nums = [int(pattern.search(f).group(1)) for f in checkpoint_files if pattern.search(f)]
    if not checkpoint_nums:
        return None, 0
    model = None
    for check in -np.sort(-np.array(checkpoint_nums)):
        try:
            model = tf.keras.models.load_model(os.path.join(checkpoints_path, f"model_backbone_checkpoint_{check}.keras"))
            break
        except:
            print("Existing checkpoints found but unable to load the model." \
            "Trying the previous one...")
    if model is None:
        return None, 0
    
    return model, check
    

def return_model_backbone(model):
    # layer_name = [layer.name for layer in model.layers][1]
    nb_layers = config.BACKBONE_TRAINING.TRAINABLE_LAYERS if config.BACKBONE_TRAINING.TRAINABLE_LAYERS else len(model.layers)
    # nb_layers = min(nb_layers, len(model.get_layer(layer_name).layers))
    nb_layers = min(nb_layers, len(model.layers))
    start_layer = len(model.layers) - nb_layers
    for l, layer in enumerate(model.layers):
        if l >= start_layer:
            if layer.name in ["batch_normalization_1", "batch_normalization"]:
                print("additional batchnorms layers are:", layer.trainable)
                
    return model


def save_final_model(model, model_path):
    heatmaps_output = model.output
    coords_output = HeatmapToCoordinates(config.HEATMAP_SIZE[0], config.INPUT_SHAPE[0], name="coords", from_pred=True)(heatmaps_output)
    inference_model = tf.keras.Model(inputs=model.input, outputs=coords_output)
    inference_model.save(model_path.replace("_backbone", ""))
    return inference_model

def fit_model(model, train, val, epochs,model_name, model_dir, training, overfit):
    if not overfit:
        history = model.fit(train.repeat(), epochs=epochs, verbose=1,
                            validation_data=val.repeat(),
                            validation_steps=val.cardinality().numpy(),
                            steps_per_epoch=train.cardinality().numpy(),
                                callbacks=create_callbacks(model_name, model_dir, backbone=False, freq_save=150*train.cardinality().numpy(),
                                                            checkpoint_suffix=training))
    else:
        history = model.fit(train.repeat(), epochs=epochs, verbose=1,
                            # validation_data=val.repeat(),
                            # validation_steps=val.cardinality().numpy(),
                            steps_per_epoch=train.cardinality().numpy(),
                                callbacks=create_callbacks(model_name, model_dir, backbone=False, freq_save=150*train.cardinality().numpy(),
                                                            checkpoint_suffix=training))
    final_model_path = os.path.join(model_dir, "models", f"{model_name}_{training}_final.keras")
    model.save(final_model_path)
    if training == "backbone":
        model = save_final_model(model, final_model_path)

    plot_path = os.path.join(os.path.join(model_dir, "logs"), f"{model_name}_{training}_loss.png")
    # if training == "head":
    history.history["loss"] = history.history["loss"][5:]
    if "val_loss" in history.history:
        history.history["val_loss"] = history.history["val_loss"][5:]
    plot_training_history(history.history, save_path=plot_path)
    return model

def save_config(model_dir):
    dict_to_save = {key: items for key, items in config.__dict__.items() if isinstance(items, (str, dict, list, tuple, int, float)) and not key.startswith("_")}
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(dict_to_save, f, indent=4)


def from_heatmaps_to_coords(heatmap, from_prediction=True, input_scale=config.INPUT_SHAPE):
    if not isinstance(heatmap, np.ndarray):
        heatmap = heatmap.numpy()
    if len(heatmap.shape) != 4:
        heatmap = heatmap[None, :, :, :]
    coords_array = np.zeros((heatmap.shape[0], 2, heatmap.shape[-1]))
    for h, heat in enumerate(heatmap):
        if from_prediction:
            heat = tf.sigmoid(heat).numpy()
        scale = input_scale[0] // config.HEATMAP_SIZE[0]
        confidence = [np.max(heat[:, :, i]).tolist() for i in range(heat.shape[-1])]
        coords = [np.array(np.where(heat[:, :, i] == confidence[i])).flatten() * scale for i in range(heat.shape[-1])]
        for a, ar in enumerate(coords):
            if len(ar) != 2:
                coords[a] = np.mean(ar.reshape(2, -1), axis=1).astype(np.uint8)

        coords_array[h, ...] = np.array(coords).T
    return coords_array, confidence


def train_model(model, 
                tf_data_set=None,
                model_name="pose_model",
                model_dir=None,
                overfit=False,
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
    save_config(model_dir)
    train, val = tf_data_set
    for training in ["head", "backbone"]:
        config_tmp = getattr(config, training.upper() + "_TRAINING")
        # with_coords = True if training == "backbone" else False
        loss_fn = LandmarkHuberLoss(heatmap_size=config.HEATMAP_SIZE[0], input_size=config.INPUT_SHAPE[0], delta=3.0, with_coords=True, huber=True)
        if not config_tmp.PERFORM:
            continue
        model_prev, prev_epochs = check_previous_checkpoints(model_dir, model_name + f"_{training}")
        if model_prev is None:
            model = return_model_backbone(model) if training == "backbone" else model
        else:
            model = model_prev

        if config_tmp.OPTIMIZER == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=config_tmp.LR)   
        elif config_tmp.OPTIMIZER == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=config_tmp.LR, momentum=config_tmp.MOMENTUM)
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[LandmarkMetrics(heatmap_size=config.HEATMAP_SIZE[0], input_size=config.INPUT_SHAPE[0])]
        )
    
        model.summary()
        epochs = config_tmp.EPOCHS - prev_epochs
        model = fit_model(model, train, val, epochs, model_name, model_dir, training, overfit)
    return model


def plot_training_history(history=None, save_path=None, csv_path=None):
    """
    Trace les courbes d'apprentissage
    
    Args:
        history: Historique de l'entraînement
        save_path: Chemin pour sauvegarder la figure (optionnel)
        
    """
    files = None
    import matplotlib.pyplot as plt
    if csv_path is not None:
        if history is not None:
            raise RuntimeError("history OR CSV file must be passed to the function.")
        files = [os.path.join(csv_path, dir) for dir in os.listdir(csv_path) if dir.endswith(".csv")]
        if save_path is None:
            save_path_list = [file.replace("csv", "png") for file in files]

    if history is not None and not isinstance(history, list):
        history = [history]  
        save_path_list = [save_path]     
    to_process = history if history is not None else files

    for data, save_path in zip(to_process, save_path_list):
        if isinstance(data, str):
            with open(data, "r") as f:
                reader = csv.DictReader(f, delimiter=",")
                history = None
                for lines in reader:
                    if history is None:
                        history = {key: [float(lines[key])] for key in lines}
                        continue
                    history = {key: history[key] + [float(lines[key])] for key in lines}
        else:
            history = data
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        to_plot = ["loss", "root_mean_squared_error"]
        prefix = ["", "val_"]
        title = ["Loss", "Metric"]
        y_labels = ["Loss (MSE + 0.01*Huber coords Loss)", "RMSE (pixels)"]

        for a, ax in enumerate(axes.flatten()):
            for i in range(len(prefix)):
                key = prefix[i] + to_plot[a]
                if key not in history:
                    continue
                ax.plot(history[key], label='Train' if i == 0 else 'Validation')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(y_labels[a])
            ax.set_title(title[a])
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Courbes d'apprentissage sauvegardées: {save_path}")
            plt.close()
    


if __name__ == "__main__":
    # Ce script est importé par main.py, mais peut aussi être testé indépendamment
    print("✅ Module train.py chargé avec succès")
    print("📝 Utilisez main.py pour lancer l'entraînement complet")
