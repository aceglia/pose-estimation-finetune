"""
Configuration pour le fine-tuning de pose estimation
"""
import os
from datetime import datetime
import tensorflow as tf 

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prefix = "/mnt/c" if os.name == "posix" else "C:\\"
LABELED_DATA_DIR = os.path.join(prefix, "Users", "Usager", "Documents", "Amedeo", "PFE", "PFE", "JambeGaucheLabeled", "1-TouteLabeledGauche")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
backbone_mapping = {
        "MobileNetV2": "MNv2",
        "MobileNetV3Small": "MNv3S",
        "MobileNetV3Large": "MNv3L",
        "EfficientNetLite0": "ENL0",
        "EfficientNetLite1": "ENL1",
        "EfficientNetLite2": "ENL2",
        "EfficientNetLite3": "ENL3",
        "EfficientNetLite4": "ENL4",
        "EfficientNetB0": "ENB0",
        "EfficientNetB1": "ENB1",
        "EfficientNetB2": "ENB2",
        "EfficientNetB3": "ENB3",
        "EfficientNetV2B0": "ENV2B0",
        "EfficientNetV2B1": "ENV2B1",
        "EfficientNetV2B2": "ENV2B2",
        "EfficientNetV2B3": "ENV2B3",
    }
# Fonction pour créer le nom du dossier modèle
def get_model_folder_name(backbone=None, timestamp=None):
    """Génère le nom du dossier pour un modèle spécifique"""
    if backbone is None:
        backbone = BACKBONE
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Nettoyer le nom du backbone (enlever caractères spéciaux)

    clean_backbone = backbone_mapping.get(backbone, backbone)

    return f"{clean_backbone}_{timestamp}"

class TrainingConfig:
    def __init__(self, dict):
        self._init_dict = dict
        self.SCHEDULE_LR = None
        self.LR = None

        self._init_from_dict()

    def _init_from_dict(self):
        for key, items in self._init_dict.items():
            if key == "SCHEDULE_LR" and items is not None:
                name = items.pop("name")
                self.SCHEDULE_LR = get_schedule(name, items)
                continue
            setattr(self, key, items)
        if self.LR is not None and self.SCHEDULE_LR is not None:
            print("Warning: learning rate and schedule learning rate were both asked: learning rate will be ignore.")
        if self.SCHEDULE_LR is not None:
            setattr(self, "LR", self.SCHEDULE_LR)


def get_schedule(name, kwargs):
    if name == "cosine_decay":
        return tf.keras.optimizers.schedules.CosineDecay(**kwargs)
    if name == "exponential_decay":
        return tf.keras.optimizers.schedules.ExponentialDecay(**kwargs)
    if name == "inverse_time_decay":
        return tf.keras.optimizers.schedules.InverseTimeDecay(**kwargs)
    if name == "piecewise_constant_decay":
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(**kwargs)
    if name == "polynomial_decay":
        return tf.keras.optimizers.schedules.PolynomialDecay(**kwargs)


BACKBONE_TRAINING_DICT = {
    "PERFORM": True, 
    "LR": 1e-3,
    "TRAINABLE_LAYERS" : None,
    "EPOCHS" : 300,
    "OPTIMIZER" : "adam",
    "MOMENTUM" : 0.9, 
    # "SCHEDULE_LR": {
    #     "name": "piecewise_constant_decay",
    #     # "initial_learning_rate": 1e-3, 
    #     # "decay_steps": 1000,
    #     # "decay_rate":0.9, 
    #     # "staircase":True, 
    #     "boundaries": [1500, 10000, 20000], 
    #     "values":[5e-3, 1e-3, 1e-4, 1e-5]
    # }
}

HEAD_TRAINING_DICT = {
    "PERFORM": False,
    "LR": 1e-3,
    "TRAINABLE_LAYERS" : -1,
    "EPOCHS" : 500,
    "OPTIMIZER" : "adam",
    "MOMENTUM" : 0.9, 
    # "SCHEDULE_LR": {
    #     "name": "piecewise_constant_decay",
    #     # "initial_learning_rate": 1e-3, 
    #     # "decay_steps": 1000,
    #     # "decay_rate":0.9, 
    #     # "staircase":True, 
    #     "boundaries": [600, 4000, 9000], 
    #     "values":[1e-3, 1e-2, 1e-3, 5e-4]
    # }
}

BACKBONE_TRAINING = TrainingConfig(BACKBONE_TRAINING_DICT)
HEAD_TRAINING = TrainingConfig(HEAD_TRAINING_DICT)

# Dossier modèle actuel (sera défini dynamiquement)
MODEL_DIR = None
MODELS_DIR = None
LOGS_DIR = None
VIDEOS_DIR = LABELED_DATA_DIR
def setup_model_directories(model_folder_name=None):
    """Configure les dossiers pour un modèle spécifique"""
    global MODEL_DIR, MODELS_DIR, LOGS_DIR, VIDEOS_DIR

    if model_folder_name is None:
        model_folder_name = get_model_folder_name()

    MODEL_DIR = os.path.join(OUTPUT_DIR, model_folder_name)
    MODELS_DIR = os.path.join(MODEL_DIR, "models")
    LOGS_DIR = os.path.join(MODEL_DIR, "logs")
    VIDEOS_DIR = os.path.join(MODEL_DIR, "videos")

    # Créer tous les dossiers
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    return MODEL_DIR, MODELS_DIR, LOGS_DIR, VIDEOS_DIR

# Initialisation par défaut (sera remplacée lors de l'entraînement)
# setup_model_directories()  # Commenté pour éviter l'appel automatique

# Points clés
BODYPARTS = ["Hanche", "Genoux", "Cheville"]
NUM_KEYPOINTS = len(BODYPARTS)
KEYPOINT_INDICES = {
    "Hanche": (3, 4),
    "Genoux": (5, 6),
    "Cheville": (7, 8)
}

# Images
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
HEATMAP_SIZE = (56, 56)
HEATMAP_SIGMA = 2.0

# Entraînement
TRAIN_SPLIT = 0.9
BATCH_SIZE = 1 #32
RANDOM_SEED = 42


# Modèle
BACKBONE = "MobileNetV3Large"  # Par défaut: MobileNetV2 (rapide, léger, performant)
PRETRAINED_WEIGHTS = "imagenet"
ALPHA = 1.0

# Export TFLite
TFLITE_QUANTIZATION = True
TFLITE_MODEL_NAME = "pose_model_quantized.tflite"

VERBOSE = 1
