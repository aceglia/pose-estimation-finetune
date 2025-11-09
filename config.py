"""
Configuration pour le fine-tuning de pose estimation
"""
import os
from datetime import datetime

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELED_DATA_DIR = os.path.join(BASE_DIR, "labeled-data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Fonction pour créer le nom du dossier modèle
def get_model_folder_name(backbone=None, timestamp=None):
    """Génère le nom du dossier pour un modèle spécifique"""
    if backbone is None:
        backbone = BACKBONE
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Nettoyer le nom du backbone (enlever caractères spéciaux)
    clean_backbone = backbone.replace("MobileNetV3", "MNv3").replace("MobileNetV2", "MNv2")

    return f"{clean_backbone}_{timestamp}"

# Dossier modèle actuel (sera défini dynamiquement)
MODEL_DIR = None
MODELS_DIR = None
LOGS_DIR = None
VIDEOS_DIR = None

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
IMAGE_SIZE = (192, 192)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
HEATMAP_SIZE = (48, 48)
HEATMAP_SIGMA = 2.0
NORMALIZE = True

# Entraînement
TRAIN_SPLIT = 0.8
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
OPTIMIZER = "adam"
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
RANDOM_SEED = 42

# Modèle
BACKBONE = "MobileNetV2"
PRETRAINED_WEIGHTS = "imagenet"
ALPHA = 1.0

# Export TFLite
TFLITE_QUANTIZATION = True
TFLITE_MODEL_NAME = "pose_model_quantized.tflite"

# Augmentation
USE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "horizontal_flip": True,
    "fill_mode": "nearest"
}

VERBOSE = 1
