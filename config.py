"""
Configuration pour le fine-tuning de pose estimation
"""
import os
from datetime import datetime

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prefix = "/mnt/c" if os.name == "posix" else "C:\\"
LABELED_DATA_DIR = os.path.join(prefix, "Users", "neuromolity-lab", "Downloads", "PFE", "PFE", "JambeGaucheLabeled", "1-TouteLabeledGauche")
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

# Dossier modèle actuel (sera défini dynamiquement)
MODEL_DIR = None
MODELS_DIR = None
LOGS_DIR = None
VIDEOS_DIR = LABELED_DATA_DIR = os.path.join(prefix, "Users", "neuromolity-lab", "Downloads", "PFE", "PFE", "JambeGaucheLabeled", "1-TouteLabeledGauche")

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
NORMALIZE = True

# Entraînement
TRAIN_SPLIT = 0.8
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
BACKBONE_LEARNING_RATE = 1e-4
BACKBONE_TRAINABLE_LAYERS = 20
HEAD_LEARNING_RATE = [0.005, 0.02, 
0.002,
0.001,
]
OPTIMIZER = "sgd"
EARLY_STOPPING_PATIENCE = 50
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
RANDOM_SEED = 42
MOMENTUM = 0.9

# Modèle
BACKBONE = "MobileNetV2"  # Par défaut: MobileNetV2 (rapide, léger, performant)
PRETRAINED_WEIGHTS = "imagenet"
ALPHA = 1.0

# Tailles d'images recommandées par backbone (pour performances optimales)
BACKBONE_INPUT_SIZES = {
    "MobileNetV2": (192, 192),
    "MobileNetV3Small": (224, 224),
    "MobileNetV3Large": (224, 224),
    "EfficientNetLite0": (224, 224),
    "EfficientNetLite1": (240, 240),
    "EfficientNetLite2": (260, 260),
    "EfficientNetLite3": (280, 280),
    "EfficientNetLite4": (300, 300),
    "EfficientNetB0": (224, 224),
    "EfficientNetB1": (240, 240),
    "EfficientNetB2": (260, 260),
    "EfficientNetB3": (300, 300),
    "EfficientNetV2B0": (224, 224),
    "EfficientNetV2B1": (240, 240),
    "EfficientNetV2B2": (260, 260),
    "EfficientNetV2B3": (300, 300),
}

# Ratios de réduction du backbone (pour adapter la tête de déconvolution)
BACKBONE_REDUCTION_RATIOS = {
    "MobileNetV2": 32,          # 192/32 = 6x6
    "MobileNetV3Small": 32,
    "MobileNetV3Large": 32,
    "EfficientNetLite0": 32,    # 224/32 = 7x7
    "EfficientNetLite1": 32,
    "EfficientNetLite2": 32,
    "EfficientNetLite3": 32,
    "EfficientNetLite4": 32,
    "EfficientNetB0": 32,
    "EfficientNetB1": 32,
    "EfficientNetB2": 32,
    "EfficientNetB3": 32,
    "EfficientNetV2B0": 32,
    "EfficientNetV2B1": 32,
    "EfficientNetV2B2": 32,
    "EfficientNetV2B3": 32,
}

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
