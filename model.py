"""
Construction du modèle de pose estimation avec support multi-backbones
Supporte: MobileNetV2/V3, EfficientNetLite, EfficientNetB, EfficientNetV2
"""
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3
)
import config

def get_backbone(backbone_name="MobileNetV2", input_shape=(192, 192, 3), alpha=1.0):
    """
    Charge le backbone pré-entraîné
    
    Args:
        backbone_name: Nom du backbone (MobileNetV2, MobileNetV3Small/Large, 
                       EfficientNetLite0-4, EfficientNetB0-3, EfficientNetV2B0-3)
        input_shape: Forme de l'entrée (H, W, C)
        alpha: Width multiplier (seulement pour MobileNet)
    
    Returns:
        backbone: Modèle Keras du backbone
    """
    # MobileNet backbones
    if backbone_name == "MobileNetV2":
        backbone = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS,
            alpha=alpha
        )
    elif backbone_name == "MobileNetV3Small":
        backbone = MobileNetV3Small(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS,
            alpha=alpha,
            include_preprocessing=True,
            minimalistic=False
        )
    elif backbone_name == "MobileNetV3Large":
        backbone = MobileNetV3Large(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS,
            include_preprocessing=True,
            alpha=alpha,
            minimalistic=False
        )
    
    # EfficientNetLite backbones (légers, optimisés edge/mobile)
    elif backbone_name == "EfficientNetLite0":
        backbone = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetLite1":
        backbone = EfficientNetB1(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetLite2":
        backbone = EfficientNetB2(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetLite3":
        backbone = EfficientNetB3(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetLite4":
        # Lite4 utilise B3 comme base avec optimisations
        backbone = EfficientNetB3(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    
    # EfficientNetB backbones (haute précision)
    elif backbone_name == "EfficientNetB0":
        backbone = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetB1":
        backbone = EfficientNetB1(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetB2":
        backbone = EfficientNetB2(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetB3":
        backbone = EfficientNetB3(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    
    # EfficientNetV2 backbones (plus rapides, meilleure précision)
    elif backbone_name == "EfficientNetV2B0":
        backbone = EfficientNetV2B0(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetV2B1":
        backbone = EfficientNetV2B1(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetV2B2":
        backbone = EfficientNetV2B2(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetV2B3":
        backbone = EfficientNetV2B3(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    
    else:
        raise ValueError(f"Backbone non supporté: {backbone_name}. "
                        f"Backbones disponibles: MobileNetV2, MobileNetV3Small/Large, "
                        f"EfficientNetLite0-4, EfficientNetB0-3, EfficientNetV2B0-3")
    
    return backbone

def get_head(num_keypoints=3, backbone_output=None, heatmaps=True, heatmap_size=(56, 56)):
    """
    Pose estimation head similar to DeepLabCut.
    Args:
        backbone_output: tf.Tensor, output from backbone feature extractor
        n_landmarks: int, number of keypoints
        heatmap_size: tuple, desired output heatmap size (H, W)
    Returns:
        heatmaps: tf.Tensor, shape [B, H, W, n_landmarks]
    """

    x = backbone_output

    if not heatmaps:
        # Head
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        # 3 landmarks × (x, y)
        outputs = tf.keras.layers.Dense(
            config.NUM_KEYPOINTS * 2,
            activation="sigmoid"  # normalized [0, 1]
        )(x)
        return outputs

    # for f in [256, 128, 64]:
    for f in [256]:
        x = layers.Conv2DTranspose(f, 3, strides=2, padding="same")(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)

    heatmaps = layers.Conv2D(
        num_keypoints,
        1,
        padding="same",
        name="heatmap_output"
    )(x)
    return heatmaps


def heatmaps_to_coords(heatmaps):
    # heatmaps: [B, H, W, n_landmarks]
    coords = []
    probs = []
    for i in range(heatmaps.shape[-1]):
        h = heatmaps[..., i]
        max_val = tf.reduce_max(h, axis=[1, 2])
        idx = tf.argmax(tf.reshape(h, [h.shape[0], -1]), axis=1)
        y = idx // h.shape[2]
        x = idx % h.shape[2]
        coords.append(tf.stack([x, y], axis=-1))
        probs.append(max_val)
    coords = tf.stack(coords, axis=1)
    probs = tf.stack(probs, axis=1)
    return coords, probs


def build_pose_model(num_keypoints=3, backbone_name="MobileNetV2", input_shape=(192, 192, 3)):
    """
    Construit le modèle complet de pose estimation
    
    Architecture:
        - Backbone (MobileNet/EfficientNet pré-entraîné sur ImageNet)
        - Upsampling progressif adaptatif
        - Tête convolutionnelle pour prédire les heatmaps
    
    Args:
        num_keypoints: Nombre de points clés à prédire
        backbone_name: Nom du backbone
        input_shape: Forme de l'entrée (H, W, C)
    
    Returns:
        model: Modèle Keras compilé
    """
    inputs = keras.Input(shape=input_shape, name="image_input")
    
    backbone = get_backbone(backbone_name, input_shape, config.ALPHA)
    
    backbone.trainable = False

    x = backbone.get_layer("activation_11").output
    x = layers.Conv2DTranspose(256, 4, strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(256, 4, strides=2, padding="same")(x)
    heatmaps = layers.Conv2D(
        num_keypoints,
        1,
        padding="same",
        name="heatmap_output"
    )(x)

    # x = layers.Conv2D(576, kernel_size=3, padding='same', dilation_rate=2, activation='relu')(x)

    # outputs = get_head(num_keypoints, x, heatmaps=True, heatmap_size=config.HEATMAP_SIZE)

    model = Model(inputs=backbone.inputs, outputs=heatmaps, name=f'pose_estimation_{backbone_name}')
    
    return model


def compile_model(model, learning_rate=1e-4, optimizer_name='adam', loss="mse"):
    """
    Compile le modèle avec la loss et l'optimiseur
    
    Args:
        model: Modèle Keras
        learning_rate: Taux d'apprentissage
        optimizer_name: Nom de l'optimiseur
    
    Returns:
        model: Modèle compilé
    """
    # Choisir l'optimiseur
    if optimizer_name.lower() == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Optimiseur non supporté: {optimizer_name}")
    
    # Compiler avec MSE loss
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean Squared Error entre heatmaps prédites et vraies
        metrics=['mae']  # Mean Absolute Error comme métrique additionnelle
    )
    
    return model


def create_model():
    """
    Pipeline complet de création et compilation du modèle
    
    Returns:
        model: Modèle Keras compilé et prêt à l'entraînement
    # """
    model = build_pose_model(
        num_keypoints=config.NUM_KEYPOINTS,
        backbone_name=config.BACKBONE,
        input_shape=config.INPUT_SHAPE
    )
    return model

if __name__ == "__main__":
    # Test de la construction du modèle
    model = create_model()
    
    print("\n📊 Informations du modèle:")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output shape: {model.output_shape}")
    print(f"   - Nombre de paramètres: {model.count_params():,}")
