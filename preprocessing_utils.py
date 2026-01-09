import tensorflow as tf
import config
import numpy as np

try:
    from imgaug.augmentables import Keypoint, KeypointsOnImage
    import imgaug.augmenters as iaa
except ModuleNotFoundError:
    pass

import cv2

import pandas as pd
import os
import math



def rotate_image(image, angles):
    # expend first dimension to allow for batch processing
    
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)

    angle_or_angles = tf.convert_to_tensor(
        angles, name="angles", dtype=tf.dtypes.float32
    )
    if len(angle_or_angles.get_shape()) == 0:
        angles = angle_or_angles[None]
    elif len(angle_or_angles.get_shape()) == 1:
        angles = angle_or_angles
    else:
        raise ValueError("angles should have rank 0 or 1.")
    cos_angles = tf.math.cos(angles)
    sin_angles = tf.math.sin(angles)
    x_offset = (
        (w - 1)
        - (cos_angles * (w - 1) - sin_angles * (h - 1))
    ) / 2.0
    y_offset = (
        (h - 1)
        - (sin_angles * (w - 1) + cos_angles * (h - 1))
    ) / 2.0
    num_angles = tf.shape(angles)[0]
    image = tf.expand_dims(image, 0)

    transform =tf.concat(
        values=[
            cos_angles[:, None],
            -sin_angles[:, None],
            x_offset[:, None],
            sin_angles[:, None],
            cos_angles[:, None],
            y_offset[:, None],
            tf.zeros((num_angles, 2), tf.dtypes.float32),
        ],
        axis=1,
    )
    out_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transform,
        output_shape= tf.shape(image)[1:3],
        interpolation="NEAREST",
        fill_mode="CONSTANT",
        fill_value=0.0
    )[0]
    return out_image

def augment(image, landmarks, aug_prob=0.8):
    landmarks = tf.reshape(landmarks, [-1, 2])
    landmarks = tf.cast(landmarks, tf.float32)

    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)

    # gaussian_blur = tf.keras.layers.RandomGaussianBlur(factor=(0.2, 0.8), kernel_size=3, sigma=1)

    def apply_aug(image, landmarks):
        # -------- rotation --------
        def rotate_fn(image, landmarks):
            angle = tf.random.uniform([], -10, 10) * math.pi / 180.0
            image = rotate_image(image, angle)

            center = tf.stack([w / 2.0, h / 2.0])
            lm = landmarks
            lm -= center

            rot = tf.stack([
                [tf.cos(angle), -tf.sin(angle)],
                [tf.sin(angle),  tf.cos(angle)]
            ])
            lm = tf.matmul(lm, rot)
            lm += center
            # lm /= tf.stack([w, h])

            return image, lm

        image, landmarks = tf.cond(
            tf.random.uniform([]) < 0.3,
            lambda: rotate_fn(image, landmarks),
            lambda: (image, landmarks)
        )

        # -------- brightness --------
        image = tf.cond(
            tf.random.uniform([]) < 0.6,
            lambda: tf.image.random_brightness(image, 0.4),
            lambda: image
        )

        # -------- contrast --------
        image = tf.cond(
            tf.random.uniform([]) < 0.6,
            lambda: tf.image.random_contrast(image, 0.1, 0.8),
            lambda: image
        )

        # image = tf.cond(
        #     tf.random.uniform([]) < 0.7,
        #     lambda: gaussian_blur(image),
        #     lambda: image
        # )

        return image, landmarks

    image, landmarks = tf.cond(
        tf.random.uniform([]) < aug_prob,
        lambda: apply_aug(image, landmarks),
        lambda: (image, landmarks)
    )
    return image, tf.reshape(landmarks, [-1])


def preprocess(image, landmarks):
    image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    return image, landmarks


def resize_with_pad_and_landmarks(image, landmarks, target=224, homogeneous_scale=None, heatmap=True):
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    # if homogeneous_scale is not None:
    #     scale = homogeneous_scale
    #     if scale > 1:
    #         image = tf.image.resize(image, (tf.cast(h, tf.float32) * scale, tf.cast(w, tf.float32) * scale))
    #         h, w = tf.shape(image)[0], tf.shape(image)[1]
    #         image = tf.image.crop_to_bounding_box(image, (h - target) // 2, (w - target) // 2, target, target)
    # else:
    scale = tf.minimum(
        target / tf.cast(w, tf.float32),
        target / tf.cast(h, tf.float32)
    )
    image = tf.image.resize_with_pad(image, target, target)

    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    # image = tf.image.resize(image, (new_w, new_h))

    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    landmarks = tf.reshape(landmarks, (-1, 2))

    landmarks = tf.cast(landmarks, tf.float32) * scale
    landmarks = landmarks + [tf.cast(pad_x, tf.float32), tf.cast(pad_y, tf.float32)]
    landmarks = landmarks / target

    heatmaps = generate_heatmaps(landmarks, config.HEATMAP_SIZE, sigma=config.HEATMAP_SIGMA)

    if heatmap:
        return image, heatmaps
    
    return image, tf.reshape(landmarks, (-1,))


def decode_image(image_path, landmarks):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image, landmarks


def preprocess_common(image, landmarks):
    return resize_with_pad_and_landmarks(image, landmarks, target=config.INPUT_SHAPE[0], heatmap=True)


def decode_and_resize(image_path, target):
    image, _ = decode_image(image_path, _)
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    scale = tf.minimum(
        target / tf.cast(w, tf.float32),
        target / tf.cast(h, tf.float32)
    )
    image = tf.image.resize_with_pad(image, target, target)

    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    # image = tf.image.resize(image, (new_w, new_h))
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    image = tf.expand_dims(image, 0)
    return image, [pad_x, pad_y]


def gaussian_heatmap(coords, size, sigma=2):
    """
    center: [x, y] in pixel coordinates
    size: (height, width)
    sigma: standard deviation of Gaussian
    """
    H, W = size
    coords = tf.cast(coords, tf.float32)

    # Convert normalized coords to pixel space
    xs = coords[:, 0] * tf.cast(W, tf.float32)
    ys = coords[:, 1] * tf.cast(H, tf.float32)

    # Create meshgrid
    x_range = tf.range(W, dtype=tf.float32)
    y_range = tf.range(H, dtype=tf.float32)
    yy, xx = tf.meshgrid(y_range, x_range, indexing="ij")  # (H, W)

    # Expand dims for broadcasting
    xx = tf.expand_dims(xx, axis=-1)  # (H, W, 1)
    yy = tf.expand_dims(yy, axis=-1)  # (H, W, 1)

    xs = tf.reshape(xs, (1, 1, -1))   # (1, 1, K)
    ys = tf.reshape(ys, (1, 1, -1))   # (1, 1, K)

    # Gaussian
    heatmaps = tf.exp(
        -((xx - xs) ** 2 + (yy - ys) ** 2) / (2.0 * sigma ** 2)
    )
    return tf.cast(heatmaps, tf.float32)


def generate_heatmaps(coords, size, sigma=2):
    coords = tf.reshape(coords, (-1, 2))
    coords = tf.cast(coords, tf.float32)
    return gaussian_heatmap(coords, size, sigma)


def create_gaussian_heatmap(center_x, center_y, height, width, sigma=2.0):
    """Crée une heatmap gaussienne centrée sur un point"""
    center_x_px = int(center_x * width)
    center_y_px = int(center_y * height)

    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    y = y[:, np.newaxis]

    heatmap = np.exp(-((x - center_x_px) ** 2 + (y - center_y_px) ** 2) / (2 * sigma ** 2))
    return heatmap


def load_image(image_path, target_size):
    """Charge et redimensionne une image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)

    if config.NORMALIZE:
        img = img.astype(np.float32) / 255.0

    return img


def parse_csv_file(csv_path, video_folder):
    """Parse un fichier CSV DeepLabCut et extrait les annotations"""
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    annotations = []

    for idx, row in df.iterrows():
        image_name = row.iloc[2]
        image_path = os.path.join(config.LABELED_DATA_DIR, video_folder, image_name)
        if not os.path.exists(image_path):
            print(f"⚠️  Image non trouvée: {image_path}")
            continue
        
        # Extraire les coordonnées des keypoints
        keypoints = {}
        try:
            for bodypart, (x_idx, y_idx) in config.KEYPOINT_INDICES.items():
                x = float(row.iloc[x_idx])
                y = float(row.iloc[y_idx])
                
                # Vérifier si les coordonnées sont valides (pas NaN)
                if not (np.isnan(x) or np.isnan(y)):
                    keypoints[bodypart] = (x, y)
                else:
                    # Si un point est manquant, on skip cette annotation
                    keypoints = None
                    break
        except (ValueError, IndexError) as e:
            print(f"⚠️  Erreur de parsing pour {image_name}: {e}")
            continue
        
        if keypoints is not None and len(keypoints) == config.NUM_KEYPOINTS:
            annotations.append({
                'image_path': image_path,
                'keypoints': keypoints,
                'video_folder': video_folder
            })
    
    return annotations