"""
Prétraitement des données pour l'entraînement
"""
import numpy as np
from pathlib import Path
import tensorflow as tf
import config
from preprocessing_utils import *

def load_all_annotations():
    """
    Charge toutes les annotations de tous les dossiers labeled-data
    
    Returns:
        all_annotations: Liste de toutes les annotations
    """
    all_annotations = []
    
    # Lister tous les sous-dossiers dans labeled-data
    labeled_data_path = Path(config.LABELED_DATA_DIR)
    
    for folder in labeled_data_path.iterdir():
        if folder.is_dir() and not folder.name.endswith('_labeled'):
            # Chercher tous les fichiers CSV commençant par "CollectedData"
            csv_files = list(folder.glob("CollectedData*.csv"))
            
            if csv_files:
                # Prendre le premier fichier trouvé
                csv_file = csv_files[0]
                annotations = parse_csv_file(str(csv_file), folder.name)
                all_annotations.extend(annotations)
            else:
                print(f"Aucun fichier CollectedData*.csv trouvé dans: {folder.name}")
    
    print(f"\nTotal: {len(all_annotations)} images annotées chargées")
    return all_annotations

def save_images(img, heatmaps):
    heatmaps = cv2.resize((heatmaps.numpy() * 255).astype(np.uint8), (int(img.shape[0]), int(img.shape[1])))
    img = cv2.addWeighted(img.numpy().astype(np.uint8), 0.5, heatmaps, 0.5, 0)
    # plt.imshow(img)
    dir = "test_images_aug/"
    list_dir = os.listdir(dir)
    if list_dir != []:
        files = [int(file.split("_")[-1].removesuffix(".png")) for file in list_dir]
        files = np.sort(files)
        idx = int(files[-1]) + 1
    else: 
        idx = 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(idx)
    cv2.imwrite(f"test_images_aug/image_{idx}.png", img)



def create_tf_dataset(annotations):
    """
    Crée les datasets d'images et de heatmaps
    
    Args:
        annotations: Liste des annotations
        image_size: Taille des images (height, width)
    
    Returns:
        images: Array numpy (N, H, W, 3)
        heatmaps: Array numpy (N, H_hm, W_hm, num_keypoints)
    """
    image_paths = np.array([annotation['image_path'] for annotation in annotations])
    landmarks = np.array([np.array(list(annotation['keypoints'].values())).reshape(-1, 6) for annotation in annotations])
    data_set_size = len(image_paths)
    train_size = int(data_set_size * config.TRAIN_SPLIT)
    random_seed = config.RANDOM_SEED
    random_train_idx = np.random.RandomState(random_seed).permutation(data_set_size)[:train_size]
    random_val_idx = np.where(np.logical_not(np.isin(np.arange(data_set_size), random_train_idx)))[0]

    train_img_paths = tf.constant(image_paths[random_train_idx])
    train_landmarks = tf.constant(landmarks[random_train_idx])

    val_img_paths = tf.constant(image_paths[random_val_idx])
    val_landmarks = tf.constant(landmarks[random_val_idx])

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_img_paths, train_landmarks)
    )

    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_img_paths, val_landmarks)
)
    train_dataset = (
        train_ds
        .map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .map(preprocess_common, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        val_ds
        .map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
        .map(preprocess_common, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    # count = 0
    # for img_path, lbl_path in zip(image_paths[random_val_idx], landmarks[random_val_idx]):
    #     img, lbl_path = decode_image(img_path, lbl_path)
    #     img, lbl_path = augment(img, lbl_path)
    #     img, heatmaps = preprocess_common(img, lbl_path)
    #     save_images(img, heatmaps)
    #     count += 1
    #     if count == 3:
    #         break
    return train_dataset, val_dataset


def prepare_data():
    """
    Pipeline complet de préparation des données
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    # 1. Charger toutes les annotations
    annotations = load_all_annotations()
    
    if len(annotations) == 0:
        raise ValueError("Aucune annotation trouvée!")
    
    train_dataset, val_dataset = create_tf_dataset(annotations, config.IMAGE_SIZE)
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test du prétraitement
    train_dataset, val_dataset = prepare_data()