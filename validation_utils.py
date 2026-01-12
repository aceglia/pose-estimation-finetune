import numpy as np
import tensorflow as tf
import config
import matplotlib.pyplot as plt
from train_utils import HeatmapToCoordinates
import os


# def soft_argmax_2d(heatmaps):
#     """
#     Differentiable 2D Soft-Argmax.
#     heatmaps: (B, H, W, K)
#     returns: (B, K, 2) -> (x, y) coordinates
#     """
#     B, H, W, K = tf.shape(heatmaps)[0], tf.shape(heatmaps)[1], tf.shape(heatmaps)[2], tf.shape(heatmaps)[3]

#     flat_heatmaps = tf.reshape(heatmaps, [B, H * W, K]) # (B, H*W, K)
#     probs = tf.nn.softmax(flat_heatmaps, axis=1)        # (B, H*W, K)

#     # 2. Create meshgrid of coordinates (normalized or raw)
#     pos_y, pos_x = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
#     pos_x = tf.cast(pos_x, tf.float32)
#     pos_y = tf.cast(pos_y, tf.float32)

#     # 3. Flatten coordinates to match flat_heatmaps shape
#     flat_x = tf.reshape(pos_x, [H * W, 1]) # (H*W, 1)
#     flat_y = tf.reshape(pos_y, [H * W, 1]) # (H*W, 1)

#     # 4. Compute expected value (weighted average) of coordinates
#     # probs shape: (B, H*W, K)
#     # result shape: (B, K)
#     expected_x = tf.reduce_sum(probs * flat_x, axis=1) 
#     expected_y = tf.reduce_sum(probs * flat_y, axis=1)

#     return tf.stack([expected_x, expected_y], axis=-1)


def from_heatmaps_to_coords(heatmap, from_prediction=True, input_scale=config.INPUT_SHAPE):
    # if not isinstance(heatmap, np.ndarray):
    #     heatmap = heatmap.numpy()
    if len(heatmap.shape) != 4:
        heatmap = heatmap[None, :, :, :]
    # coords_array = np.zeros((heatmap.shape[0], 2, heatmap.shape[-1]))
    # for h, heat in enumerate(heatmap):
    #     if from_prediction:
    #         heat = tf.sigmoid(heat).numpy()
    #     scale = input_scale[0] // config.HEATMAP_SIZE[0]
    #     confidence = [np.max(heat[:, :, i]).tolist() for i in range(heat.shape[-1])]
    #     coords = [np.array(np.where(heat[:, :, i] == confidence[i])).flatten() * scale for i in range(heat.shape[-1])]
    #     for a, ar in enumerate(coords):
    #         if len(ar) != 2:
    #             coords[a] = np.mean(ar.reshape(2, -1), axis=1).astype(np.uint8)

    #     coords_array[h, ...] = np.array(coords).T
    # coords_array.swapaxes(1, 2)
    heatmaps = tf.cast(heatmap, tf.float32)
    B = tf.shape(heatmaps)[0]
    H = tf.shape(heatmaps)[1]
    W = tf.shape(heatmaps)[2]
    K = tf.shape(heatmaps)[3]

    heatmaps = tf.reshape(heatmaps, [B, H * W, K])

    idx = tf.argmax(heatmaps, axis=1, output_type=tf.int32)
    # if from_prediction:  # (B, K)
    #     softmax = tf.nn.softmax(heatmaps, axis=1)
    #     idx = tf.argmax(softmax, axis=1, output_type=tf.int32)

    y = idx // W
    x = idx % W

    coords = tf.stack([x, y], axis=-1)                        # (B, K, 2)
    coords = tf.cast(coords, tf.float32) * (input_scale[0] // config.HEATMAP_SIZE[0])

    return coords


def plot_validation_images(model, val_ds, model_dir=""):
    img, gt_heatmaps = next(iter(val_ds.take(1)))
    y_pred = model.predict(img)
    pr_coords = y_pred
    if model.output.shape[1:] != (3, 2):
        pr_coords = HeatmapToCoordinates(config.HEATMAP_SIZE[0], config.INPUT_SHAPE[0], from_pred=True)(y_pred).numpy()
    # gt_coords = from_heatmaps_to_coords(gt_heatmaps, from_prediction=False).numpy()
    gt_coords = HeatmapToCoordinates(config.HEATMAP_SIZE[0], config.INPUT_SHAPE[0], from_pred=False)(gt_heatmaps).numpy()
    images_paths = os.path.join(model_dir, "images")
    os.makedirs(os.path.join(model_dir, "images"), exist_ok=True)
    for i in range(len(gt_coords)):
        gt_coords_tmp = gt_coords[i]
        pr_coords_tmp = pr_coords[i]
        img_tmp = img[i]
        img_tmp = img_tmp.numpy().astype(np.uint8)
        [plt.scatter(gt_coords_tmp[i, 0], gt_coords_tmp[i, 1], color="r") for i in range(gt_coords_tmp.shape[0])]
        [plt.scatter(pr_coords_tmp[i, 0], pr_coords_tmp[i, 1], color="b", s=10) for i in range(pr_coords_tmp.shape[0])]
        plt.imshow(img_tmp)
        # plt.show(block=True)
        plt.savefig(os.path.join(images_paths, f"image_{i}.png"))
        plt.close()

def project_keypoints(keypoints, image_size, paddings, input_size = config.INPUT_SHAPE):
    """
    Keypoints ar already projected to the input size, we need to project them on the original image with according padding.
    """
    image_size_square = max(image_size)
    input_size = input_size[0]
    keypoints *= image_size_square // input_size 
    keypoints[0, 1, :] -= np.array((paddings[0]) * (image_size_square // input_size))
    keypoints[0, 0, :] -= np.array((paddings[1]) * (image_size_square // input_size))
    return keypoints


def evaluate_model(model, val_ds, model_dir):
    plot_validation_images(model, val_ds, model_dir)
