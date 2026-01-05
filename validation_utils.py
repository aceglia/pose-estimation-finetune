import numpy as np
import tensorflow as tf
import config
import matplotlib.pyplot as plt
import os


def from_heatmaps_to_coords(heatmap, from_prediction=True):
    if not isinstance(heatmap, np.ndarray):
        heatmap = heatmap.numpy()
    if len(heatmap.shape) != 4:
        heatmap = heatmap[None, :, :, :]
    coords_array = np.zeros((heatmap.shape[0], 2, heatmap.shape[-1]))
    for h, heat in enumerate(heatmap):
        if from_prediction:
            heat = tf.sigmoid(heat).numpy()
        scale = config.INPUT_SHAPE[0] // config.HEATMAP_SIZE[0]
        confidence = [np.max(heat[:, :, i]).tolist() for i in range(heat.shape[-1])]
        coords = [np.array(np.where(heat[:, :, i] == confidence[i])).flatten() * scale for i in range(heat.shape[-1])]
        for a, ar in enumerate(coords):
            if len(ar) != 2:
                coords[a] = np.mean(ar.reshape(2, -1), axis=0)

        coords_array[h, ...] = np.array(coords).T
    return coords_array, confidence

def plot_validation_images(model, val_ds, model_dir=""):
    img, gt_heatmaps = next(iter(val_ds.take(1)))
    y_pred = model.predict(img)
    gt_coords, _ = from_heatmaps_to_coords(gt_heatmaps, from_prediction=False)
    pr_coords, _ = from_heatmaps_to_coords(y_pred, from_prediction=True)
    images_paths = os.path.join(model_dir, "images")
    os.makedirs(os.path.join(model_dir, "images"), exist_ok=True)
    for i in range(len(gt_coords)):
        gt_coords_tmp = gt_coords[i]
        pr_coords_tmp = pr_coords[i]
        img_tmp = img[i]
        img_tmp = img_tmp.numpy().astype(np.uint8)
        [plt.scatter(gt_coords_tmp[1, i], gt_coords_tmp[0, i], color="r") for i in range(gt_coords_tmp.shape[-1])]
        [plt.scatter(pr_coords_tmp[1, i], pr_coords_tmp[0, i], color="b", s=10) for i in range(pr_coords_tmp.shape[-1])]
        plt.imshow(img_tmp)
        plt.savefig(os.path.join(images_paths, f"image_{i}.png"))
        plt.close()