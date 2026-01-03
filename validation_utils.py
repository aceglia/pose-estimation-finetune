import numpy as np
import tensorflow as tf
import config


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
        coords = [np.array(np.where(heat[:, :, i] == np.max(heat[:, :, i]))).flatten() * scale for i in range(heat.shape[-1])]
        coords_array[h, ...] = np.array(coords).T
    return coords_array

