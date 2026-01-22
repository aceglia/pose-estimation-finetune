import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class HeatmapToCoordinates(tf.keras.layers.Layer):
    def __init__(self, heatmap_size, input_size, from_pred=False, **kwargs):
        super().__init__(**kwargs)
        self.heatmap_size = heatmap_size
        self.input_size = input_size
        self.scale = input_size // heatmap_size
        self.from_pred=from_pred

    def call(self, heatmaps):
        """
        heatmaps: (B, H, W, K)
        returns:  (B, K, 2)
        """
        # heatmaps = tf.sigmoid(heatmaps)

        # B = tf.shape(heatmaps)[0]
        # H = tf.shape(heatmaps)[1]
        # W = tf.shape(heatmaps)[2]
        # K = tf.shape(heatmaps)[3]

        # heatmaps = tf.reshape(heatmaps, [B, H * W, K])

        # idx = tf.argmax(heatmaps, axis=1, output_type=tf.int32)  # (B, K)

        # y = idx // W
        # x = idx % W

        # coords = tf.stack([x, y], axis=-1)                        # (B, K, 2)
        coords = soft_argmax_2d(heatmaps, from_pred=self.from_pred)
        coords = tf.cast(coords, tf.float32) * self.scale
        return coords

    def get_config(self):
        config = super().get_config()
        config.update({
            "heatmap_size": self.heatmap_size,
            "input_size": self.input_size,
        })
        return config
    
# @tf.keras.utils.register_keras_serializable()
# def soft_argmax_2d(heatmaps):
#     """
#     heatmaps: (B, H, W, K)
#     returns:  (B, K, 2) -> (x, y)
#     """
#     B = tf.shape(heatmaps)[0]
#     H = tf.shape(heatmaps)[1]
#     W = tf.shape(heatmaps)[2]
#     K = tf.shape(heatmaps)[3]

#     heatmaps = tf.reshape(heatmaps, [B, H * W, K])
#     # heatmaps = tf.nn.softmax(heatmaps, axis=1)
#     heatmaps = tf.exp(heatmaps) / tf.reduce_sum(tf.exp(heatmaps), axis=1, keepdims=True)
#     idx = tf.argmax(heatmaps, axis=1, output_type=tf.int32)   
#     y = idx // W
#     x = idx % W

#     coords = tf.stack([x, y], axis=-1)            
#     return coords


def softargmax(x, beta=1e10):
  x = tf.convert_to_tensor(x)
  x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
  return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

@tf.keras.utils.register_keras_serializable()
def soft_argmax_2d(heatmaps, from_pred=True):
    """
    Differentiable 2D Soft-Argmax.
    heatmaps: (B, H, W, K)
    returns: (B, K, 2) -> (x, y) coordinates
    """
    B, H, W, K = tf.shape(heatmaps)[0], tf.shape(heatmaps)[1], tf.shape(heatmaps)[2], tf.shape(heatmaps)[3]

    flat_heatmaps = tf.reshape(heatmaps, [B, H * W, K]) # (B, H*W, K)
    if from_pred:
        # T = 0.1 # Lower = sparser/sharper
        # # flat_heatmaps *= 100
        # e_x = np.exp(((flat_heatmaps - np.max(flat_heatmaps)) / T) - 3)
        # sharpen_heatmap = e_x / e_x.sum()
        probs = tf.nn.softmax(flat_heatmaps * 100, axis=1) 
        # import matplotlib.pyplot as plt
        # import numpy as np
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # x = np.arange(0, 56)
        # y = np.arange(0, 56)
        # X, Y = np.meshgrid(x, y)
        # probs = probs.numpy()[0].reshape(56, 56, -1)
        # Z = probs[..., 1] + probs[..., 0] + probs[..., 2]
        # ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='green')
        # # ax.scatter(keypoints[:, 0] / 4, keypoints[:, 1] / 4, confidence)
        # plt.show()
    else:
        flat_heatmaps /= tf.reduce_sum(flat_heatmaps, axis=1, keepdims=True) + 1e-6
        probs = flat_heatmaps        # (B, H*W, K)

    # 2. Create meshgrid of coordinates (normalized or raw)
    pos_y, pos_x = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
    pos_x = tf.cast(pos_x, tf.float32)
    pos_y = tf.cast(pos_y, tf.float32)

    # 3. Flatten coordinates to match flat_heatmaps shape
    flat_x = tf.reshape(pos_x, [H * W, 1]) # (H*W, 1)
    flat_y = tf.reshape(pos_y, [H * W, 1]) # (H*W, 1)

    # 4. Compute expected value (weighted average) of coordinates
    # probs shape: (B, H*W, K)
    # result shape: (B, K)


    expected_x = tf.reduce_sum(probs * flat_x, axis=1) 
    expected_y = tf.reduce_sum(probs * flat_y, axis=1)
    return tf.stack([expected_x, expected_y], axis=-1)


@tf.keras.utils.register_keras_serializable()
class LandmarkHuberLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        heatmap_size,
        input_size,
        delta=1.0,
        name="landmark_huber_loss",
        with_coords = True,
        huber=True,
        **kwargs
    ):
        super().__init__(name=name)
        self.heatmap_size = heatmap_size
        self.input_size = input_size
        self.scale = input_size // heatmap_size
        self.delta = delta
        self.with_coords = with_coords
        self.huber = huber

    def call(self, y_true, y_pred):
        # y_pred = tf.sigmoid(y_pred)
        # crossentrloss =  tf.keras.losses.BinaryFocalCrossentropy(
        #     from_logits=True, 
        #     gamma=1.0,
        #     # alpha=0.6,
        #     apply_class_balancing=False
        # )(y_true, y_pred)
        mseloss =  tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        if not self.with_coords:
            return mseloss

        # Soft-argmax → coordinates
        pred_coords = soft_argmax_2d(tf.stop_gradient(y_pred))
        true_coords = soft_argmax_2d(tf.stop_gradient(y_true), from_pred=False)
        pred_coords *= tf.cast(self.scale, tf.float32)
        true_coords *= tf.cast(self.scale, tf.float32)

        if self.huber:
            coords_loss = tf.keras.losses.Huber(delta=self.delta, reduction="sum_over_batch_size", name="huber_loss")(true_coords, pred_coords)
        else:
            coords_loss = tf.keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mse_loss")(true_coords, pred_coords)
            
        return mseloss + (0.03 * coords_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "heatmap_size": self.heatmap_size,
            "input_size": self.input_size,
            "delta": self.delta,
        })
        return config
    
@tf.keras.utils.register_keras_serializable()
class LandmarkMetrics(tf.keras.metrics.Metric):
    def __init__(self, name="root_mean_squared_error", heatmap_size=56, input_size=224, **kwargs):
        super().__init__(name=name, **kwargs)
        self.heatmap_size = heatmap_size
        self.input_size = input_size
        self.scale = input_size // heatmap_size
        # Use add_weight for stateful metrics in Keras 3
        self.error_sum = self.add_weight(name='error_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to tensors using backend-agnostic ops
        pred_coords = soft_argmax_2d(tf.stop_gradient(y_pred))
        true_coords = soft_argmax_2d(tf.stop_gradient(y_true), from_pred=False)
        pred_coords *= tf.cast(self.scale, tf.float32)
        true_coords *= tf.cast(self.scale, tf.float32)
        y_true = tf.keras.ops.cast(true_coords, self.dtype)
        y_pred = tf.keras.ops.cast(pred_coords, self.dtype)
        
        # Perform calculations using ops (no .numpy() needed)
        # Example logic: MSE for this batch
        diff = tf.keras.ops.square(y_pred - y_true)
        batch_error = tf.keras.ops.mean(diff)
        
        # Update weights using assign_add
        self.error_sum.assign_add(batch_error)
        self.total_samples.assign_add(1.0)

    def result(self):
        # Compute final metric from accumulated state
        return tf.keras.ops.sqrt(self.error_sum / self.total_samples)

    def reset_state(self):
        # Essential to reset state between epochs
        self.error_sum.assign(0.0)
        self.total_samples.assign(0.0)