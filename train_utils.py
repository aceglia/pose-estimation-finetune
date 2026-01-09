import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class HeatmapToCoordinates(tf.keras.layers.Layer):
    def __init__(self, heatmap_size, input_size, **kwargs):
        super().__init__(**kwargs)
        self.heatmap_size = heatmap_size
        self.input_size = input_size
        self.scale = input_size // heatmap_size

    def call(self, heatmaps):
        """
        heatmaps: (B, H, W, K)
        returns:  (B, K, 2)
        """
        heatmaps = tf.sigmoid(heatmaps)

        B = tf.shape(heatmaps)[0]
        H = tf.shape(heatmaps)[1]
        W = tf.shape(heatmaps)[2]
        K = tf.shape(heatmaps)[3]

        heatmaps = tf.reshape(heatmaps, [B, H * W, K])

        idx = tf.argmax(heatmaps, axis=1, output_type=tf.int32)  # (B, K)

        y = idx // W
        x = idx % W

        coords = tf.stack([x, y], axis=-1)                        # (B, K, 2)
        coords = tf.cast(coords, tf.float32) * self.scale
        return coords

    def get_config(self):
        config = super().get_config()
        config.update({
            "heatmap_size": self.heatmap_size,
            "input_size": self.input_size,
        })
        return config
    

def soft_argmax_2d(heatmaps):
    """
    heatmaps: (B, H, W, K)
    returns:  (B, K, 2) -> (x, y)
    """
    B = tf.shape(heatmaps)[0]
    H = tf.shape(heatmaps)[1]
    W = tf.shape(heatmaps)[2]
    K = tf.shape(heatmaps)[3]

    heatmaps = tf.reshape(heatmaps, [B, H * W, K])
    heatmaps = tf.nn.softmax(heatmaps, axis=1)

    xs = tf.linspace(0.0, tf.cast(W - 1, tf.float32), W)
    ys = tf.linspace(0.0, tf.cast(H - 1, tf.float32), H)
    xs, ys = tf.meshgrid(xs, ys)
    coords = tf.stack([xs, ys], axis=-1)          # (H, W, 2)
    coords = tf.reshape(coords, [H * W, 2])       # (HW, 2)

    coords = tf.expand_dims(coords, axis=-1)      # (HW, 2, 1)

    exp_coords = tf.reduce_sum(
        heatmaps[:, :, tf.newaxis, :] * coords, axis=1
    )                                             # (B, 2, K)

    return tf.transpose(exp_coords, [0, 2, 1])

@tf.keras.utils.register_keras_serializable()
class LandmarkHuberLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        heatmap_size,
        input_size,
        delta=5.0,
        name="landmark_huber_loss",
    ):
        super().__init__(name=name)
        self.heatmap_size = heatmap_size
        self.input_size = input_size
        self.scale = input_size // heatmap_size
        self.delta = delta

    def call(self, y_true, y_pred):
        # Convert logits → probabilities
        y_pred = tf.sigmoid(y_pred)

        # Soft-argmax → coordinates
        pred_coords = soft_argmax_2d(y_pred)
        true_coords = soft_argmax_2d(y_true)

        # Rescale to input resolution
        pred_coords *= tf.cast(self.scale, tf.float32)
        true_coords *= tf.cast(self.scale, tf.float32)

        error = pred_coords - true_coords
        abs_error = tf.abs(error)

        # Huber loss
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * tf.square(quadratic) + self.delta * linear

        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "heatmap_size": self.heatmap_size,
            "input_size": self.input_size,
            "delta": self.delta,
        })
        return config