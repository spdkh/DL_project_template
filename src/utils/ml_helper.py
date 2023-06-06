"""
    caGAN project ML helper functions

Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv3D, LeakyReLU


def conv_block_3d(x_in, filters, kernel):
    """
        params:
        x_in: tf tensor
            input of NN
        filters: int
            number of filters
        kernel: int
            kernel size

        returns: tf tensor
            output of NN
    """
    x_out = Conv3D(filters, kernel_size=kernel, padding='same')(x_in)
    x_out = LeakyReLU(alpha=0.1)(x_out)
    return x_out


class AutoClipper:
    """
        source: https://github.com/pseeth/autoclip/tree/master
    """
    def __init__(self, clip_percentile, history_size=10000):

        self.clip_percentile = clip_percentile
        self.grad_history = tf.Variable(tf.zeros(history_size), trainable=False)
        self.i = tf.Variable(0, trainable=False)
        self.history_size = history_size

    def __call__(self, grads_and_vars):
        grad_norms = [self._get_grad_norm(g) for g, _ in grads_and_vars]
        total_norm = tf.norm(grad_norms)
        assign_idx = tf.math.mod(self.i, self.history_size)
        self.grad_history = self.grad_history[assign_idx].assign(total_norm)
        self.i = self.i.assign_add(1)
        clip_value = tfp.stats.percentile(self.grad_history[: self.i], q=self.clip_percentile)
        return [(tf.clip_by_norm(g, clip_value), v) for g, v in grads_and_vars]

    def _get_grad_norm(self, t, axes=None):
        values = tf.convert_to_tensor(t.values if isinstance(t, tf.IndexedSlices) else t, name="t")

        # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        l2sum = tf.math.reduce_sum(values * values, axes, keepdims=True)
        pred = l2sum > 0
        # Two-tap tf.where trick to bypass NaN gradients
        l2sum_safe = tf.where(pred, l2sum, tf.ones_like(l2sum))
        return tf.squeeze(tf.where(pred, tf.math.sqrt(l2sum_safe), l2sum))
