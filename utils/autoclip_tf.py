# MIT License

# Copyright (c) 2020 pseeth

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""# AutoClip: Adaptive Gradient Clipping

This [repository](https://github.com/pseeth/autoclip) accompanies the [paper](https://arxiv.org/abs/2007.14469):

> Prem Seetharaman, Gordon Wichern, Bryan Pardo, Jonathan Le Roux. "AutoClip: Adaptive Gradient Clipping for Source Separation Networks." 2020 IEEE 30th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2020.

## Abstract
> Clipping the gradient is a known approach to improving gradient descent, but requires hand selection of a clipping threshold hyperparameter. We present AutoClip, a simple method for automatically and adaptively choosing a gradient clipping threshold, based on the history of gradient norms observed during training. Experimental results show that applying AutoClip results in improved generalization performance for audio source separation networks. Observation of the training dynamics of a separation network trained with and without AutoClip show that AutoClip guides optimization into smoother parts of the loss landscape. AutoClip is very simple to implement and can be integrated readily into a variety of applications across multiple domains.

## Presentation

This work was presented at MLSP2020 in a special session. If you missed my talk, no worries, there's a pandemic happening so it's recorded! [Here it is](https://share.descript.com/view/18725e02-95fe-4fb0-b32d-26c63617d482).

## Citation
```
@inproceedings{seetharaman2020autoclip,
  title={AutoClip: Adaptive Gradient Clipping for Source Separation Networks},
  author={Seetharaman, Prem, and Wichern, Gordon, and Pardo, Bryan, and Le Roux, Jonathan},
  booktitle={2020 IEEE 30th International Workshop on Machine Learning for Signal Processing (MLSP)},
  year={2020},
  organization={IEEE}
}
```

It needed tensorflow_probability. Install tensorflow-probability 0.15 for Tensorflow 2.7.
https://github.com/tensorflow/probability/releases/tag/v0.15.0

pip install -U tensorflow-probability==0.15
"""

import tensorflow as tf
import tensorflow_probability as tfp


class AutoClipper:
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
        clip_value = tfp.stats.percentile(
            self.grad_history[: self.i], q=self.clip_percentile
        )
        return [(tf.clip_by_norm(g, clip_value), v) for g, v in grads_and_vars]

    def _get_grad_norm(self, t, axes=None, name=None):
        values = tf.convert_to_tensor(
            t.values if isinstance(t, tf.IndexedSlices) else t, name="t"
        )

        # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        l2sum = tf.math.reduce_sum(values * values, axes, keepdims=True)
        pred = l2sum > 0
        # Two-tap tf.where trick to bypass NaN gradients
        l2sum_safe = tf.where(pred, l2sum, tf.ones_like(l2sum))
        return tf.squeeze(tf.where(pred, tf.math.sqrt(l2sum_safe), l2sum))


if __name__ == "__main__":
    # Example usage
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001, gradient_transformers=[AutoClipper(10)]
        ),
        loss="mean_absolute_error",
        metrics=["accuracy"],
    )

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model.fit(x_train, y_train)
