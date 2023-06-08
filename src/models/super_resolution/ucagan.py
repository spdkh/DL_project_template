# -*- coding: utf-8 -*-
"""
    Unrolled Physics Informed Channel Attention GAN
    author: spdkh
"""

import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf
import visualkeras

from src.models.super_resolution.cagan import CAGAN
from src.utils.architectures.super_resolution import rcan
from src.utils.physics_informed.psf_generator import psf_estimator_3d


class UCAGAN(CAGAN):
    """
        Physics Informed Unrolled Channel Attention GAN class
    """

    def __init__(self, args):
        """
            Init
        """
        CAGAN.__init__(self, args)

    def generator(self, model_input):
        """
            Physics Informed Unrolled Generator design
        """
        sr_out = rcan(model_input,
                      n_rcab=self.args.n_rcab,
                      n_res_group=self.args.n_ResGroup,
                      channel=self.args.n_channel)
        initial_x = sr_out

        kernel_t = self.data.psf.transpose(1, 0, 2, 3)
        k_ft_norm = tf.norm(tf.signal.fft3d(self.data.psf))
        print('K norm:', k_ft_norm)
        # plt.imshow(self.data.psf)
        # plt.colorbar()
        # kernel_t = self.data.psf[:, :, :]
        # kernel_transpose = np.expand_dims(kernel_t, axis=0)
        # kernel_transpose = kernel_t.reshape(1,
        #                                     kernel_t.shape[0],
        #                                     kernel_t.shape[1],
        #                                     kernel_t.shape[2],
        #                                     kernel_t.shape[3])
        gamma = self.args.gamma

        for _ in range(self.args.unrolling_iter):
            sr_out = rcan(sr_out, scale=1,
                          n_rcab=self.args.n_rcab,
                          n_res_group=self.args.n_ResGroup,
                          channel=self.args.n_channel)
            # x = x[:, :, :, :, 0]
            sr_out = tf.add(initial_x, sr_out)

            y_hat = self.conv3d(initial_x, kernel_t)
            y_hat = tf.multiply(y_hat, gamma)
            sr_out = tf.add(sr_out, y_hat)
            sr_ft = tf.signal.fft3d(tf.cast(sr_out,
                                            tf.complex64,
                                            name=None),
                                    name=None)
            sr_out = tf.multiply(sr_ft, 1 / (1 + gamma * k_ft_norm ** 2))
            sr_out = tf.cast(tf.signal.ifft3d(sr_out,
                                              name=None),
                             tf.float32,
                             name=None)
            # x = np.expand_dims(x, axis=-1)
            gamma /= 2

        self.model_output = sr_out

        gen = Model(inputs=self.model_input,
                    outputs=self.model_output)
        tf.keras.utils.plot_model(gen,
                                  to_file='Unrolled_generator.png',
                                  show_shapes=True,
                                  dpi=64,
                                  rankdir='LR')

        visualkeras.layered_view(gen,
                                 draw_volume=False,
                                 legend=True,
                                 to_file='Unrolled_generator2.png')  # write to disk
        return gen

    def conv3d(self, x_in, psf):
        """
            apply mathematical convolution by multiplication in FT domain
        """
        x_in = tf.cast(x_in,
                    tf.complex64,
                    name=None)
        psf = tf.cast(psf,
                      tf.complex64,
                      name=None)
        print(psf.shape, x_in.shape)
        psf = np.expand_dims(psf, axis=0)

        print(psf.shape, x_in.shape)
        if psf.shape[3] > x_in.shape[3]:
            psf = psf[:, :, :,
                  psf.shape[3] // 2 - (x_in.shape[3] - 1) // 2:
                  psf.shape[3] // 2 + (x_in.shape[3] - 1) // 2 + 1,
                  :]
        print(psf.shape, x_in.shape)

        input_fft = tf.signal.fft3d(x_in)
        weights_fft = tf.signal.fft3d(psf)
        conv_fft = tf.multiply(input_fft, weights_fft)
        layer_output = tf.signal.ifft3d(conv_fft)

        layer_output = tf.cast(layer_output,
                               tf.float32,
                               name=None)
        return layer_output
