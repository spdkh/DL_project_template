# -*- coding: utf-8 -*-
"""
"""

import os
import sys

from utils.autoclip_tf import AutoClipper
from utils.lr_controller import ReduceLROnPlateau

from models.RCAN import RCAN
from models.super_resolution import rcan

import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf
import visualkeras
from tensorflow.keras.layers import Flatten, LeakyReLU, ReLU


class URCAN(RCAN):
    """

    """

    def __init__(self, args, unrolling_its=2):
        RCAN.__init__(self, args)

    def build_model(self):
        sys.setrecursionlimit(10 ** 4)
        x = rcan(
            self.input, scale=2,
            channel=self.args.n_channel,
            n_res_group=self.args.n_ResGroup,
            n_rcab=self.args.n_RCAB)

        initial_x = x

        kernel_T = self.data.psf.transpose(0, 2, 1, 3)
        K_norm = tf.norm(tf.signal.fft3d(self.data.psf))

        gamma = self.args.gamma

        for iteration in range(self.args.unrolling_iter):
            x = rcan(x, scale=1,
                     n_rcab=self.args.n_RCAB,
                     n_res_group=self.args.n_ResGroup,
                     channel=self.args.n_channel)
            # x = x[:, :, :, :, 0]
            x = tf.add(initial_x, x)

            y = self.conv3d(initial_x, kernel_T)

            if gamma >= 0:
                y = tf.multiply(y, gamma)
            x = tf.add(x, y)
            x = tf.signal.fft3d(tf.cast(x,
                                        tf.complex64,
                                        name=None),
                                name=None)
            if gamma >= 0:
                x = tf.multiply(x, 1 / (1 + gamma * K_norm ** 2))

            x = tf.cast(tf.signal.ifft3d(x,
                                         name=None),
                        tf.float32,
                        name=None)
            # x = np.expand_dims(x, axis=-1)
            gamma /= 2

        self.output = x

        self.model = Model(inputs=self.input,
                           outputs=self.output)

        visualkeras.layered_view(self.model,  draw_volume=False,legend=True,  to_file='URCAN.png')  # write to disk

        if self.args.opt == "adam":
            opt = tf.keras.optimizers.Adam(
                self.args.start_lr,
                gradient_transformers=[AutoClipper(20)]
            )
        else:
            opt = self.args.opt

        self.model.compile(loss=self.loss_object,
                           optimizer=opt)

        self.lr_controller = ReduceLROnPlateau(
            model=self.model,
            factor=self.args.lr_decay_factor,
            patience=3,
            mode="min",
            min_delta=1e-2,
            cooldown=0,
            min_lr=self.args.start_lr * 0.001,
            verbose=1,
        )

        if os.path.exists(self.data.save_weights_path + "weights_best.h5"):
            self.model.load_weights(self.data.save_weights_path + "weights_best.h5")
            print(
                "Loading weights successfully: "
                + self.data.save_weights_path + "weights_best.h5"
            )
        elif os.path.exists(self.data.save_weights_path + "weights_latest.h5"):
            self.model.load_weights(self.data.save_weights_path + "weights_latest.h5")
            print(
                "Loading weights successfully: "
                + self.data.save_weights_path
                + "weights_latest.h5"
            )

    def conv3d(self, x, psf):
        psf = np.expand_dims(psf, axis=0)
        if psf.shape[3] > x.shape[3]:
            psf = psf[:, :, :,
                  psf.shape[3] // 2 - (x.shape[3] - 1) // 2:
                  psf.shape[3] // 2 + (x.shape[3] - 1) // 2 + 1,
                  :]

        return tf.nn.conv3d(x,
                            psf,
                            strides=[1] * 5,
                            padding='SAME')
