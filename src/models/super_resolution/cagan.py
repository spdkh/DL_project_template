# -*- coding: utf-8 -*-
"""
    channel attention GAN
    author: spdkh
    date: 2023
"""

import os

import tensorflow as tf
import visualkeras
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from src.models.gan import GAN
from src.utils.architectures.binary_classification import discriminator
from src.utils.architectures.super_resolution import rcan
from src.utils.ml_helper import AutoClipper
from src.utils import const


class CAGAN(GAN):
    """
        class to implement channel attention gan arch
    """
    def __init__(self, args):
        """
            params:
                args: parsearg object
        """
        GAN.__init__(self, args)

        self.disc_opt = tf.keras.optimizers.Adam(args.d_start_lr,
                                                 beta_1=args.d_lr_decay_factor)

    def build_model(self):
        """
            function to define the model,
            loss functions,
            optimizers,
            learning rate controller,
            compile model,
            load initial weights if available
            GAN model includes discriminator and generator separately
        """
        self.d_model, _ = self.discriminator()
        self.model = self.generator(self.model_input)

        fake_hp = self.model(inputs=self.model_input)
        judge = self.d_model(fake_hp)

        # last fake hp
        gen_loss = self.generator_loss(judge)

        # loss_wf = create_psf_loss(self.data.psf)

        if self.args.opt == "adam":
            opt = tf.keras.optimizers.Adam(
                self.args.start_lr,
                gradient_transformers=[AutoClipper(20)]
            )
        else:
            opt = self.args.opt

        self.lr_controller = ReduceLROnPlateau(
            model=self.model,
            factor=self.args.lr_decay_factor,
            patience=self.args.iteration * 1e-2,
            mode="min",
            min_delta=1e-2,
            cooldown=0,
            min_lr=self.args.start_lr * 1e-3,
            verbose=1,
        )

        self.model.compile(loss=[self.loss_mse_ssim_3d, gen_loss],
                           optimizer=opt,
                           loss_weights=[1,
                                         self.args.alpha])

        self.d_lr_controller = ReduceLROnPlateau(
            model=self.d_model,
            factor=self.args.d_lr_decay_factor,
            patience=3,
            mode="min",
            min_delta=1e-2,
            cooldown=0,
            min_learning_rate=self.args.d_start_lr * 0.001,
            verbose=1,
        )

        self.d_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        if self.args.load_weights:
            if os.path.exists(const.WEIGHTS_DIR
                              + 'weights_best.h5'):
                self.model.load_weights(const.WEIGHTS_DIR
                                        + 'weights_best.h5')
                self.d_model.load_weights(const.WEIGHTS_DIR
                                          + 'weights_disc_best.h5')
                print('Loading weights successfully: '
                      + const.WEIGHTS_DIR
                      + 'weights_best.h5')
            elif os.path.exists(const.WEIGHTS_DIR + 'weights_latest.h5'):
                self.model.load_weights(const.WEIGHTS_DIR
                                        + 'weights_latest.h5')
                self.d_model.load_weights(const.WEIGHTS_DIR
                                          + 'weights_disc_latest.h5')
                print('Loading weights successfully: '
                      + const.WEIGHTS_DIR
                      + 'weights_latest.h5')

    def train_gan(self):
        """
            one full iteration of generator + discriminator training
        """
        loss = {'disc': 0, 'gen': 0}
        batch_size_d = self.args.batch_size

        #         train discriminator
        for _ in range(self.args.train_discriminator_times):
            # todo:  Question: is this necessary? (reloading the data for disc) :
            #       I think yes: update: I dont think so
            # todo: Question: should they be the same samples? absolutely yes(They already are):
            #       I think they should not : update: this is wrong: they should
            input_d, gt_d = \
                self.data.image2image_batch_load(batch_size_d,
                                                 self.batch_iterator(),
                                                 self.scale_factor)
            # wf_d = self.data.calc_wf(input_d)
            fake_input_d = self.model.predict(input_d)

            # discriminator loss separate for real/fake:
            # https://stackoverflow.com/questions/49988496/loss-functions-in-gans
            with tf.GradientTape() as disc_tape:
                disc_real_output = self.d_model(gt_d)
                disc_fake_output = self.d_model(fake_input_d)
                d_loss = self.discriminator_loss(disc_real_output,
                                                       disc_fake_output)

            disc_gradients = disc_tape.gradient(d_loss,
                                                self.d_model.trainable_variables)

            self.disc_opt.apply_gradients(zip(disc_gradients,
                                              self.d_model.trainable_variables))

            self.d_loss_record.append(d_loss)
        #         train generator
        for _ in range(self.args.train_generator_times):
            input_g, gt_g = \
                self.data.image2image_batch_load(self.args.batch_size,
                                                 self.batch_iterator(),
                                                 self.scale_factor)
            # wf_g = self.data.calc_wf(input_g)
            loss = self.model.train_on_batch(input_g, gt_g)
            self.loss_record.append(loss)
        return d_loss, loss

    def discriminator(self):
        """
            discriminator architecture and model definition
        """
        self.d_output = discriminator(self.d_input)

        disc = Model(inputs=self.d_input,
                     outputs=self.d_output)

        frozen_disc = Model(inputs=disc.inputs, outputs=disc.outputs)
        frozen_disc.trainable = False

        tf.keras.utils.plot_model(disc,
                                  to_file='Disc.png',
                                  show_shapes=True,
                                  dpi=64,
                                  rankdir='LR')
        visualkeras.layered_view(disc,
                                 draw_volume=False,
                                 legend=True,
                                 to_file='Disc2.png')  # write to disk
        return disc, frozen_disc

    def generator(self):
        """
            generator architecture

            params:
                model_input: tf Input object
        """
        self.model_output = rcan(self.model_input,
                                 n_rcab=self.args.n_rcab,
                                 n_res_group=self.args.n_ResGroup,
                                 channel=self.args.n_channel)
        gen = Model(inputs=self.model_input,
                    outputs=self.model_output)
        tf.keras.utils.plot_model(gen,
                                  to_file='Generator.png',
                                  show_shapes=True,
                                  dpi=64,
                                  rankdir='LR')
        visualkeras.layered_view(gen,
                                 draw_volume=False,
                                 legend=True,
                                 to_file='Generator2.png')  # write to disk
        return gen

    def loss_mse_ssim_3d(self, y_true, y_pred):
        """
            cagan paper defined this loss
        """
        ssim_para = self.args.ssim_loss
        mse_para = self.args.mse_loss
        mae_para = self.args.mae_loss

        # SSIM loss and MSE loss
        y_true_ = K.permute_dimensions(y_true, (0, 4, 1, 2, 3))
        y_pred_ = K.permute_dimensions(y_pred, (0, 4, 1, 2, 3))
        y_true_ = (y_true_ - K.min(y_true_)) / (K.max(y_true_) - K.min(y_true_))
        y_pred_ = (y_pred_ - K.min(y_pred_)) / (K.max(y_pred_) - K.min(y_pred_))

        ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(y_true_, y_pred_, 1))) / 2
        mse_loss = mse_para * K.mean(K.square(y_pred_ - y_true_))
        mae_loss = mae_para * K.mean(K.abs(y_pred_ - y_true_))

        output = mae_loss + mse_loss + ssim_loss
        return output


def create_psf_loss(psf):
    """
            WF loss calculator for builtin tf loss callback
    """
    def loss_wf(y_true, y_pred):
        """
            WF loss calculator for builtin tf loss callback
        """
        # Wide field loss

        x_wf = K.conv3d(y_pred, psf, padding='same')
        x_wf = K.pool3d(x_wf, pool_size=(2, 2, 1), strides=(2, 2, 1), pool_mode="avg")
        x_min = K.min(x_wf)
        x_wf = (x_wf - x_min) / (K.max(x_wf) - x_min)
        wf_loss = K.mean(K.square(y_true - x_wf))
        return wf_loss

    return loss_wf
