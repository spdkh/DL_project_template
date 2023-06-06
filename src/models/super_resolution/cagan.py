# -*- coding: utf-8 -*-
"""
    channel attention GAN
    author: spdkh
    date: 2023
"""

import datetime
import glob
import sys
import os

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import visualkeras
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from src.models.gan import GAN
from src.utils.architectures.binary_classification import discriminator
from src.utils.architectures.super_resolution import rcan
from src.utils.ml_helper import AutoClipper
from src.utils.img_helper import img_comp


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

        self.d_loss_object = None
        self.lr_controller = None
        self.d_lr_controller = None

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
        self.model['disc'], _ = self.discriminator()
        self.model['gen'] = self.generator(self.model_input['gen'])

        fake_hp = self.model['gen'](inputs=self.model_input['gen'])
        judge = self.model['disc'](fake_hp)

        # last fake hp
        gen_loss = self.generator_loss(judge)

        loss_wf = create_psf_loss(self.data.psf)

        if self.args.opt == "adam":
            opt = tf.keras.optimizers.Adam(
                self.args.start_lr,
                gradient_transformers=[AutoClipper(20)]
            )
        else:
            opt = self.args.opt

        self.lr_controller = ReduceLROnPlateau(
            model=self.model['gen'],
            factor=self.args.lr_decay_factor,
            patience=self.args.iteration * 1e-2,
            mode="min",
            min_delta=1e-2,
            cooldown=0,
            min_lr=self.args.start_lr * 1e-3,
            verbose=1,
        )

        print('debugging:', self.args.alpha, self.args.beta)
        self.model['gen'].compile(loss=[self.loss_mse_ssim_3d, gen_loss, loss_wf],
                                  optimizer=opt,
                                  loss_weights=[1,
                                                self.args.alpha,
                                                self.args.beta])

        self.lr_controller_d = ReduceLROnPlateau(
            model=self.model['disc'],
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
            if os.path.exists(self.data.save_weights_path
                              + 'weights_best.h5'):
                self.model['gen'].load_weights(self.data.save_weights_path
                                               + 'weights_best.h5')
                self.model['disc'].load_weights(self.data.save_weights_path
                                                + 'weights_disc_best.h5')
                print('Loading weights successfully: '
                      + self.data.save_weights_path
                      + 'weights_best.h5')
            elif os.path.exists(self.data.save_weights_path + 'weights_latest.h5'):
                self.model['gen'].load_weights(self.data.save_weights_path
                                               + 'weights_latest.h5')
                self.model['disc'].load_weights(self.data.save_weights_path
                                                + 'weights_disc_latest.h5')
                print('Loading weights successfully: '
                      + self.data.save_weights_path
                      + 'weights_latest.h5')

    def train(self):
        """
            iterate over epochs
        """

        start_time = datetime.datetime.now()
        self.lr_controller.on_train_begin()
        train_names = ['Generator_loss', 'Discriminator_loss']

        print('Training...')

        for iteration in range(self.args.iteration):
            loss_discriminator, loss_generator = \
                self.train_gan()
            elapsed_time = datetime.datetime.now() - start_time

            tf.print("%d iteration: time: %s, g_loss = %s, d_loss= " % (
                iteration + 1,
                elapsed_time,
                loss_generator),
                     loss_discriminator, output_stream=sys.stdout)

            if (iter + 1) % self.args.sample_interval == 0:
                self.validate(iter + 1, sample=1)

            if (iter + 1) % self.args.validate_interval == 0:
                self.validate(iter + 1, sample=0)
                self.write_log(self.writer,
                               train_names[0],
                               np.mean(self.loss_record['gen']),
                               iteration + 1)
                self.write_log(self.writer,
                               train_names[1],
                               np.mean(self.loss_record['disc']),
                               iteration + 1)
                loss_record['gen'] = []
                loss_record['disc'] = []

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
            input_d, gt_d, wf_d = \
                self.data.data_loader('train',
                                      self.batch_iterator(),
                                      batch_size_d,
                                      self.scale_factor,
                                      self.args.beta)

            fake_input_d = self.model['gen'].predict(input_d)

            # discriminator loss separate for real/fake:
            # https://stackoverflow.com/questions/49988496/loss-functions-in-gans
            with tf.GradientTape() as disc_tape:
                disc_real_output = self.model['disc'](gt_d)
                print(np.shape(fake_input_d))
                disc_fake_output = self.model['disc'](fake_input_d)
                loss['disc'] = self.discriminator_loss(disc_real_output,
                                                    disc_fake_output)

            disc_gradients = disc_tape.gradient(loss['disc'],
                                                self.model['disc'].trainable_variables)

            self.disc_opt.apply_gradients(zip(disc_gradients,
                                              self.model['disc'].trainable_variables))

            self.loss_record['disc'].append(loss['disc'])
        #         train generator
        for _ in range(self.args.train_generator_times):
            input_g, gt_g, wf_g = \
                self.data.data_loader('train',
                                      self.batch_iterator(),
                                      self.args.batch_size,
                                      self.scale_factor,
                                      self.args.beta)
            loss['gen'] = self.model['gen'].train_on_batch(input_g, gt_g)
            self.loss_record['gen'].append(loss['gen'])
        return loss['disc'], loss['gen']

    def validate(self, iteration, sample=0):
        """
                :param iteration: current iteration number
                :param sample: sample id
                :return:
        """
        validate_nrmse = [np.Inf]

        val_names = ['val_MSE',
                     'val_SSIM',
                     'val_PSNR',
                     'val_NRMSE',
                     'val_UQI']

        patch_z = self.data.input_dim[2]
        validate_path = glob.glob(self.data.data_dirs['val'] + '*')
        validate_path.sort()

        metrics = {'mse':[],
                   'nrmse': [],
                   'psnr': [],
                   'ssim': [],
                   'uqi': []}

        imgs, imgs_gt, wf_batch = \
            self.data.data_loader('val',
                                  self.batch_iterator('val'),
                                  self.args.batch_size,
                                  self.scale_factor)

        outputs = self.model['gen'].predict(imgs)
        for output, img_gt in zip(outputs, imgs_gt):
            # predict generates [1, x, y, z, 1]
            # It is converted to [x, y, z] below
            output = np.reshape(output,
                                self.data.output_dim[:-1])

            output_proj = np.max(output, 2)

            gt_proj = np.max(np.reshape(img_gt,
                                        self.data.output_dim[:-1]),
                             2)
            metrics.values = \
                img_comp(gt_proj,
                         output_proj,
                         metrics['mse'],
                         metrics['nrmse'],
                         metrics['psnr'],
                         metrics['ssim'],
                         metrics['uqi'])

        if sample == 0:
            self.model['gen'].save_weights(os.path.join(self.data.save_weights_path,
                                                        'weights_gen_latest.h5'))
            self.model['disc'].save_weights(os.path.join(self.data.save_weights_path,
                                                         'weights_disc_latest.h5'))

            if min(validate_nrmse) > np.mean(metrics['nrmse']):
                self.model['gen'].save_weights(os.path.join(self.data.save_weights_path,
                                                            'weights_gen_best.h5'))
                self.model['disc'].save_weights(os.path.join(self.data.save_weights_path,
                                                             'weights_disc_best.h5'))

            validate_nrmse.append(np.mean(metrics['nrmse']))
            cur_lr = self.lr_controller.on_iteration_end(iteration, np.mean(metrics['nrmse']))
            self.write_log(self.writer, 'lr_sr', cur_lr, iteration)
            cur_lr_d = self.lr_controller.on_iteration_end(iteration, np.mean(metrics['nrmse']))
            self.write_log(self.writer, 'lr_d', cur_lr_d, iteration)
            self.write_log(self.writer, val_names[0], np.mean(metrics['mse']), iteration)
            self.write_log(self.writer, val_names[1], np.mean(metrics['ssim']), iteration)
            self.write_log(self.writer, val_names[2], np.mean(metrics['psnr']), iteration)
            self.write_log(self.writer, val_names[3], np.mean(metrics['nrmse']), iteration)
            self.write_log(self.writer, val_names[4], np.mean(metrics['uqi']), iteration)
        else:
            plt.figure(figsize=(22, 6))
            validation_id = 0
            # figures equal to the number of z patches in columns
            for j in range(patch_z):
                output_results = {'Raw Input': imgs[validation_id, :, :, j, 0],
                                  'SR Output': self.data.norm(outputs[validation_id, :, :, j, 0]),
                                  'Ground Truth': imgs_gt[validation_id, :, :, j, 0]}

                plt.title('Z = ' + str(j))
                for i, (label, img) in enumerate(output_results.items()):
                    # first row: input image average of angles and phases
                    # second row: resulting output
                    # third row: ground truth
                    plt.subplot(3, patch_z, j + patch_z * i + 1)
                    plt.ylabel(label)
                    plt.imshow(img, cmap=plt.get_cmap('hot'))

                    plt.gca().axes.yaxis.set_ticklabels([])
                    plt.gca().axes.xaxis.set_ticklabels([])
                    plt.gca().axes.yaxis.set_ticks([])
                    plt.gca().axes.xaxis.set_ticks([])
                    plt.colorbar()

            plt.savefig(self.data.sample_path + '%d.png' % iteration)  # Save sample results
            plt.close("all")  # Close figures to avoid memory leak

    def discriminator(self):
        """
            discriminator architecture and model definition
        """
        self.model_output['disc'] = discriminator(self.model_input['disc'])

        disc = Model(inputs=self.model_input['disc'],
                     outputs=self.model_output['disc'])

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

    def generator(self, model_input):
        """
            generator architecture

            params:
                model_input: tf Input object
        """
        self.model_output['gen'] = rcan(model_input,
                                        n_rcab=self.args.n_rcab,
                                        n_res_group=self.args.n_ResGroup,
                                        channel=self.args.n_channel)
        gen = Model(inputs=self.model_input['gen'],
                    outputs=self.model_output['gen'])
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

    def generator_loss(self, generated_output):
        """
                generator loss calculator for builtin tf loss callback
        """

        def gen_loss(y_true, y_pred):
            """
                    generator loss calculator for builtin tf loss callback
            """
            gan_loss = self.loss_object(tf.ones_like(generated_output),
                                        generated_output)
            return gan_loss

        return gen_loss

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
