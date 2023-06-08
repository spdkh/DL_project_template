# -*- coding: utf-8 -*-
"""
    GAN abstract
    author: SPDKH
    date: 2023
"""
import sys
from abc import abstractmethod
import datetime
import glob

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input

from src.models.dnn import DNN
from src.utils.img_helper import img_comp
from src.utils import const


class GAN(DNN):
    """
        Abstract class for any GAN architecture
    """

    def __init__(self, args):
        """
            params:
                args: argparse object
        """
        DNN.__init__(self, args)

        self.d_model = None
        self.d_input = Input(self.data.output_dim)
        self.d_output = None

        self.d_loss_record = []

        self.d_loss_object = None
        self.d_lr_controller = None

    def train_epoch(self, batch_log: bool = 0):
        """
            Training process per epoch
            loop over all data samples / number of batches
            train per batch to complete an epoch
            todo: update the loop
        """
        start_time = datetime.datetime.now()
        self.lr_controller.on_train_begin()
        batch_id = -1
        iteration = 0
        while batch_id != 0:
            iteration += 1
            batch_id = self.batch_iterator('train')
            print(batch_id)
            loss_discriminator, loss_generator = \
                self.train_gan()
            elapsed_time = datetime.datetime.now() - start_time

            if batch_log:
                tf.print("%d batch iteration: time: %s, g_loss = %s, d_loss= " % (
                    iteration + 1,
                    elapsed_time,
                    loss_generator),
                         loss_discriminator, output_stream=sys.stdout)

            if (iteration) % self.args.sample_interval == 0:
                self.validate(iteration, sample=1)

            self.loss_record.append(loss_generator)
            self.d_loss_record.append(loss_discriminator)

        return np.mean(self.loss_record), np.mean(self.d_loss_record)

    def validate(self, iteration, sample=0):
        """
                :param iteration: current iteration number
                :param sample: sample id
                :return:
                todo: review
        """
        validate_nrmse = [np.Inf]

        val_names = ['val_MSE',
                     'val_SSIM',
                     'val_PSNR',
                     'val_NRMSE',
                     'val_UQI']

        validate_path = glob.glob(self.data.data_dirs['val'] + '*')
        validate_path.sort()

        metrics = {'mse': [],
                   'nrmse': [],
                   'psnr': [],
                   'ssim': [],
                   'uqi': []}

        imgs, imgs_gt = \
            self.data.image2image_batch_load(self.args.batch_size,
                                             self.batch_iterator('val'),
                                             self.scale_factor,
                                             'val')

        outputs = self.model.predict(imgs)
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
            self.model.save_weights(const.WEIGHTS_DIR
                                    / 'weights_gen_latest.h5')
            self.d_model.save_weights(const.WEIGHTS_DIR
                                      / 'weights_disc_latest.h5')

            if min(validate_nrmse) > np.mean(metrics['nrmse']):
                self.model.save_weights(const.WEIGHTS_DIR
                                        / 'weights_gen_best.h5')
                self.d_model.save_weights(const.WEIGHTS_DIR
                                          / 'weights_disc_best.h5')

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
            for j in range(self.data.input_dim[2]):
                output_results = {
                    'WF Raw Input': self.data.calc_wf(imgs)[validation_id, :, :, j, 0],
                    'SR Output': self.data.norm(outputs[validation_id, :, :, j, 0]),
                    'Ground Truth': imgs_gt[validation_id, :, :, j, 0]
                }

                plt.title('Z = ' + str(j))
                for i, (label, img) in enumerate(output_results.items()):
                    # first row: input image average of angles and phases
                    # second row: resulting output
                    # third row: ground truth
                    plt.subplot(3,
                                self.data.input_dim[2],
                                j + self.data.input_dim[2] * i + 1)
                    plt.ylabel(label)
                    plt.imshow(img, cmap=plt.get_cmap('hot'))

                    plt.gca().axes.yaxis.set_ticklabels([])
                    plt.gca().axes.xaxis.set_ticklabels([])
                    plt.gca().axes.yaxis.set_ticks([])
                    plt.gca().axes.xaxis.set_ticks([])
                    plt.colorbar()

            plt.savefig(const.SAMPLE_DIR + '%d.png' % iteration)  # Save sample results
            plt.close("all")  # Close figures to avoid memory leak

    @abstractmethod
    def train_gan(self):
        """
            train generator and discriminator altogether
            for one iteration
        """

    @abstractmethod
    def discriminator(self):
        """
            call discriminator function
        """

    @abstractmethod
    def generator(self):
        """
            call generator function
        """

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """
            calculate discriminator loss

            params:
                disc_real_output: tf tensor
                    result of applying disc to the gt
                disc_generated_output: tf tensor
                    result of applying disc to the gt

            returns:
                discriminator loss value
        """
        real_loss = self.loss_object(tf.ones_like(disc_real_output),
                                     disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output),
                                          disc_generated_output)
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def generator_loss(self, generated_output):
        """
            calculate generator loss

            params:
                fake_output: tf tensor
                    generated images

            returns:
                discriminator loss value
        """

        def gen_loss(y_true, y_pred):
            """
                    generator loss calculator for builtin tf loss callback
            """
            gan_loss = self.loss_object(tf.ones_like(generated_output),
                                        generated_output)
            return gan_loss

        return gen_loss
