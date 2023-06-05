# -*- coding: utf-8 -*-
"""
"""

from utils.lr_controller import ReduceLROnPlateau

from models.GAN import GAN
from models.binary_classification import discriminator
from models.super_resolution import rcan
from utils.autoclip_tf import AutoClipper

import datetime
import glob
import sys
import os

import visualkeras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, LeakyReLU, ReLU

import tensorflow as tf
import numpy as np
from sewar.full_ref import uqi
from skimage.metrics import mean_squared_error as compare_mse, \
    normalized_root_mse as compare_nrmse, \
    peak_signal_noise_ratio as compare_psnr, \
    structural_similarity as compare_ssim
from matplotlib import pyplot as plt


class CAGAN(GAN):
    """

    """

    def __init__(self, args):
        GAN.__init__(self, args)

        self.d_loss_object = None
        self.lr_controller = None
        self.d_lr_controller = None
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.disc_opt = tf.keras.optimizers.Adam(args.d_start_lr,
                                                 beta_1=args.d_lr_decay_factor)

    def build_model(self):
        """

        """
        # --------------------------------------------------------------------------------
        #                              define combined model
        # --------------------------------------------------------------------------------
        self.disc, self.frozen_d = self.discriminator()

        self.gen = self.generator(self.g_input)
        # print(self.gen.summary())

        fake_hp = self.gen(inputs=self.g_input)
        judge = self.disc(fake_hp)
        label = np.zeros(self.args.batch_size)

        # last fake hp
        gen_loss = self.generator_loss(judge)

        loss_wf = create_psf_loss(self.data.psf)
        # gen_total_loss, gen_gan_loss, gen_l1_loss =\
        # self.generator_loss(judge, fake_hp, self.g_output)
        # disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        if self.args.opt == "adam":
            opt = tf.keras.optimizers.Adam(
                self.args.start_lr,
                gradient_transformers=[AutoClipper(20)]
            )
        else:
            opt = self.args.opt

        self.lr_controller = ReduceLROnPlateau(
            model=self.gen,
            factor=self.args.lr_decay_factor,
            patience=self.args.epoch * 1e-2,
            mode="min",
            min_delta=1e-2,
            cooldown=0,
            min_lr=self.args.start_lr * 1e-3,
            verbose=1,
        )

        print('debugging:', self.args.alpha, self.args.beta)
        self.gen.compile(loss=[self.loss_mse_ssim_3d, gen_loss, loss_wf],
                         optimizer=opt,
                         loss_weights=[1,
                                       self.args.alpha,
                                       self.args.beta])

        self.lr_controller_d = ReduceLROnPlateau(
            model=self.disc,
            factor=self.args.d_lr_decay_factor,
            patience=3,
            mode="min",
            min_delta=1e-2,
            cooldown=0,
            min_learning_rate=self.args.d_start_lr * 0.001,
            verbose=1,
        )

        self.d_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        checkpoint_dir = './training_checkpoints'
        # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        # checkpoint = tf.train.Checkpoint(generator_optimizer=self.args.opt,
        #                                  discriminator_optimizer=self.args.d_opt,
        #                                  generator=self.gen,
        #                                  discriminator=self.disc)
        # --------------------------------------------------------------------------------
        #                             if exist, load weights
        # --------------------------------------------------------------------------------
        if self.args.load_weights:
            if os.path.exists(self.data.save_weights_path + 'weights_best.h5'):
                self.gen.load_weights(self.data.save_weights_path + 'weights_best.h5')
                self.disc.load_weights(self.data.save_weights_path + 'weights_disc_best.h5')
                print('Loading weights successfully: ' + self.data.save_weights_path + 'weights_best.h5')
            elif os.path.exists(self.data.save_weights_path + 'weights_latest.h5'):
                self.gen.load_weights(self.data.save_weights_path + 'weights_latest.h5')
                self.disc.load_weights(self.data.save_weights_path + 'weights_disc_latest.h5')
                print('Loading weights successfully: ' + self.data.save_weights_path + 'weights_latest.h5')

    def train(self):
        """

        """

        start_time = datetime.datetime.now()
        self.lr_controller.on_train_begin()
        train_names = ['Generator_loss', 'Discriminator_loss']

        print('Training...')

        for it in range(self.args.epoch):
            loss_discriminator, loss_generator = \
                self.train_gan()
            elapsed_time = datetime.datetime.now() - start_time

            tf.print("%d epoch: time: %s, g_loss = %s, d_loss= " % (
                it + 1,
                elapsed_time,
                loss_generator),
                loss_discriminator, output_stream=sys.stdout)

            if (it + 1) % self.args.sample_interval == 0:
                self.validate(it + 1, sample=1)

            if (it + 1) % self.args.validate_interval == 0:
                self.validate(it + 1, sample=0)
                self.write_log(self.writer,
                               train_names[0],
                               np.mean(self.gloss_record),
                               it + 1)
                self.write_log(self.writer,
                               train_names[1],
                               np.mean(self.dloss_record),
                               it + 1)
                gloss_record = []
                dloss_record = []

    def train_gan(self):
        """
        todo: disc part is absolutely wrong: use pix2pix code instead
            https://www.tensorflow.org/tutorials/generative/pix2pix
        """
        disc_loss = 0
        loss_generator = 0
        batch_size_d = self.args.batch_size
        valid_d = np.ones(batch_size_d).reshape((batch_size_d, 1))
        fake_d = np.zeros(batch_size_d).reshape((batch_size_d, 1))

        # ------------------------------------
        #         train discriminator
        # ------------------------------------
        for i in range(self.args.train_discriminator_times):
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

            fake_input_d = self.gen.predict(input_d)

            # discriminator loss separate for real/fake:
            # https://stackoverflow.com/questions/49988496/loss-functions-in-gans

            # loss_discriminator = self.disc.train_on_batch(gt_d, valid_d)
            # loss_discriminator += self.disc.train_on_batch(fake_input_d, fake_d)

            with tf.GradientTape() as disc_tape:
                disc_real_output = self.disc(gt_d)
                print(np.shape(fake_input_d))
                disc_fake_output = self.disc(fake_input_d)
                disc_loss = self.discriminator_loss(disc_real_output,
                                                    disc_fake_output)

            disc_gradients = disc_tape.gradient(disc_loss,
                                                self.disc.trainable_variables)

            self.disc_opt.apply_gradients(zip(disc_gradients,
                                              self.disc.trainable_variables))

            self.dloss_record.append(disc_loss)

            # loss_disc_real = -tf.reduce_mean(tf.log(gt_d, valid_d))  # maximise
            # loss_disc_fake = -tf.reduce_mean(tf.log(fake_input_d, fake_d))
            # loss_disc = loss_disc_fake + loss_disc_real

            # train_disc = tf.train(self.lr_controller_d).minimize(loss_disc)

        # ------------------------------------
        #         train generator
        # ------------------------------------
        for i in range(self.args.train_generator_times):
            input_g, gt_g, wf_g = \
                self.data.data_loader('train',
                                      self.batch_iterator(),
                                      self.args.batch_size,
                                      self.scale_factor,
                                      self.args.beta)
            loss_generator = self.gen.train_on_batch(input_g, gt_g)
            self.gloss_record.append(loss_generator)
        return disc_loss, loss_generator

    def validate(self, epoch, sample=0):
        """
                :param epoch:
                :param sample:
                :return:
        """

        # initialization

        validate_nrmse = [np.Inf]

        # -------------------------------------------------------------------
        #                       about Tensor Board
        # -------------------------------------------------------------------

        val_names = ['val_MSE',
                     'val_SSIM',
                     'val_PSNR',
                     'val_NRMSE',
                     'val_UQI']

        patch_y, patch_x, patch_z, _ = self.data.input_dim
        validate_path = glob.glob(self.data.data_dirs['val'] + '*')
        validate_path.sort()
        # if sample == 1:
        #     validate_path = np.random.choice(validate_path, size=1)
        # elif self.args.validate_num < validate_path.__len__():
        #     validate_path = validate_path[0:self.args.validate_num]

        # save_weights_name = model_name + '-SIM_' + data_name
        #
        # save_weights_path = os.path.join(self.args.checkpoint_dir, save_weights_name)
        # sample_path = save_weights_path + 'sampled_img/'

        # check_folder(save_weights_path)

        # check_folder(sample_path)

        mses, nrmses, psnrs, ssims, uqis = [], [], [], [], []
        # imgs, imgs_gt, output = [], [], []
        # for path in validate_path:

        imgs, imgs_gt, wf_batch = \
            self.data.data_loader('val',
                                  self.batch_iterator('val'),
                                  self.args.batch_size,
                                  self.scale_factor)

        outputs = self.gen.predict(imgs)
        for output, img_gt in zip(outputs, imgs_gt):
            # predict generates [1, x, y, z, 1]
            # It is converted to [x, y, z] below
            output = np.reshape(output,
                                self.data.output_dim[:-1])

            output_proj = np.max(output, 2)

            gt_proj = np.max(np.reshape(img_gt,
                                        self.data.output_dim[:-1]),
                             2)
            mses, nrmses, psnrs, ssims, uqis = \
                self.img_comp(gt_proj,
                              output_proj,
                              mses,
                              nrmses,
                              psnrs,
                              ssims,
                              uqis)

        if sample == 0:
            # if best, save weights.best
            # self.gen.save(os.path.join(self.data.save_weights_path,
            #                       'gen_latest.h5'))
            # self.disc.save(os.path.join(self.data.save_weights_path,
            #                            'disc_latest.h5'))
            self.gen.save_weights(os.path.join(self.data.save_weights_path,
                                  'weights_gen_latest.h5'))
            self.disc.save_weights(os.path.join(self.data.save_weights_path,
                                   'weights_disc_latest.h5'))

            if min(validate_nrmse) > np.mean(nrmses):
                # self.gen.save(os.path.join(self.data.save_weights_path,
                #                            'gen_best.h5'))
                # self.disc.save(os.path.join(self.data.save_weights_path,
                #                             'disc_best.h5'))
                self.gen.save_weights(os.path.join(self.data.save_weights_path,
                                      'weights_gen_best.h5'))
                self.disc.save_weights(os.path.join(self.data.save_weights_path,
                                       'weights_disc_best.h5'))

            validate_nrmse.append(np.mean(nrmses))
            cur_lr = self.lr_controller.on_epoch_end(epoch, np.mean(nrmses))
            self.write_log(self.writer, 'lr_sr', cur_lr, epoch)
            cur_lr_d = self.lr_controller.on_epoch_end(epoch, np.mean(nrmses))
            self.write_log(self.writer, 'lr_d', cur_lr_d, epoch)
            self.write_log(self.writer, val_names[0], np.mean(mses), epoch)
            self.write_log(self.writer, val_names[1], np.mean(ssims), epoch)
            self.write_log(self.writer, val_names[2], np.mean(psnrs), epoch)
            self.write_log(self.writer, val_names[3], np.mean(nrmses), epoch)
            self.write_log(self.writer, val_names[4], np.mean(uqis), epoch)


        else:
            plt.figure(figsize=(22, 6))
            validation_id = 0
            # figures equal to the number of z patches in columns
            for j in range(patch_z):
                output_results = {'Raw Input': imgs[validation_id, :, :, j, 0],
                                  'Super Resolution Output': self.data.norm(outputs[validation_id, :, :, j, 0]),
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

            plt.savefig(self.data.sample_path + '%d.png' % epoch)  # Save sample results
            plt.close("all")  # Close figures to avoid memory leak

    def img_comp(self,
                 gt, pr,
                 mses=None, nrmses=None,
                 psnrs=None, ssims=None,
                 uqis=None):
        """
        :param gt:
        :param pr:
        :param mses:
        :param nrmses:
        :param psnrs:
        :param ssims:
        :param uqis
        :return:
        """
        if ssims is None:
            ssims = []
        if psnrs is None:
            psnrs = []
        if nrmses is None:
            nrmses = []
        if mses is None:
            mses = []
        if uqis is None:
            uqis = []

        gt, pr = np.squeeze(gt), np.squeeze(pr)
        gt = gt.astype(np.float32)
        if gt.ndim == 2:
            num = 1
            gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1]))
            pr = np.reshape(pr, (1, pr.shape[0], pr.shape[1]))
        else:
            num = np.size(gt, 0)

        for i in range(num):
            mses.append(compare_mse(self.data.norm(np.squeeze(gt[i])),
                                    self.data.norm(np.squeeze(pr[i]))))
            nrmses.append(compare_nrmse(self.data.norm(np.squeeze(gt[i])),
                                        self.data.norm(np.squeeze(pr[i]))))
            psnrs.append(compare_psnr(self.data.norm(np.squeeze(gt[i])),
                                      self.data.norm(np.squeeze(pr[i]))))
            ssims.append(compare_ssim(self.data.norm(np.squeeze(gt[i])),
                                      self.data.norm(np.squeeze(pr[i]))))
            uqis.append(uqi(self.data.norm(np.squeeze(pr[i])),
                            self.data.norm(np.squeeze(gt[i]))))
        return mses, nrmses, psnrs, ssims, uqis

    def discriminator(self):
        self.d_output = discriminator(self.d_input)

        disc = Model(inputs=self.d_input,
                     outputs=self.d_output)
        # disc.compile(loss='binary_crossentropy',
        #              optimizer=self.args.d_opt,
        #              metrics=['accuracy'])

        frozen_disc = Model(inputs=disc.inputs, outputs=disc.outputs)
        frozen_disc.trainable = False

        tf.keras.utils.plot_model(disc, to_file='Disc.png',show_shapes=True, dpi=64, rankdir='LR')
        visualkeras.layered_view(disc,  draw_volume=False,legend=True, to_file='Disc2.png')  # write to disk
        return disc, frozen_disc

    def generator(self, g_input):
        self.g_output = rcan(g_input,
                             n_rcab=self.args.n_RCAB,
                             n_res_group=self.args.n_ResGroup,
                             channel=self.args.n_channel)
        gen = Model(inputs=self.g_input,
                    outputs=self.g_output)
        tf.keras.utils.plot_model(gen, to_file='Generator.png', show_shapes=True, dpi=64, rankdir='LR')
        visualkeras.layered_view(gen,  draw_volume=False,legend=True, to_file='Generator2.png')  # write to disk
        return gen

    def generator_loss(self, disc_generated_output):
        def gen_loss(y_true, y_pred):
            gan_loss = self.loss_object(tf.ones_like(disc_generated_output),
                                        disc_generated_output)
            return gan_loss

        return gen_loss

    def loss_mse_ssim_3d(self, y_true, y_pred):
        ssim_para = self.args.ssim_loss
        mse_para = self.args.mse_loss
        mae_para = self.args.mae_loss

        # SSIM loss and MSE loss
        x = K.permute_dimensions(y_true, (0, 4, 1, 2, 3))
        y = K.permute_dimensions(y_pred, (0, 4, 1, 2, 3))
        x = (x - K.min(x)) / (K.max(x) - K.min(x))
        y = (y - K.min(y)) / (K.max(y) - K.min(y))

        ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))/2
        mse_loss = mse_para * K.mean(K.square(y - x))
        mae_loss = mae_para * K.mean(K.abs(y - x))

        output = mae_loss + mse_loss + ssim_loss
        return output


def create_psf_loss(psf):
    def loss_wf(y_true, y_pred):
        # Wide field loss

        x_wf = K.conv3d(y_pred, psf, padding='same')
        x_wf = K.pool3d(x_wf, pool_size=(2, 2, 1), strides=(2, 2, 1), pool_mode="avg")
        x_min = K.min(x_wf)
        x_wf = (x_wf - x_min) / (K.max(x_wf) - x_min)
        wf_loss = K.mean(K.square(y_true - x_wf))
        return wf_loss

    return loss_wf
