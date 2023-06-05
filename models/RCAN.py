# Copyright 2023 The Improved caGAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains the definition of the Residual Channel Attention
   Networks (RCAN) architecture.

As described in https://openaccess.thecvf.com/content_ECCV_2018/html/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.html.

  Image Super-Resolution Using Very Deep Residual Channel Attention Networks

  Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, Yun Fu

Example usage:
    conda activate tf_gpu
    python -m train --dnn_type RCAN --epoch 400 --sample_interval 10 --validate_interval 20

Experiment 01: Use 3.5 Implementation Details
    --n_ResGroup 3 --n_RCAB 5 --checkpoint_dir experiment01 --data_dir D:\Data\FairSIM\cropped3d_128_3

Experiment 03:
    --n_ResGroup 3 --n_RCAB 5 --checkpoint_dir experiment03 --data_dir D:\Data\FixedCell\PFA_eGFP\cropped3d_128_3

tensorboard --logdir=

"""
import datetime
import glob
import os
import sys

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from data.fairsim import FairSIM
from data.fixed_cell import FixedCell
from models.DNN import DNN
from models.super_resolution import rcan
from utils.autoclip_tf import AutoClipper
from utils.fcns import img_comp
from utils.lr_controller import ReduceLROnPlateau


class RCAN(DNN):
    def __init__(self, args):
        DNN.__init__(self, args)

        if "FixedCell" in self.args.data_dir:
            self.data = FixedCell(self.args)
        elif "FairSIM" in self.args.data_dir:
            self.data = FairSIM(self.args)

        self.input = Input(self.data.input_dim)
        self.output = self.data.output_dim

        self.writer_training = tf.summary.create_file_writer(
            os.path.join(self.data.log_path, "train"))
        self.writer_val = tf.summary.create_file_writer(
            os.path.join(self.data.log_path, "val"))
        self.lr_controller = None

        # self.loss_object = loss_mse_ssim_3d
        self.loss_object = tf.keras.losses.MeanAbsoluteError()
        self.batch_id = {'train': 0, 'val': 0, 'test': 0}

    def build_model(self):
        sys.setrecursionlimit(10 ** 4)
        output = rcan(
            self.input, scale=2,
            channel=self.args.n_channel,
            n_res_group=self.args.n_ResGroup,
            n_rcab=self.args.n_RCAB)
        # print(output)
        self.model = Model(inputs=self.input, outputs=output)

        # for layer in self.model.layers:
        #     print(layer.output_shape)
        # print(self.output)

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

    def train(self):
        start_time = datetime.datetime.now()
        self.lr_controller.on_train_begin()
        loss_record = []

        train_names = ['SR_loss']

        for it in range(self.args.epoch):
            # batch_id flag for iteration number including the inner loops
            temp_loss = []
            for b_id in range(8):
                batch_id = self.batch_iterator()
                if batch_id == 0:
                    # print(b_id, "is zero.")
                    break

                input_g, gt_g, wf_d = self.data.data_loader(
                    'train',
                    self.batch_iterator(),
                    self.args.batch_size,
                    self.scale_factor
                )

                loss = self.model.train_on_batch(input_g, gt_g)
                temp_loss.append(loss)
            loss_record.append(np.mean(temp_loss))

            elapsed_time = datetime.datetime.now() - start_time
            print(f"{it + 1} epoch: time: {elapsed_time}, loss = {loss}")

            if (it + 1) % self.args.sample_interval == 0:
                self.validate(it + 1, sample=1)

            if (it + 1) % self.args.validate_interval == 0:
                self.validate(it + 1, sample=0)
                self.write_log(self.writer_training, train_names[0], np.mean(loss_record), it + 1)
                loss_record = []

    def validate(self, epoch, sample=0):
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

        mses, nrmses, psnrs, ssims, uqis = [], [], [], [], []

        # for path in validate_path:
        imgs, imgs_gt, imgs_wf = self.data.data_loader('val',
                                                       self.batch_iterator('val'),
                                                       self.args.batch_size,
                                                       self.scale_factor)

        outputs = self.model.predict(imgs)
        for output, img_gt in zip(outputs, imgs_gt):
            output = np.reshape(output,
                                self.data.output_dim[:-1])

            output_proj = np.max(output, 2)

            gt_proj = np.max(np.reshape(img_gt,
                                        self.data.output_dim[:-1]),
                             2)
            mses, nrmses, psnrs, ssims, uqis = img_comp(gt_proj,
                                                        output_proj,
                                                        mses,
                                                        nrmses,
                                                        psnrs,
                                                        ssims,
                                                        uqis)

        if sample == 0:
            # if best, save weights.best
            self.model.save_weights(self.data.save_weights_path +
                                    'weights_latest.h5')

            if min(validate_nrmse) > np.mean(nrmses):
                self.model.save_weights(self.data.save_weights_path +
                                        'weights_best.h5')

            validate_nrmse.append(np.mean(nrmses))

            cur_lr = self.lr_controller.on_epoch_end(epoch, np.mean(nrmses))
            self.write_log(self.writer_training, 'lr_g', cur_lr, epoch)

            self.write_log(self.writer_val, val_names[0], np.mean(mses), epoch)
            self.write_log(self.writer_val, val_names[1], np.mean(ssims), epoch)
            self.write_log(self.writer_val, val_names[2], np.mean(psnrs), epoch)
            self.write_log(self.writer_val, val_names[3], np.mean(nrmses), epoch)
            self.write_log(self.writer_val, val_names[4], np.mean(uqis), epoch)

        else:
            plt.figure(figsize=(22, 6))

            # figures equal to the number of z patches in columns
            for j in range(patch_z):
                output_results = {'Raw Input': imgs[0, :, :, j, 0],
                                  'Super Resolution Output': outputs[0, :, :, j, 0],
                                  'Ground Truth': imgs_gt[0, :, :, j, 0]}

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

    def visualize_results(self, epoch):
        pass
