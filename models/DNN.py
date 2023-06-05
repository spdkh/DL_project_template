"""
    author: SPDKH
    todo: complete
"""
from __future__ import division
import os
import re
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from data.data import Data
from data.fairsim import FairSIM
from data.fixed_cell import FixedCell


class DNN(ABC):
    """
        Abstract class for DNN architectures
    """

    def __init__(self, args):
        """
            todo: what other info needed from load_sample?
            todo: write load_sample
            todo: figure out what is z_dim in CGAN
            todo: fix optimizer
        """
        self.model = Model()
        self.args = args
        # self.data = Data(self.args)
        self.optimizer = self.args.opt

        print('Init', self.args.dnn_type)

        if "FixedCell" in self.args.data_dir:
            self.data = FixedCell(self.args)
        elif "FairSIM" in self.args.data_dir:
            self.data = FairSIM(self.args)

        self.scale_factor = int(self.data.output_dim[0] / \
                                self.data.input_dim[0])

        self.writer = tf.summary.create_file_writer(self.data.log_path)

        for param, val in vars(self.args).items():
            self.write_log(self.writer, param, str(val), mode='')

        super().__init__()

    def batch_iterator(self, mode='train'):
        # how many total data in that mode exists
        data_size = len(os.listdir(self.data.data_dirs['x' + mode]))
        if data_size // self.args.batch_size - 1 <= self.batch_id[mode]:
            self.batch_id[mode] = 0
        else:
            self.batch_id[mode] += 1
        return self.batch_id[mode]

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    # --------------------------------------------------------------------------------
    #                             Sample and validate
    # --------------------------------------------------------------------------------
    @abstractmethod
    def validate(self, epoch, sample=0):
        pass

    @abstractmethod
    def visualize_results(self, epoch):
        pass

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.args.dnn_type, self.args.dataset_name,
            self.args.batch_size)

    def save(self, checkpoint_dir, step):
        """
        todo: test
        Parameters
        ----------
        checkpoint_dir
        step

        Returns
        -------
        callback
        """
        checkpoint_dir = os.path.join(checkpoint_dir,
                                      self.model_dir,
                                      self.args.dnn_type)
        checkpoint_path = os.path.join(checkpoint_dir,
                                       "-{epoch:04d}.ckpt")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_freq=step * self.args.batch_size)

        self.model.save_weights(checkpoint_path.format(epoch=0))
        return cp_callback

    def load(self, checkpoint_dir):
        """
        todo: test
        source: https://www.tensorflow.org/tutorials/keras/save_and_load

        Parameters
        ----------
        checkpoint_dir

        Returns
        -------

        """
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir,
                                      self.model_dir,
                                      self.model_name)
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        if checkpoint_dir:
            ckpt_name = os.path.basename(checkpoint_dir)
            ckpt = self.model.load_weights(latest)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def write_log(self, writer, names, logs, batch_no=0, mode='float'):
        """
        todo: test
        Parameters
        ----------
        names
        logs
        batch_no
        """
        with writer.as_default():
            if mode == 'float':
                tf.summary.scalar(names, logs, step=batch_no)
            else:
                tf.summary.text(names,
                                tf.convert_to_tensor(str(logs),
                                                     dtype=tf.string),
                                step=batch_no)
            writer.flush()
