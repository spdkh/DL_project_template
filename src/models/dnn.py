"""
    author: SPDKH
    todo: complete
"""
from __future__ import division
import sys
import os
from abc import ABC, abstractmethod
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from src.utils import const


class DNN(ABC):
    """
        Abstract class for DNN architectures
    """

    def __init__(self, args):
        """
            args: argparse object
        """
        self.batch_id = {'train': 0, 'val': 0, 'test': 0}

        self.args = args
        print(self.args)
        print('Init DNN Arch:', self.args.dnn_type)

        module_name = '.'.join(['data',
                                args.dataset.lower()])
        dataset_module = __import__(module_name,
                                    fromlist=[args.dataset])
        self.data = getattr(dataset_module,
                            args.dataset)(self.args)

        self.scale_factor = int(self.data.output_dim[0]
                                / self.data.input_dim[0])

        self.model_input = Input(self.data.input_dim)
        self.model_output = None
        self.model = None
        self.optimizer = self.args.opt
        self.lr_controller = None
        self.loss_record = []
        self.d_loss_record = []

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.writer = tf.summary.create_file_writer(str(const.LOG_DIR))

        super().__init__()

    def batch_iterator(self, mode='train'):
        """
            takes care of loading batches iteratively
        """
        # how many total data in that mode exists
        data_size = len(os.listdir(self.data.data_dirs['x' + mode]))
        if data_size // self.args.batch_size - 1 <= self.batch_id[mode]:
            self.batch_id[mode] = 0
        else:
            self.batch_id[mode] += 1
        return self.batch_id[mode]

    def train(self):
        """
            iterate over epochs
        """
        start_time = datetime.datetime.now()
        self.lr_controller.on_train_begin()
        train_names = ['Generator_loss', 'Discriminator_loss']

        print('Training...')

        for iteration in range(self.args.iteration):
            elapsed_time = datetime.datetime.now() - start_time

            loss_generator, loss_discriminator = self.train_epoch()

            tf.print("%d epoch: time: %s, g_loss = %s, d_loss= " % (
                iteration + 1,
                elapsed_time,
                loss_generator),
                     loss_discriminator, output_stream=sys.stdout)

            if (iteration) % self.args.validate_interval == 0:
                self.validate(iteration, sample=0)
                self.write_log(self.writer,
                               train_names[0],
                               np.mean(self.loss_record),
                               iteration)
                self.write_log(self.writer,
                               train_names[1],
                               np.mean(self.d_loss_record),
                               iteration)
                self.d_loss_record = []
                self.loss_record = []

            self.loss_record.append(loss_generator)
            self.d_loss_record.append(loss_discriminator)

    @abstractmethod
    def train_epoch(self):
        """
            Training process per epoch
            loop over all data samples / number of batches
            train per batch to complete an epoch
        """

    @abstractmethod
    def build_model(self):
        """
            function to define the model,
            loss functions,
            optimizers,
            learning rate controller,
            compile model,
            load initial weights if available
        """

    @abstractmethod
    def train_epoch(self, batch_log: bool = 0):
        """
            iterate over batches
        """

    @abstractmethod
    def validate(self, iteration, sample=0):
        """
            validate and write logs
        """

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
