"""
    author: SPDKH
    todo: complete
"""
from __future__ import division
import os
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras.models import Model


class DNN(ABC):
    """
        Abstract class for DNN architectures
    """

    def __init__(self, args):
        """
            args: argparse object
        """
        self.batch_id = 0
        self.model = Model()
        self.args = args
        # self.data = Data(self.args)
        self.optimizer = self.args.opt

        print('Init', self.args.dnn_type)

        module_name = '.'.join(['data',
                                args.dataset.lower()])
        dataset_module = __import__(module_name,
                                    fromlist=[args.dataset])
        self.data = getattr(dataset_module,
                            args.dataset)(self.args)

        self.scale_factor = int(self.data.output_dim[0]
                                / self.data.input_dim[0])

        self.writer = tf.summary.create_file_writer(self.data.log_path)

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
    def epoch(self):
        """
            iterate over batches
        """

    @abstractmethod
    def train(self):
        """
            iterate over epochs
        """

    @abstractmethod
    def validate(self, iteration, sample=0):
        """
            validate and write logs
        """
