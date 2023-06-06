# -*- coding: utf-8 -*-
"""
    GAN abstract
    author: SPDKH
    date: 2023
"""

from abc import abstractmethod

import tensorflow as tf
from tensorflow.keras.layers import Input

from src.models.dnn import DNN


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

        self.model = {'gen': None,
                      'disc': None}
        self.model_input = {'gen': Input(self.data.input_dim),
                            'disc': Input(self.data.output_dim)}
        self.model_output = {'gen': None,
                            'disc': None}

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_record = {'disc': [], 'gen': []}
        self.batch_id = {'train': 0, 'val': 0, 'test': 0}

        self.lr_controller_d = None

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

    def generator_loss(self, fake_output):
        """
            calculate generator loss

            params:
                fake_output: tf tensor
                    generated images

            returns:
                discriminator loss value
        """
        return self.loss_object(tf.ones_like(fake_output),
                                fake_output)
