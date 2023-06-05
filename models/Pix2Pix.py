"""
    spdkh
    Feb 8, 2023
    source: https://www.tensorflow.org/tutorials/generative/pix2pix
"""
from utils.lr_controller import ReduceLROnPlateau
from data.fixed_cell import FixedCell
from models.GAN import GAN
from models.binary_classification import discriminator
from models.super_resolution import rcan

import os
import pathlib
import time
import datetime

import tensorflow as tf

from matplotlib import pyplot as plt
from IPython import display

from utils.lr_controller import ReduceLROnPlateau
from data.fixed_cell import FixedCell
from models.GAN import GAN
from models.binary_classification import discriminator
from models.super_resolution import rcan

import datetime
import glob

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Dense, Flatten, Input, add, multiply
import tensorflow as tf

from matplotlib import pyplot as plt
import matplotlib


class Pix2Pix(GAN):
    def __init__(self, args):
        GAN.__init__(self, args)

        print('Pix2Pix')
        self.data = FixedCell(self.args)
        self.g_input = Input(self.data.input_dim)
        self.d_input = Input(self.data.output_dim)

        self.g_output = None
        self.d_output = None

        self.writer = None

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.batch_id = {'train': 0, 'val': 0, 'test': 0}
        # optimizer_d = self.args.d_opt
        # optimizer_g = self.args.opt

        self.disc = None
        self.frozen_d = None
        self.gen = None
        self.lr_controller_g = None
        self.lr_controller_d = None
        self.dloss_record = []
        self.gloss_record = []
        self.scale_factor = int(self.data.output_dim[0] / \
                                self.data.input_dim[0])


def resize(self, input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image