import argparse
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Input, add, multiply

import imageio
import os
from models import *
from utils.fcns import prctile_norm
import tifffile as tiff
from data.data import Data

from models.super_resolution import rcan


from utils.config import parse_args

"""
    author: SPDKH
    date: Nov 2, 2023
"""

def main():
    """
        Main Predicting Function
    """
    # parse arguments
    args = parse_args()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # data = Data(args)
    dnn = tf.keras.models.load_model(args.model_weights)

    output_name = 'output_' + args.dnn_type + '-'
    test_images_path = args.test_dir
    output_dir = args.result_dir + '\\' + output_name

    # --------------------------------------------------------------------------------
    #                              glob test data path
    # --------------------------------------------------------------------------------

    img_path = [test_images_path]
    img_path.sort()
    n_channel = 15

    print(img_path)
    img = tiff.imread(img_path[0])
    shape = img.shape
    input_y, input_x = shape[1], shape[2]
    input_z = shape[0] // n_channel
    output_dir = output_dir + 'SIM'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Processing ' + test_images_path + '...')
    im_count = 0
    for curp in img_path:
        print(curp)
        img = tiff.imread(curp)
        img = np.array(img).reshape((n_channel, input_z, input_y, input_x),
                                    order='F').transpose((2, 3, 1, 0))
        img = img[np.newaxis, :]

        img = prctile_norm(img)
        pr = prctile_norm(np.squeeze(dnn.predict(img)))

        outName = curp.replace(test_images_path, output_dir)

        pr = np.transpose(65535 * pr, (2, 0, 1)).astype('uint16')
        tiff.imwrite(outName, pr, dtype='uint16')


if __name__ == '__main__':
    main()



