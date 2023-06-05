"""
    Discriminator Architectures
"""
from tensorflow.keras.layers import Dense, Flatten, Input, add, multiply
from tensorflow.keras.layers import Conv3D, UpSampling3D, LeakyReLU, Lambda, ReLU
from utils.common import global_average_pooling3d, conv_block3d


def discriminator(inputs):
    """

    Parameters
    ----------
    input_shape

    Returns
    -------

    """
    x0 = Conv3D(32, kernel_size=3, padding='same')(inputs)
    x0 = LeakyReLU(alpha=0.1)(x0)

    x1 = conv_block3d(x0, (32, 64))
    x2 = conv_block3d(x1, (128, 256))
    x3 = Lambda(global_average_pooling3d)(x2)

    y0 = Flatten(input_shape=(1, 1))(x3)
    y1 = Dense(128)(y0)
    y1 = LeakyReLU(alpha=0.1)(y1)
    outputs = Dense(1, activation='sigmoid')(y1)
    return outputs
