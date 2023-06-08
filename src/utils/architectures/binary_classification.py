"""
    Discriminator Architectures
"""
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import GlobalAveragePooling3D
from src.utils.ml_helper import conv_block_3d


def discriminator(input_shape):
    """

    Parameters
    ----------
    input_shape

    Returns
    -------

    """
    in_0 = conv_block_3d(input_shape, 32, 3)
    in_1 = conv_block_3d(in_0, 64, 3)
    in_2 = conv_block_3d(in_1, 128, 3)
    in_3 = conv_block_3d(in_2, 256, 3)

    in_4 = GlobalAveragePooling3D()(in_3)

    out_0 = Flatten(input_shape=(1, 1))(in_4)
    out_1 = Dense(128)(out_0)
    out_1 = LeakyReLU(alpha=0.1)(out_1)
    outputs = Dense(1, activation='sigmoid')(out_1)
    return outputs
