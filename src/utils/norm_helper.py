"""
    Normalization Functions
"""
import numpy as np


def prctile_norm(x_in, min_prc=0, max_prc=100):
    """
    :param x_in:
    :param min_prc:
    :param max_prc:
    :return: output
    """
    output = (x_in-np.percentile(x_in, min_prc)) \
             / (np.percentile(x_in, max_prc)
                - np.percentile(x_in, min_prc) + 1e-7)
    output[output > 1] = 1
    output[output < 0] = 0
    return output


def max_norm(x_in):
    """
    :param x_in:
    :return: output
    """
    output = x_in / np.max(x_in)
    return output


def log_norm(x_in):
    """
    :param x_in:
    :return: output
    """
    output = min_max_norm(x_in - np.log10(x_in + 1))
    return output


def min_max_norm(x_in):
    """
    :param x_in:
    :return: output
    """
    output = (x_in - np.min(x_in)) / (np.max(x_in) - np.min(x_in))
    return output
