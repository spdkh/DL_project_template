"""
    caGAN project functions

Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import division

import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf
# import tensorflow.contrib.slim as slim
#
# from keras.datasets import mnist

import numpy as np
from sewar.full_ref import uqi
from skimage.metrics import mean_squared_error as compare_mse, \
    normalized_root_mse as compare_nrmse, \
    peak_signal_noise_ratio as compare_psnr, \
    structural_similarity as compare_ssim


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale=False):
    if grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, image)
    return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


""" Drawing Tools """


def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    """
    borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
    """
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    """

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def reorder(img, phases=5, angles=3):
    """
        Change the z data order from angles, z, phases
        to z, angles, phases
        todo: add plot_besides function
    :param img:
    :param phases:
    :param angles:
    :return:
    """

    [n_zs, n_x, n_y] = np.shape(img)
    n_z = n_zs // (angles * phases)
    five_d_img = np.reshape(img, (angles, n_z, phases, n_x, n_y))
    # swap angles with z
    new_img = five_d_img.swapaxes(1, 0)
    return np.reshape(new_img, (n_zs, n_x, n_y))


def prctile_norm(x_in, min_prc=0, max_prc=100):
    """

    :param x_in:
    :param min_prc:
    :param max_prc:
    :return: output
    """
    output = (x_in - np.percentile(x_in, min_prc)) / (np.percentile(x_in, max_prc)
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


def min_max_norm(x_in):
    """
    :param x_in:
    :return: output

    todo: vip
    """
    output = x_in
    return output


def img_comp(gt, pr, mses=None, nrmses=None, psnrs=None, ssims=None, uqis=None):
    """
    :param gt:
    :param pr:
    :param mses:
    :param nrmses:
    :param psnrs:
    :param ssims:
    :param uqis
    :return:
    """
    if ssims is None:
        ssims = []
    if psnrs is None:
        psnrs = []
    if nrmses is None:
        nrmses = []
    if mses is None:
        mses = []
    if uqis is None:
        uqis = []

    gt, pr = np.squeeze(gt), np.squeeze(pr)
    gt = gt.astype(np.float32)
    if gt.ndim == 2:
        num = 1
        gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1]))
        pr = np.reshape(pr, (1, pr.shape[0], pr.shape[1]))
    else:
        num = np.size(gt, 0)

    for i in range(num):
        mses.append(compare_mse(prctile_norm(np.squeeze(gt[i])),
                                prctile_norm(np.squeeze(pr[i]))))
        nrmses.append(compare_nrmse(prctile_norm(np.squeeze(gt[i])),
                                    prctile_norm(np.squeeze(pr[i]))))
        psnrs.append(compare_psnr(prctile_norm(np.squeeze(gt[i])),
                                  prctile_norm(np.squeeze(pr[i]))))
        ssims.append(compare_ssim(prctile_norm(np.squeeze(gt[i])),
                                  prctile_norm(np.squeeze(pr[i]))))
        uqis.append(uqi(prctile_norm(np.squeeze(pr[i])),
                        prctile_norm(np.squeeze(gt[i]))))
    return mses, nrmses, psnrs, ssims, uqis


def fix_path(path):
    """

    :param path:
    :return:
    """
    return path.replace('/', '\\')


def diffxy(img, order=3):
    """

    :param img:
    :param order:
    :return:
    """
    for _ in range(order):
        img = prctile_norm(img)
        d = np.zeros_like(img)
        dx = (img[1:-1, 0:-2] + img[1:-1, 2:]) / 2
        dy = (img[0:-2, 1:-1] + img[2:, 1:-1]) / 2
        d[1:-1, 1:-1] = img[1:-1, 1:-1] - (dx + dy) / 2
        d[d < 0] = 0
        img = d
    return img


def rm_outliers(img, order=3, thresh=0.2):
    """

    :param img:
    :param order:
    :param thresh:
    :return:
    """
    img_diff = diffxy(img, order)
    mask = img_diff > thresh
    img_rm_outliers = img
    img_mean = np.zeros_like(img)
    for i in [-1, 1]:
        for ax in range(0, 2):
            img_mean = img_mean + np.roll(img, i, axis=ax)
    img_mean = img_mean / 4
    img_rm_outliers[mask] = img_mean[mask]
    img_rm_outliers = prctile_norm(img_rm_outliers)
    return img_rm_outliers
