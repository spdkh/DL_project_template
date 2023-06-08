"""
    author: Parisa Daj
    date: May 10, 2022
    parsing and configuration
"""
import argparse

import pathlib
import numpy as np

from src.utils.path_helper import check_folder


def check_args(args):
    """
    checking arguments
    """
    # --checkpoint_dir
    check_folder(args.data_dir)

    # --iteration
    assert args.iteration >= 1, 'number of iterations must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    return args


def parse_args():
    """
        Define terminal input arguments
    Returns
    -------
    arguments
    """
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--data_dir", type=str,
                        # default="D:/Data/FixedCell/PFA_eGFP/cropped2d_128",
                        default="D:/Data/FairSIM/cropped3d_128_3",
                        help='The directory of the data')
    parser.add_argument('--dataset', type=str, default='FairSIM',
                        help='FixedCell or FairSIM')
    parser.add_argument('--task', type=str, default='super_resolution',
                        choices=['super_resolution'],
                        help='What kind of task are you trying to solve?')
    parser.add_argument('--dnn_type', type=str, default='cagan',
                        choices=['cagan',
                                 'srgan',
                                 'ucagan',
                                 'cgan',
                                 'srgan',
                                 'ugan',
                                 'urcan'],
                        help='The type of DNN')

    parser.add_argument("--load_weights", type=int, default=0,
                        choices=range(2))
    parser.add_argument("--mae_loss", type=float, default=1)
    parser.add_argument("--mse_loss", type=float, default=0)
    parser.add_argument("--ssim_loss", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=0.25) # gan_loss
    parser.add_argument("--beta", type=float, default=0)  # weight_wf_loss
    parser.add_argument("--gamma", type=float, default=0.1)  # weight_unrolling gamma
    parser.add_argument("--unrolling_iter", type=int, default=2,
                        choices=range(5))

    # Generator Setup
    parser.add_argument("--start_lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--train_generator_times", type=int, default=10)
    parser.add_argument("--opt", type=str, default="adam")

    # Discriminator Setup
    parser.add_argument("--d_start_lr", type=float, default=1e-6)  # 2e-5
    parser.add_argument("--d_lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--train_discriminator_times", type=int, default=1)
    parser.add_argument("--d_opt", type=str, default="adam")

    default_iterations = 5000
    parser.add_argument('--batch_size', type=int, default=2,
                        choices=range(2, 16),
                        help='The size of batch')
    parser.add_argument('--iteration', type=int,
                        default=default_iterations, help='The number of iterations to run')
    parser.add_argument("--sample_interval",
                        type=int, default=int(1 + 10 * (np.log10(1 + default_iterations // 50))))
    parser.add_argument("--validate_interval", type=int, default=5)
    parser.add_argument("--validate_num", type=int, default=5)
    parser.add_argument("--norm", type=str, default='prctile',
                        help='Image normalization Method.',
                        choices=['max',
                                'min_max',
                                'prctile'])
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')

    parser.add_argument("--n_ResGroup", type=int, default=2)
    parser.add_argument("--n_rcab", type=int, default=3)
    parser.add_argument("--n_channel", type=int, default=64)

    parser.add_argument("--n_phases", type=int, default=5)
    parser.add_argument("--n_angles", type=int, default=3)

    #               Predict

    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--gpu_memory_fraction", type=float, default=0.25)
    parser.add_argument("--model_weights", type=str,
    default=pathlib.Path(
            """D:/OneDrive - The University of Memphis/Parisa_Daj/Codes/
            caGAN_git/trained_models/2d/SIM_cropped_0.05/weights_disc_best.h5"""))

    return check_args(parser.parse_args())
