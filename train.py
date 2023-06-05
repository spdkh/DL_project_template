"""
    author: SPDKH
    date: Nov 2, 2023
"""

import sys
import tensorflow as tf

# ________________ architecture Variants
from models.SRGAN import SRGAN
from models.CAGAN import CAGAN
from models.UCAGAN import UCAGAN
from models.CGAN import CGAN
from models.DNN import DNN
from models.RCAN import RCAN
from models.URCAN import URCAN

from utils.config import parse_args

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def main():
    """
        Main Training Function
    """

    # parse arguments
    args = parse_args()
    if args is None:
        sys.exit()
    tf.random.set_seed(args.seed)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # gpu_options = tf.compat.v1.GPUOptions(TF_GPU_ALLOCATOR=cuda_malloc_async, per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    # tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # open session
    model_fns = {'CAGAN': CAGAN,
                 'SRGAN': SRGAN,
                 'CGAN': CGAN,
                 'DNN': DNN,
                 'UCAGAN': UCAGAN,
                 'RCAN': RCAN,
                 'URCAN': URCAN}

    # declare instance for GAN
    dnn = model_fns[args.dnn_type](args)

    # build graph
    dnn.build_model()

    # show network architecture
    # show_all_variables()
    #
    # launch the graph in a session
    dnn.train()
    print(" [*] Training finished!")
    #
    # # visualize learned generator
    # dnn.visualize_results(args.epoch-1)
    # print(" [*] Validation finished!")


if __name__ == '__main__':
    main()
