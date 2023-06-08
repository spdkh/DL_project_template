"""
    author: SPDKH
    date: Nov 2, 2023
"""

import sys

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from src.utils.config import parse_args


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

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    _ = InteractiveSession(config=config)

    # open session
    if args.task is not None:
        module_name = '.'.join(['models',
                                args.task,
                                args.dnn_type])
    else:
        module_name = '.'.join(['models',
                                args.dnn_type])
    dnn_module = __import__(module_name,
                            fromlist=[args.dnn_type.upper()])
    dnn = getattr(dnn_module,
                  args.dnn_type.upper())(args)

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
    # dnn.visualize_results(args.iteration-1)


if __name__ == '__main__':
    main()
