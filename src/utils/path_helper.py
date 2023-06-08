"""
    Path helper functions
"""
import os


def check_folder(log_dir):
    """
        check if directory does not exist,
        make it.

        params:

            log_dir: str
                directory to check
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
