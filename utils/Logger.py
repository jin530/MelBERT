import os
from time import strftime
import logging


def make_log_dir(log_dir):
    """
    Generate directory path to log

    :param log_dir:

    :return:
    """
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dirs = os.listdir(log_dir)
    if len(log_dirs) == 0:
        idx = 0
    else:
        idx_list = sorted([int(d.split("_")[0]) for d in log_dirs])
        idx = idx_list[-1] + 1

    cur_log_dir = "%d_%s" % (idx, strftime("%Y%m%d-%H%M"))
    full_log_dir = os.path.join(log_dir, cur_log_dir)
    if not os.path.exists(full_log_dir):
        os.mkdir(full_log_dir)

    return full_log_dir


class Logger:
    def __init__(self, log_dir):
        log_file_format = "[%(lineno)d]%(asctime)s: %(message)s"
        log_console_format = "%(message)s"

        # Main logger
        self.log_dir = log_dir

        self.logger = logging.getLogger(log_dir)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_console_format))

        file_handler = logging.FileHandler(os.path.join(log_dir, "experiments.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_file_format))

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def info(self, msg):
        self.logger.info(msg)

    def close(self):
        for handle in self.logger.handlers[:]:
            self.logger.removeHandler(handle)
        logging.shutdown()


def setup_logger(log_dir):
    log_file_format = "[%(lineno)d]%(asctime)s: %(message)s"
    log_console_format = "%(message)s"

    # Main logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_console_format))

    file_handler = logging.FileHandler(os.path.join(log_dir, "experiments.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_file_format))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
