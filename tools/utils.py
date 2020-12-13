import os
import sys
import logging


def configure_logger(exp_dir, exp_name):
    """
    Configure logger on given level. Logging will occur on standard
    output and in a log file saved in model_dir.
    """
    loglevel = "info"
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(loglevel))

    log_format = logging.Formatter('%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    file_handler = logging.FileHandler(os.path.join(exp_dir,
                                                    '{}.log'.format(os.path.basename(exp_name))))
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    file_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logging.root = logger
    logging.info("saving log, checkpoint and back up code in folder: {}".format(exp_dir))


def print_args(args):
    logging.info("==========       args      =============")
    for arg, content in args.__dict__.items():
        logging.info("{}:{}".format(arg, content))
    logging.info("==========     args END    =============")
    logging.info("\n")

