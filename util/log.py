import logging
import os
import time


def log(path):
    current_path = os.getcwd()
    abs_path = os.path.join(current_path, path)
    if not os.path.exists(abs_path):
        try:
            os.makedirs(abs_path)
        except Exception:
            pass
    logger = logging.getLogger("LOG_INFO")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        "{0}/log_{1}.log".format(abs_path, time.strftime('%Y-%m-%d-%H-%M')))

    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info('create logger sucessfully!')
    return logger
