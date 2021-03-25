import logging
import sys

logger = logging.getLogger('corgie')


def configure_logger(verbose):
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logger.addHandler(ch)
