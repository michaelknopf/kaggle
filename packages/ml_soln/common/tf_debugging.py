import logging

import tensorflow as tf
from ml_soln.common.sagemaker_utils import sm_utils

logger = logging.getLogger(__name__)


def init():
    if sm_utils.enable_tensorflow_debugging:
        enable_logging()
        log_physical_devices()

def enable_logging():
    logger.info('Enabling device placement logging')
    tf.debugging.set_log_device_placement(True)

def log_physical_devices():
    logger.info('Physical devices: %s', tf.config.list_physical_devices())
