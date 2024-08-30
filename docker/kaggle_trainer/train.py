"""
See https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
See https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html#byoc-training-step2
See https://docs.aws.amazon.com/sagemaker/latest/dg/prebuilt-containers-extend.html#prebuilt-containers-extend-tutorial
"""

import logging

from sagemaker_training.logging_config import configure_logger

from ml_soln.common.sagemaker_utils import sm_utils

configure_logger(logging.INFO)
logger = logging.getLogger(__name__)

def main(model_name=sm_utils.model_name):
    logger.info(f'Training for model: {model_name}')
    if model_name == 'digit_recognizer':
        from ml_soln.digit_recognizer import ctx
        model = ctx().model.model
        history = ctx().trainer.train()
        ctx().model_persistence.save_model(model, history)

if __name__ == '__main__':
    main('digit_recognizer')
