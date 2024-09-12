"""
See https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
See https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html#byoc-training-step2
See https://docs.aws.amazon.com/sagemaker/latest/dg/prebuilt-containers-extend.html#prebuilt-containers-extend-tutorial
"""
import argparse
import logging

from sagemaker_training.logging_config import configure_logger

from ml_soln.common.trainers import TRAINERS
from ml_soln.common import tf_debugging

configure_logger(logging.INFO)
logger = logging.getLogger(__name__)

tf_debugging.init()


def main(trainer_name):
    logger.info(f'Training for model: {trainer_name}')
    trainer = TRAINERS.get(trainer_name)
    if not trainer:
        raise ValueError(f'No trainer found for model {trainer_name}')
    trainer.func()


def cli():
    parser = argparse.ArgumentParser(prog='train',
                                     description='Train ML models')
    parser.add_argument('trainer',
                        type=str,
                        choices=list(sorted(TRAINERS.keys())),
                        help='Name of the trainer to execute')
    args = parser.parse_args()
    main(args.trainer)

if __name__ == '__main__':
    cli()
