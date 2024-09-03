"""
See https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
See https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html#byoc-training-step2
See https://docs.aws.amazon.com/sagemaker/latest/dg/prebuilt-containers-extend.html#prebuilt-containers-extend-tutorial
"""
import argparse
import logging
from typing import Dict, Callable

from sagemaker_training.logging_config import configure_logger

configure_logger(logging.INFO)
logger = logging.getLogger(__name__)


def train_digit_recognizer():
    from ml_soln.digit_recognizer import ctx
    model = ctx().model.model
    history = ctx().trainer.train()
    ctx().model_persistence.save_model(model, history)


TRAINERS: Dict[str, Callable[[], None]] = {
    'digit_recognizer': train_digit_recognizer
}


def main(model_name):
    logger.info(f'Training for model: {model_name}')
    trainer = TRAINERS.get(model_name)
    if not trainer:
        raise ValueError(f'No trainer found for model {model_name}')
    trainer()


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
