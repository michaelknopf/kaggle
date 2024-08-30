"""
See https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
See https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html#byoc-training-step2
See https://docs.aws.amazon.com/sagemaker/latest/dg/prebuilt-containers-extend.html#prebuilt-containers-extend-tutorial
"""

import logging
import os

from sagemaker_training.environment import Environment

logger = logging.getLogger(__name__)

logger.info("Loaded training script")
logger.info(f"Hyperparameters: {os.environ['SM_HPS']}")

env = Environment()

def main():
    env.output_dir

if __name__ == '__main__':
    main()
