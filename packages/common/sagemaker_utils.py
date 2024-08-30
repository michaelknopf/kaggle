import os
from functools import cache

from sagemaker_training.environment import Environment


@cache
def sagemaker_environment():
    return Environment()

@cache
def is_sagemaker():
    return os.environ.get('IS_SAGEMAKER', '').lower() == 'true'
