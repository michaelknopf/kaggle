import os
from functools import cached_property

from sagemaker_training.environment import Environment


class SagemakerUtils:

    @cached_property
    def sagemaker_environment(self):
        return Environment()

    @cached_property
    def is_sagemaker(self):
        return os.environ.get('IS_SAGEMAKER', '').lower() == 'true'

    @cached_property
    def model_name(self):
        return os.environ.get('MODEL_NAME')

sm_utils = SagemakerUtils()
