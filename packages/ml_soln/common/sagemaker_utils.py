import os
from datetime import datetime
from functools import cached_property

from sagemaker_training.environment import Environment


class SagemakerUtils:

    start_time = datetime.now().replace(microsecond=0).isoformat()

    @cached_property
    def sagemaker_env(self):
        return Environment()

    @cached_property
    def is_sagemaker(self):
        return os.environ.get('IS_SAGEMAKER', '').lower() == 'true'

    @cached_property
    def model_name(self):
        return os.environ.get('TRAINER_NAME')

    @cached_property
    def job_name(self):
        if self.is_sagemaker:
            return sm_utils.sagemaker_env.job_name
        else:
            return f'local_{self.start_time}'

sm_utils = SagemakerUtils()
