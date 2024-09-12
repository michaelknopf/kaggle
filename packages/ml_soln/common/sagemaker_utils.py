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
        return self._environ_bool('IS_SAGEMAKER')

    @cached_property
    def enable_tensorflow_debugging(self):
        return self._environ_bool('ENABLE_TF_DEBUG')

    @cached_property
    def model_name(self):
        return os.environ.get('TRAINER_NAME')

    @cached_property
    def job_name(self):
        if self.is_sagemaker:
            return sm_utils.sagemaker_env.job_name
        else:
            return f'local_{self.start_time}'

    @cached_property
    def hyperparams(self):
        if self.is_sagemaker:
            return sm_utils.sagemaker_env.hyperparameters
        else:
            return {}

    @staticmethod
    def _environ_bool(name: str):
        return os.environ.get(name, '').lower() == 'true'

sm_utils = SagemakerUtils()
