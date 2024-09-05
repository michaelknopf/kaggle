from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path

from ml_soln.common.sagemaker_utils import sm_utils

__FILE_PATH = Path(__file__)

COMMON_DIR = __FILE_PATH.parent
ML_SOLN_DIR = COMMON_DIR.parent
PACKAGES_DIR = ML_SOLN_DIR.parent
ROOT_DIR = PACKAGES_DIR.parent

@dataclass
class Paths:
    package_dir: Path
    input_dir: Path
    output_dir: Path
    predictions_dir: Path

    @staticmethod
    def _make_job_output_dir(output_dir, job_name: str):
        return output_dir / 'training_jobs' / job_name

    def make_job_output_dir(self, job_name: str):
        return self._make_job_output_dir(self.output_dir, job_name)

    @cached_property
    def job_output_dir(self):
        return self.make_job_output_dir(sm_utils.job_name)

@cache
def paths_for_package_name(package_name: str):
    package_name = package_name.split('.')[-1]
    package_dir = ML_SOLN_DIR / package_name
    if sm_utils.is_sagemaker:
        env = sm_utils.sagemaker_env
        output_dir = Path(env.output_dir)
        input_dir = Path(env.channel_input_dirs['train'])
        predictions_dir = output_dir / 'predictions',
    else:
        data_dir = package_dir / 'data'
        output_dir = data_dir / 'output'
        input_dir = data_dir / 'input'
        predictions_dir = data_dir / 'predictions'
    return Paths(
        package_dir=package_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        predictions_dir=predictions_dir,
    )
