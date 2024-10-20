from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Optional

from ml_soln.common.sagemaker_utils import sm_utils

__FILE_PATH = Path(__file__)

COMMON_DIR = __FILE_PATH.parent
ML_SOLN_DIR = COMMON_DIR.parent
PACKAGES_DIR = ML_SOLN_DIR.parent
ROOT_DIR = PACKAGES_DIR.parent


@dataclass
class Paths:
    package_name: str
    package_dir: Path
    input_dir: Optional[Path]
    model_dir: Path
    output_data_dir: Path
    output_intermediate_dir: Path
    predictions_dir: Path

    def clone(self, job_name: str):
        return self.for_package_name(package_name=self.package_name,
                                     job_name=job_name)

    @classmethod
    def for_package_name(cls,
                         package_name: str,
                         job_name: str = sm_utils.job_name,
                         is_sagemaker: bool = sm_utils.is_sagemaker):
        if is_sagemaker:
            return cls.sagemaker_paths(package_name)
        else:
            return cls.local_paths(package_name, job_name)

    @classmethod
    @cache
    def sagemaker_paths(cls, package_name: str):
        """
        Example folder structure when running in docker container:
            opt/
            └── ml/
                ├── input/
                │   ├── data/
                │   │   └── train/
                │   │       ├── kaggle_dataset
                │   │       ├── sample_submission.csv
                │   │       ├── test.csv
                │   │       └── train.csv
                │   └── config
                ├── output/
                │   ├── data/
                │   │   ├── history.pkl
                │   │   └── meta.json
                │   └── intermediate
                ├── model/
                │   └── model.pkl
                └── code/
                    └── entrypoint.py
        """
        package_dir = cls.make_package_dir(package_name)
        env = sm_utils.sagemaker_env
        output_data_dir = Path(env.output_data_dir)
        input_dir = env.channel_input_dirs.get('train')
        input_dir = Path(input_dir) if input_dir else None
        return Paths(
            package_name=package_name,
            package_dir=package_dir,
            input_dir=input_dir,
            model_dir=Path(env.model_dir),
            output_data_dir=output_data_dir,
            output_intermediate_dir=Path(env.output_intermediate_dir),
            predictions_dir=output_data_dir / 'predictions',
        )

    @classmethod
    @cache
    def local_paths(cls,
                    package_name: str,
                    job_name: str = sm_utils.job_name):
        """
        Example local folder structure:
            data/
            ├── input/
            │   └── kaggle_dataset/
            │       ├── sample_submission.csv
            │       ├── test.csv
            │       └── train.csv
            └── jobs/
                ├── local_2024-09-06T11:50:21/
                │   ├── data/
                │   │   ├── history.pkl
                │   │   └── meta.json
                │   └── models/
                │       └── model.pkl
                └── local_2024-09-05T01:30:13/
                    ├── data/
                    │   ├── history.pkl
                    │   └── meta.json
                    └── models/
                        └── model.pkl
        """

        package_dir = cls.make_package_dir(package_name)
        data_dir = package_dir / 'data'
        job_root = data_dir / 'jobs' / job_name
        output_data_dir = job_root / 'data'
        return Paths(
            package_name=package_name,
            package_dir=package_dir,
            input_dir=data_dir / 'input',
            model_dir=job_root / 'models',
            output_data_dir=output_data_dir,
            output_intermediate_dir=job_root / 'intermediate',
            predictions_dir=output_data_dir / 'predictions',
        )

    @staticmethod
    def make_package_dir(package_name: str):
        package_name = package_name.split('.')[-1]
        return ML_SOLN_DIR / package_name
