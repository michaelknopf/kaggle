import logging
import os
import tarfile
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib import parse

import botocore.exceptions
from mypy_boto3_sagemaker.type_defs import DescribeTrainingJobResponseTypeDef

from ml_soln.common.paths import Paths
from ml_soln.common.trainers import TRAINERS
from ml_soln.sagemaker_ops.aws_context import aws_context, config

logger = logging.getLogger(__name__)

class ArtifactUtils:
    """
    S3 structure:

        sagemaker-{region}-{account_id}/
        └── training_jobs/
            └── digit_recognizer/
                └── digit-recognizer-2024-09-06-04-41-52-434/
                    └── output/
                        ├── model.tar.gz
                        └── output.tar.gz

    Local temp folder structure, after unzipping both .tar.gz files:

        temp_folder/
        ├── model.pkl
        └── output/
            ├── data/
            │   ├── meta.json
            │   └── history.pkl
            └── success
    """

    def __init__(self,
                 paths: Paths = None,
                 job_name: str = None,
                 job: DescribeTrainingJobResponseTypeDef = None):
        self._paths = paths
        self._job = job
        self._job_name = job_name

    @cached_property
    def paths(self):
        return self._paths or self._make_default_paths()

    def _make_default_paths(self):
        trainer_name = self.job['Environment']['TRAINER_NAME']
        trainer = TRAINERS[trainer_name]
        return Paths.local_paths(package_name=trainer.package_name,
                                 job_name=self.job['TrainingJobName'])

    @cached_property
    def job(self) -> DescribeTrainingJobResponseTypeDef:
        return self._job or aws_context.sagemaker_client.describe_training_job(TrainingJobName=self.job_name)

    @cached_property
    def job_name(self):
        return self._job_name or self.job['TrainingJobName']

    @cached_property
    def tar_to_dir(self):
        return {
            'output.tar.gz': self.paths.output_data_dir,
            'model.tar.gz': self.paths.model_dir,
        }

    def fetch_training_job_artifacts(self, job_name: str):
        job = aws_context.sagemaker_client.describe_training_job(TrainingJobName=job_name)
        s3_model_artifacts_path = job.get('ModelArtifacts', {}).get('S3ModelArtifacts')
        self.fetch_s3_artifacts(s3_model_artifacts_path)

    def fetch_s3_artifacts(self, s3_model_artifacts_path):
        bucket, path, _filename = self.parse_s3_uri(s3_model_artifacts_path)
        tarfile_names = ['output.tar.gz', 'model.tar.gz']

        with TemporaryDirectory() as dir_name:
            dir_path = Path(dir_name)
            for filename in tarfile_names:
                key = f'{path}/{filename}'
                file_path = dir_path / filename
                if self.head_object(bucket=bucket, key=key):
                    logger.info('Downloading s3://%s/%s to %s', bucket, key, dir_name)
                    aws_context.s3_client.download_file(Bucket=bucket, Key=key, Filename=str(file_path))
                    dest_folder = self.tar_to_dir[filename]
                    dest_folder.mkdir(exist_ok=True, parents=True)
                    logger.info('Extracting %s to %s', file_path, dest_folder)
                    with tarfile.open(file_path) as f:
                        f.extractall(dest_folder)
                else:
                    logger.info('s3://%s/%s does not exist. Skipping...', bucket, key)

    def push_training_artifacts(self):
        """
        Adapted from https://gist.github.com/feelinc/d1f541af4f31d09a2ec3
        """
        bucket = config().aws.sagemaker_root_bucket

        for root, dirs, files in os.walk(self.paths.input_dir):
            root_path = Path(root)
            for filename in files:
                # construct the full local path
                local_path = root_path / filename

                # construct the full S3 path
                training_data_path = Path(config().training.training_data_path)
                relative_path = local_path.relative_to(self.paths.input_dir)
                key = str(training_data_path / self.paths.package_name / relative_path)

                if self.head_object(bucket=bucket, key=key):
                    logger.info('Skipping since s3://%s/%s already exists', bucket, key)
                else:
                    logger.info('Uploading s3://%s/%s', bucket, key)
                    aws_context.s3_client.upload_file(local_path, bucket, key)

    @staticmethod
    def parse_s3_uri(uri):
        # see for reference:
        # https://github.com/aws/sagemaker-python-sdk/blob/7aa39f99d8f67e39c32fb2d441dc4e86814f3809/src/sagemaker/utils.py#L572
        url = parse.urlparse(uri)
        bucket = url.netloc
        key = url.path.lstrip("/")
        path = os.path.dirname(key)
        filename = os.path.basename(key)
        return bucket, path, filename

    @staticmethod
    def head_object(bucket, key):
        try:
            return aws_context.s3_client.head_object(Bucket=bucket, Key=key)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != '404':
                raise e
