from dataclasses import dataclass, field
from functools import cache
from functools import cached_property
from typing import Dict

import boto3
import sagemaker
import yaml
from mypy_boto3_s3 import S3Client
from mypy_boto3_sagemaker import SageMakerClient
from mypy_boto3_sts import STSClient

from ml_soln.common.dataclass_utils import DictClassMixin
from ml_soln.common.paths import ROOT_DIR

IMAGE_NAME = 'ml-soln-trainer:latest'

@dataclass
class Config(DictClassMixin):
    aws: 'AwsConfig'
    training: 'TrainingConfig' = field(default_factory=lambda: TrainingConfig())

@dataclass
class AwsConfig(DictClassMixin):
    profile: str
    execution_role_arn: str
    sagemaker_root_bucket: str
    ecr_image_uri: str
    local_image_uri: str = IMAGE_NAME
    default_instance_type: str = 'ml.m5.xlarge'

@dataclass
class TrainingConfig(DictClassMixin):
    hyperparams: Dict = field(default_factory=lambda: {})
    training_data_path: str = 'training_data'
    output_path: str = 'training_jobs'

    def __post_init__(self):
        if self.training_data_path:
            self.training_data_path = self.training_data_path.strip('/')
        if self.output_path:
            self.output_path = self.output_path.strip('/')

@cache
def config() -> Config:
    config_path = ROOT_DIR / 'private' / 'config.yml'
    if config_path.exists():
        with open(config_path) as f:
            d = yaml.safe_load(f)
    return Config.from_dict(d)

class Context:

    def __init__(self, boto_session=None):
        if not boto_session:
            boto_session = boto3.Session(profile_name=config().aws.profile)
        self.boto_session = boto_session

    @cached_property
    def sagemaker_session(self):
        return sagemaker.session.Session(boto_session=self.boto_session)

    @cached_property
    def local_session(self):
        return sagemaker.local.LocalSession(boto_session=self.boto_session)

    @property
    def region(self):
        return self.boto_session.region_name

    @cached_property
    def sts_client(self) -> STSClient:
        return self.boto_session.client('sts')

    @cached_property
    def s3_client(self) -> S3Client:
        return self.boto_session.client('s3')

    @cached_property
    def sagemaker_client(self) -> SageMakerClient:
        return self.boto_session.client('sagemaker')

    @cached_property
    def caller_identity(self) -> Dict[str, str]:
        return self.sts_client.get_caller_identity()

    @property
    def aws_account_id(self):
        return self.caller_identity['Account']

    @cached_property
    def ecr_domain(self):
        return f'{self.aws_account_id}.dkr.ecr.{self.region}.amazonaws.com'

    @cached_property
    def ecr_image_uri(self):
        if config().aws.ecr_image_uri:
            return config().aws.ecr_image_uri
        return f'{self.ecr_domain}/{IMAGE_NAME}'

    @cached_property
    def training_data_base_uri(self):
        return f's3://{config().aws.sagemaker_root_bucket}/{config().training.training_data_path}'

    @cached_property
    def output_base_uri(self):
        return f's3://{config().aws.sagemaker_root_bucket}/{config().training.output_path}'

aws_context = Context()
