import json

import boto3
import sagemaker
from sagemaker.estimator import Estimator

from common.paths import ROOT_DIR

# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# image = '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.16.2-gpu-py310-cu123-ubuntu20.04-sagemaker'
# instance = 'ml.g4dn.xlarge'
# @remote(
#     image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.16.2-gpu-py310-cu123-ubuntu20.04-sagemaker',
#     instance_type='ml.m5.xlarge',
#     # instance_type='ml.g4dn.xlarge',
#     # dependencies=''
#     job_name_prefix='mnist',
#     role='arn:aws:iam::394547497655:role/service-role/AmazonSageMaker-ExecutionRole-20240306T211248',
#     s3_root_uri='s3://sagemaker-us-west-2-394547497655/remote-function/mnist',
#     sagemaker_session=sagemaker_session,
#     include_local_workdir=True,
#     custom_file_filter=CustomFileFilter(
#         ignore_name_patterns=[
#             ".venv/*"
#             "*.ipynb",
#             "data",
#         ]
#     )
# )

with open(ROOT_DIR / 'private/aws_creds.json') as f:
    aws_creds = json.load(f)

boto_session = boto3.Session(
    aws_access_key_id=aws_creds['key'],
    aws_secret_access_key=aws_creds['secret'],
    region_name='us-west-2',
)
# sagemaker_session = sagemaker.session.Session(boto_session=boto_session)
sagemaker_session = sagemaker.local.LocalSession(boto_session=boto_session)

IMAGE_URI = 'kaggle-trainer:latest'
# role = get_execution_role(sagemaker_session=sagemaker_session)
role = 'arn:aws:iam::394547497655:role/service-role/AmazonSageMaker-ExecutionRole-20240306T211248'
training_data_path = 's3://sagemaker-us-west-2-394547497655/training_data/digit_recognizer'
output_path = 's3://sagemaker-us-west-2-394547497655/training_jobs/digit_recognizer'

# see https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html#byoc-training-step5
# see https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html
estimator = Estimator(
    sagemaker_session=sagemaker_session,
    image_uri=IMAGE_URI,
    role=role,
    base_job_name='mnist',
    instance_count=1,
    output_path=output_path,
    # instance_type='ml.m5.xlarge',
    instance_type='local',
    environment={
        'IS_SAGEMAKER': 'true',
        'MODEL_NAME': 'digit_recognizer',
    },
    hyperparameters={},
    # entry_point='',
    # metric_definitions=[],
    # checkpoint_s3_uri='',
    # checkpoint_local_path='',
)

estimator.fit(
    inputs={
        'train': training_data_path
    }
)
