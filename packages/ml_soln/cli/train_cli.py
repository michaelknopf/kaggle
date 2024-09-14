import argparse
import json

from sagemaker.estimator import Estimator
from ml_soln.sagemaker_ops.aws_context import aws_context, config
from ml_soln.sagemaker_ops.train_entrypoint import TRAINERS

def run(args):
    if args.image:
        image_uri = args.image
    elif args.local:
        image_uri = config().aws.local_image_uri
    else:
        image_uri = aws_context.ecr_image_uri

    if args.job_name:
        base_job_name = args.job_name
    else:
        base_job_name = args.trainer
        # the job name must satisfy this regex:
        # ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}
        base_job_name = base_job_name.replace('_', '-')

    if args.local:
        if args.instance_type:
            raise Exception("Cannot pass 'instance_type' when using local mode")
        instance_type = 'local'
    elif args.instance_type:
        instance_type = args.instance_type
    else:
        instance_type = config().aws.default_instance_type

    env_variables = {
        'IS_SAGEMAKER': 'true',
        'TRAINER_NAME': args.trainer,
    }
    env_variables.update(args.env)

    # TODO: deep merge, do not mutate
    hyperparams = config().training.hyperparams
    hyperparams.update(args.hyper_params)

    if args.local:
        sagemaker_session = aws_context.local_session
    else:
        sagemaker_session = aws_context.sagemaker_session

    # see https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html#byoc-training-step5
    # see https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html
    estimator = Estimator(
        sagemaker_session=sagemaker_session,
        image_uri=image_uri,
        role=config().aws.execution_role_arn,
        base_job_name=base_job_name,
        instance_count=args.instance_count,
        output_path=f'{aws_context.output_base_uri}/{args.trainer}',
        instance_type=instance_type,
        environment=env_variables,
        hyperparameters=hyperparams,
        metric_definitions=[],
        checkpoint_s3_uri=None,
        checkpoint_local_path=None,
    )

    estimator.fit(
        inputs={
            'train': f'{aws_context.training_data_base_uri}/{args.trainer}'
        }
    )


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('trainer',
                        type=str,
                        choices=list(sorted(TRAINERS.keys())),
                        help='Name of the trainer to execute')
    parser.add_argument('--local',
                        action='store_true',
                        default=False,
                        help='Run the sagemaker container locally')
    parser.add_argument('--image',
                        type=str,
                        help='The URI to an image in ECR, or to a local image.')
    parser.add_argument('--instance-count',
                        type=int,
                        help='The number of parallel instances to run',
                        default=1)
    parser.add_argument('--instance-type',
                        type=str,
                        help='The name of a sagemaker-compatible instance type')
    parser.add_argument('--job-name',
                        type=str,
                        help='The base name (prefix) of the sagemaker job')
    parser.add_argument('--train-data-uri',
                        type=str,
                        help='The full S3/file URI of the training data.')
    parser.add_argument('--env',
                        type=json.loads,
                        default={},
                        help='A JSON of environment variables')
    parser.add_argument('--hyper-params',
                        type=json.loads,
                        default={},
                        help='A JSON of hyper-parameters')
    parser.set_defaults(func=run)

def main():
    parser = argparse.ArgumentParser(prog='mlops_train',
                                     description='Train ML models in a sagemaker runtime')
    add_arguments(parser)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
