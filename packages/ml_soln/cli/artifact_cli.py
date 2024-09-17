import argparse

from ml_soln.common.paths import Paths
from ml_soln.common.trainers import PACKAGES
from ml_soln.sagemaker_ops.artifact_utils import ArtifactUtils


def pull_artifacts(args):
    paths = Paths.for_package_name(package_name=args.package,
                                   job_name=args.job,
                                   is_sagemaker=False)
    artifact_utils = ArtifactUtils(paths)
    artifact_utils.fetch_training_job_artifacts(args.job)

def push_training_data(args):
    paths = Paths.for_package_name(package_name=args.package,
                                   is_sagemaker=False)
    artifact_utils = ArtifactUtils(paths)
    artifact_utils.push_training_artifacts()

def add_arguments(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(title='operation', dest='docker_operation')

    # pull training artifacts
    build_parser = subparsers.add_parser('pull',
                                         help='Download training job artifacts from S3 to the standard local location')
    build_parser.add_argument('package',
                              type=str,
                              choices=PACKAGES,
                              help='Name of the package for which to download download artifacts')
    build_parser.add_argument('job',
                              type=str,
                              help='Name of the sagemaker job for which to pull output artifacts from S3')
    build_parser.set_defaults(func=pull_artifacts)

    # push training data
    publish_parser = subparsers.add_parser('push_train',
                                           help='Push training data to S3 to use as input channel in sagemaker jobs')
    publish_parser.add_argument('package',
                                type=str,
                                choices=PACKAGES,
                                help='Name of the package for which to push training artifacts to S3')
    publish_parser.set_defaults(func=push_training_data)

def main():
    parser = argparse.ArgumentParser(prog='mlops_artifacts',
                                     description='CRUD artifacts in S3 associated with training jobs')
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
