import argparse

from ml_soln.cli import train_cli, docker_cli, kaggle_cli

def main():
    parser = argparse.ArgumentParser(prog='sage',
                                     description='Sagemaker operations')
    subparsers = parser.add_subparsers(title='commands', required=True)

    train_parser = subparsers.add_parser('train', help='Train ML models in a sagemaker runtime')
    train_cli.add_arguments(train_parser)

    docker_parser = subparsers.add_parser('docker', help='Operate on the docker image(s) used in sagemaker jobs')
    docker_cli.add_arguments(docker_parser)

    kaggle_parser = subparsers.add_parser('kaggle', help='Run common high-level Kaggle API flows')
    kaggle_cli.add_arguments(kaggle_parser)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
