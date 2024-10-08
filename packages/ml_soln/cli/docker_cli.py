import argparse
import os
import subprocess

BUILD = 'build'
PUBLISH = 'publish'

def run(args):

    # defer imports to improve CLI startup time
    from ml_soln.common.paths import ROOT_DIR

    if args.docker_operation == BUILD:
        script_filename = 'build.sh'
    elif args.docker_operation == PUBLISH:
        script_filename = 'publish.sh'
    else:
        raise ValueError(f'Unrecognized docker operation: {args.docker_operation}')
    script = str(ROOT_DIR / 'docker' / 'ml_soln_trainer' / script_filename)

    os.environ['ROOT_DIR'] = str(ROOT_DIR)
    subprocess.run(
        script,
        # this allows the CLI to be run from anywhere, since it doesn't rely on
        # git to determine the root directory
        env=os.environ,
        # raise exception on failure
        check=True,
        shell=True,
    )

def add_arguments(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(title='operation', dest='docker_operation', required=True)

    build_parser = subparsers.add_parser(BUILD, help='Build image')
    build_parser.set_defaults(func=run)

    publish_parser = subparsers.add_parser(PUBLISH, help='Push image to ECR')
    publish_parser.set_defaults(func=run)

def main():
    parser = argparse.ArgumentParser(prog='mlops_docker',
                                     description='Operate on the docker image(s) used in sagemaker jobs')
    add_arguments(parser)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
