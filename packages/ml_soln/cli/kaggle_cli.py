import argparse
import os.path

from ml_soln.kaggle.kaggle_facade import KaggleFacade
from ml_soln.common.paths import paths_for_package_name
from ml_soln.common.manifest import get_manifest


def pull_data(args):
    paths = paths_for_package_name(competition_name=args.competition)
    kaggle = KaggleFacade(competition=args.competition, paths=paths)
    kaggle.download_data()

def submit(args):
    paths = paths_for_package_name(competition_name=args.competition)
    if not args.filename:
        file_path = find_latest_submission(paths)
    else:
        file_path = paths.predictions_dir / args.filename

    kaggle = KaggleFacade(competition=args.competition, paths=paths)
    kaggle.submit_predictions(file_path, args.message)

def find_latest_submission(paths):
    submission_files = list_file_names(paths.predictions_dir)
    return max(submission_files, key=lambda filename: os.path.getmtime(paths.predictions_dir / filename))

def list_file_names(dir_path):
    for path, dirs, filenames in dir_path.walk():
        for filename in filenames:
            yield filename

def add_arguments(parser: argparse.ArgumentParser = None):
    subparsers = parser.add_subparsers(dest='command', required=True)

    competition_names = list(get_manifest().list_competitions())

    # pull_data command
    parser_pull_data = subparsers.add_parser('pull_data', help='Pull data for a competition')
    parser_pull_data.add_argument('competition',
                                  type=str,
                                  choices=competition_names,
                                  help='Name/ID of the Kaggle competition')
    parser_pull_data.set_defaults(func=pull_data)

    # submit command
    parser_submit = subparsers.add_parser('submit', help='Submit a solution to a competition')
    parser_submit.add_argument('competition',
                               type=str,
                               choices=competition_names,
                               help='Name/ID of the Kaggle competition')
    parser_submit.add_argument('--filename', type=str, help='Name of the file to submit')
    parser_submit.add_argument('--message', type=str, help='Message to annotate submission')
    parser_submit.set_defaults(func=submit)

def main():
    parser = argparse.ArgumentParser(prog='kaggle_ops',
                                     description='Performs common high-level Kaggle operations')
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
