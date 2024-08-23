import argparse
import os.path

from common.kaggle_facade import KaggleFacade
from common.paths import competition_paths_for_package_name
from common.manifest import get_manifest

def main():
    parser = argparse.ArgumentParser(prog='kaggle_ops',
                                     description='Performs common high-level Kaggle operations')
    subparsers = parser.add_subparsers(dest='command', required=True)

    competition_names = list(get_manifest().list_competitions())

    # pull_data command
    parser_pull_data = subparsers.add_parser('pull_data', help='Pull data for a competition')
    parser_pull_data.add_argument('competition',
                                  type=str,
                                  choices=competition_names,
                                  help='Name/ID of the Kaggle competition')

    # submit command
    parser_submit = subparsers.add_parser('submit', help='Submit a solution to a competition')
    parser_submit.add_argument('competition',
                               type=str,
                               choices=competition_names,
                               help='Name/ID of the Kaggle competition')
    parser_submit.add_argument('--filename', type=str, help='Name of the file to submit')
    parser_submit.add_argument('--message', type=str, help='Message to annotate submission')

    args = parser.parse_args()

    if args.command == 'pull_data':
        pull_data(args.competition)
    elif args.command == 'submit':
        submit(args.competition, args.filename, args.message)

def pull_data(competition_name):
    paths = competition_paths_for_package_name(competition_name=competition_name)
    kaggle = KaggleFacade(competition=competition_name, paths=paths)
    kaggle.download_data()

def submit(competition_name, filename, message):
    paths = competition_paths_for_package_name(competition_name=competition_name)
    if not filename:
        file_path = find_latest_submission(paths)
    else:
        file_path = paths.SUBMISSIONS_DIR / filename

    kaggle = KaggleFacade(competition=competition_name, paths=paths)
    kaggle.submit_predictions(file_path, message)

def find_latest_submission(paths):
    submission_files = list_file_names(paths.SUBMISSIONS_DIR)
    return max(submission_files, key=lambda filename: os.path.getmtime(paths.SUBMISSIONS_DIR / filename))

def list_file_names(dir_path):
    for path, dirs, filenames in dir_path.walk():
        for filename in filenames:
            yield filename

if __name__ == '__main__':
    main()
