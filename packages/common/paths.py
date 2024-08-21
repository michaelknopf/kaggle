from pathlib import Path

__FILE_DIR = Path(__file__)

ROOT_DIR = __FILE_DIR.parents[2]
COMMON_DIR = __FILE_DIR.parents[0]

class CompetitionPaths:

    def __init__(self, package_dir):
        self.ROOT_DIR = ROOT_DIR
        self.PACKAGE_DIR = package_dir
        self.DATA_DIR = package_dir / 'data'
        self.MODEL_REPO_DIR = package_dir / 'saved_models'
        self.MODEL_SELECTION_DIR = package_dir / 'model_selection'
        self.SUBMISSIONS_DIR = package_dir / 'submissions'
