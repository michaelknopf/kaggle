from functools import cache
from pathlib import Path

__FILE_PATH = Path(__file__)

ROOT_DIR = __FILE_PATH.parents[2]
PACKAGES_DIR = __FILE_PATH.parents[1]
COMMON_DIR = __FILE_PATH.parent

class CompetitionPaths:

    def __init__(self, package_dir: Path):
        self.ROOT_DIR = ROOT_DIR
        self.PACKAGE_DIR = package_dir
        self.DATA_DIR = package_dir / 'data'
        self.MODEL_REPO_DIR = package_dir / 'saved_models'
        self.MODEL_SELECTION_DIR = package_dir / 'model_selection'
        self.SUBMISSIONS_DIR = package_dir / 'submissions'

@cache
def competition_paths_for_package_name(package_name: str):
    return CompetitionPaths(PACKAGES_DIR / package_name)
