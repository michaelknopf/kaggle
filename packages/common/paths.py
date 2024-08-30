from dataclasses import dataclass
from functools import cache
from pathlib import Path

from common.sagemaker_utils import sm_utils

__FILE_PATH = Path(__file__)

ROOT_DIR = __FILE_PATH.parents[2]
PACKAGES_DIR = __FILE_PATH.parents[1]
COMMON_DIR = __FILE_PATH.parent

@dataclass
class CompetitionPaths:
    root_dir: Path
    package_dir: Path
    data_dir: Path
    model_repo_dir: Path
    model_selection_dir: Path
    submissions_dir: Path

@cache
def competition_paths_for_package_name(package_name: str):
    if sm_utils.is_sagemaker:
        return _sagemaker_paths(package_name)
    else:
        return _local_paths(package_name)

def _local_paths(package_name: str):
    package_dir = PACKAGES_DIR / package_name
    return CompetitionPaths(
        root_dir=ROOT_DIR,
        package_dir=package_dir,
        data_dir=package_dir / 'data',
        model_repo_dir=package_dir / 'saved_models',
        model_selection_dir=package_dir / 'model_selection',
        submissions_dir=package_dir / 'submissions',
    )

def _sagemaker_paths(package_name: str):
    package_dir = PACKAGES_DIR / package_name
    env = sm_utils.sagemaker_environment
    output_dir = Path(env.output_dir)
    return CompetitionPaths(
        root_dir=ROOT_DIR,
        package_dir=package_dir,
        data_dir=Path(env.channel_input_dirs['train']),
        model_repo_dir=Path(env.model_dir),
        model_selection_dir=output_dir / 'model_selection',
        submissions_dir=output_dir / 'submissions',
    )
