from pathlib import Path

from common.paths import CompetitionPaths

__FILE_PATH = Path(__file__)
__HOUSING_PRICES_DIR = __FILE_PATH.parent

paths = CompetitionPaths(__HOUSING_PRICES_DIR)
