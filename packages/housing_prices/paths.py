from pathlib import Path

from common.paths import CompetitionPaths

__FILE_DIR = Path(__file__)
__HOUSING_PRICES_DIR = __FILE_DIR.parents[0]

paths = CompetitionPaths(__HOUSING_PRICES_DIR)
