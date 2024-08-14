from pathlib import Path

__FILE_DIR = Path(__file__)

ROOT_DIR = __FILE_DIR.parents[1]
HOUSING_PRICES_DIR = __FILE_DIR.parents[0]
DATA_DIR = HOUSING_PRICES_DIR / 'data'
MODEL_REPO_DIR = HOUSING_PRICES_DIR / 'saved_models'
MODEL_SELECTION_DIR = HOUSING_PRICES_DIR / 'model_selection'
