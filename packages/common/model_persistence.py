import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from housing_prices.paths import paths
import pickle
import json


@dataclass
class ModelPersistenceMeta:
    timestamp: str
    git_commit_hash: str
    note: str


def persist_model(model, note=''):
    meta = ModelPersistenceMeta(timestamp=datetime.now().replace(microsecond=0).isoformat(),
                                git_commit_hash=_get_git_commit_hash(),
                                note=note)
    paths.MODEL_REPO_DIR.mkdir(exist_ok=True)
    with open(paths.MODEL_REPO_DIR / f'{meta.timestamp}_meta.json', 'w') as f:
        json.dump(asdict(meta), f, indent=2)
    with open(paths.MODEL_REPO_DIR / f'{meta.timestamp}.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=5)


def load_model(file_name: str):
    if not file_name.endswith('.pkl'):
        file_name += '.pkl'
    with open(paths.MODEL_REPO_DIR / f'{file_name}', 'rb') as f:
        return pickle.load(f)


def _get_git_commit_hash():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                            capture_output=True,
                            text=True,
                            check=True)
    return result.stdout.strip()
