import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime

from keras.src.callbacks import History

from common.paths import CompetitionPaths
from common.sagemaker_utils import sm_utils
import pickle
import json


@dataclass
class ModelPersistenceMeta:
    timestamp: str
    git_commit_hash: str
    note: str

class ModelPersistence:

    def __init__(self,
                 paths: CompetitionPaths,
                 commit_hash=None):
        self.paths = paths
        if not commit_hash and not sm_utils.is_sagemaker:
            commit_hash = _get_git_commit_hash()
        self.commit_hash = commit_hash

    def save_model(self,
                   model,
                   history: History = None,
                   note=''):
        meta = ModelPersistenceMeta(timestamp=datetime.now().replace(microsecond=0).isoformat(),
                                    git_commit_hash=self.commit_hash,
                                    note=note)
        self.paths.model_repo_dir.mkdir(exist_ok=True)
        with open(self.paths.model_repo_dir / f'model_{meta.timestamp}_meta.json', 'w') as f:
            json.dump(asdict(meta), f, indent=2)
        with open(self.paths.model_repo_dir / f'model_{meta.timestamp}.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=5)
        if history:
            with open(self.paths.model_repo_dir / f'history_{meta.timestamp}.pkl', 'wb') as f:
                pickle.dump(model, f, protocol=5)

    def load_model(self, file_name: str):
        if not file_name.endswith('.pkl'):
            file_name += '.pkl'
        with open(self.paths.model_repo_dir / f'{file_name}', 'rb') as f:
            return pickle.load(f)

def _get_git_commit_hash():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                            capture_output=True,
                            text=True,
                            check=True)
    return result.stdout.strip()
